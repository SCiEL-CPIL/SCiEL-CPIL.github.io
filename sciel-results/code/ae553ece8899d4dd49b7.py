"""HP lattice protein folding on the cubic lattice.
Uses greedy initialization followed by Monte Carlo local search with
single-residue moves, chain breaks, and simulated annealing.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 318.0  # Safety margin before 330s kill


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D (fast path)."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _build_contact_map(sequence, chain):
    """Build a set of (i,j) pairs representing H-H contacts (i<j, |i-j|>1)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts


def _energy_from_contacts(contacts):
    """Energy is negative count of contacts."""
    return -len(contacts)


def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    contacts = _build_contact_map(sequence, chain)
    return -len(contacts)


def _delta_energy_move(sequence, chain, i, new_pos, occupied, contacts):
    """Compute energy change if residue i moves to new_pos.
    occupied: dict pos->index for all residues
    contacts: set of (i,j) H-H contact pairs
    Returns (delta_energy, new_occupied, new_contacts) or None if move invalid."""
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0, occupied, contacts
    if new_pos in occupied:
        return None
    
    is_h = (sequence[i] == "H")
    # Remove old contacts involving i
    removed = set()
    for (a, b) in list(contacts):
        if a == i or b == i:
            removed.add((a, b))
    # Remove old position from occupied
    del occupied[old_pos]
    
    # Add new position
    occupied[new_pos] = i
    
    # Check self-avoidance: new_pos must not be adjacent to i-1 or i+1 in a way that violates
    # Actually self-avoidance is just that new_pos is not occupied (checked above)
    # But we also need new_pos != old_pos (checked) and unit step from neighbors
    
    # Verify unit steps
    if i > 0:
        prev_pos = chain[i-1]
        dx = abs(new_pos[0] - prev_pos[0])
        dy = abs(new_pos[1] - prev_pos[1])
        dz = abs(new_pos[2] - prev_pos[2])
        if dx + dy + dz != 1:
            # Revert
            del occupied[new_pos]
            occupied[old_pos] = i
            return None
    if i < len(chain) - 1:
        next_pos = chain[i+1]
        dx = abs(new_pos[0] - next_pos[0])
        dy = abs(new_pos[1] - next_pos[1])
        dz = abs(new_pos[2] - next_pos[2])
        if dx + dy + dz != 1:
            del occupied[new_pos]
            occupied[old_pos] = i
            return None
    
    # Add new contacts involving i
    added = set()
    if is_h:
        x, y, z = new_pos
        for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            j = occupied.get(q)
            if j is not None and j != i and j != i-1 and j != i+1 and sequence[j] == "H":
                a, b = min(i, j), max(i, j)
                added.add((a, b))
    
    # Remove old contacts that are no longer valid (contacts from old_pos that involved i)
    # We already removed all contacts involving i, so we just need to add new ones
    # But we need to be careful: contacts not involving i are unchanged
    
    new_contacts = (contacts - removed) | added
    delta = -len(new_contacts) - (-len(contacts))
    
    return delta, occupied, new_contacts


def _grow(sequence, dimension, rng, temperature=1.0):
    """One randomized greedy chain growth with temperature control."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        x, y, z = chain[-1]
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbors for entropy bonus
            free = 0
            sx, sy, sz = site
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied:
                    free += 1
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _mc_local_search(sequence, chain, rng, time_budget=60.0):
    """Monte Carlo local search with single-residue moves and simulated annealing.
    Uses incremental energy updates for speed."""
    n = len(sequence)
    if n < 4:
        return list(chain), _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Build contact set and occupied map
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_contacts = _build_contact_map(sequence, current_chain)
    
    start_time = time.time()
    t0 = 3.0
    t_end = 0.3
    # Number of steps to perform
    # Estimate steps per second and set total steps
    total_steps = 0
    cooling_rate = 0.9997
    
    for step in range(1000000):  # Will break on time
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        if time.time() > _DEADLINE - 5:
            break
        
        # Temperature for this step
        frac = min(1.0, elapsed / time_budget)
        T = t0 * (t_end / t0) ** frac
        
        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try to move to a random neighbor
        x, y, z = old_pos
        candidates = []
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site == current_chain[i-1] or site == current_chain[i+1]:
                continue
            if site in current_occupied:
                continue
            candidates.append(site)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        
        # Compute delta energy
        # Remove old position
        del current_occupied[old_pos]
        current_occupied[new_pos] = i
        
        # Check validity: new_pos must not be in occupied (already checked)
        # and must maintain unit steps (checked by construction since we excluded i-1 and i+1 positions)
        # But we need to verify new_pos is not occupied by another residue
        # (already excluded above)
        
        # Compute delta
        is_h = (sequence[i] == "H")
        # Old contacts involving i
        old_c = 0
        for (a, b) in current_contacts:
            if a == i or b == i:
                old_c += 1
        
        # New contacts involving i
        new_c = 0
        if is_h:
            nx, ny, nz = new_pos
            for q in ((nx+1,ny,nz),(nx-1,ny,nz),(nx,ny+1,nz),(nx,ny-1,nz),(nx,ny,nz+1),(nx,ny,nz-1)):
                j = current_occupied.get(q)
                if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                    new_c += 1
        
        delta = -(new_c - old_c)
        
        # Accept/reject
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept
            # Update contacts
            # Remove old contacts involving i
            new_contacts = set()
            for (a, b) in current_contacts:
                if a != i and b != i:
                    new_contacts.add((a, b))
            # Add new contacts involving i
            if is_h:
                nx, ny, nz = new_pos
                for q in ((nx+1,ny,nz),(nx-1,ny,nz),(nx,ny+1,nz),(nx,ny-1,nz),(nx,ny,nz+1),(nx,ny,nz-1)):
                    j = current_occupied.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                        a, b = min(i, j), max(i, j)
                        new_contacts.add((a, b))
            
            current_contacts = new_contacts
            current_chain[i] = new_pos
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Reject: revert
            del current_occupied[new_pos]
            current_occupied[old_pos] = i
    
    return best_chain, best_energy


def _chain_break_rejoin(sequence, chain, rng, time_budget=15.0):
    """Try breaking the chain at a random point and regrowing the tail.
    This is a structural move that can escape local minima."""
    n = len(sequence)
    if n < 6:
        return list(chain), _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    start_time = time.time()
    
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 5:
            break
        
        # Pick a break point (not too close to ends)
        break_idx = rng.randint(3, n - 4)
        prefix = chain[:break_idx + 1]
        occupied = {p: i for i, p in enumerate(prefix)}
        
        # Regrow tail greedily with some randomness
        current_tail = [tuple(p) for p in prefix]
        success = True
        for i in range(break_idx + 1, n):
            x, y, z = current_tail[-1]
            options = []
            for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    sx, sy, sz = site
                    for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = 0
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q not in occupied:
                        free += 1
                score = gain + 0.05 * free + rng.random() * 0.8
                options.append((score, site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            current_tail.append(site)
        
        if not success:
            continue
        
        new_energy = _energy(sequence, current_tail)
        if new_energy < best_energy:
            best_chain = current_tail
            best_energy = new_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy init + MC local search + chain break/rejoin."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    remaining = _DEADLINE - time.time()
    
    # Phase 1: Multiple randomized greedy growths
    if n <= 50:
        num_initial = min(60, int(remaining * 1.5))
    elif n <= 100:
        num_initial = min(30, int(remaining * 0.8))
    else:
        num_initial = min(15, int(remaining * 0.4))
    
    num_initial = max(num_initial, 8)
    
    for trial in range(num_initial):
        if time.time() > _DEADLINE - 80:
            break
        
        # Vary temperature
        temp = 1.2 if trial < num_initial // 3 else (0.6 if trial < 2 * num_initial // 3 else 0.3)
        chain = _grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Phase 2: MC local search (main refinement)
    remaining = _DEADLINE - time.time() - 8
    if remaining > 15:
        mc_budget = min(remaining * 0.55, 120.0)
        best_chain, best_energy = _mc_local_search(
            sequence, best_chain, rng, time_budget=mc_budget
        )
    
    # Phase 3: Chain break and rejoin (structural escape)
    remaining = _DEADLINE - time.time() - 8
    if remaining > 10:
        cb_budget = min(remaining * 0.35, 40.0)
        improved_chain, improved_energy = _chain_break_rejoin(
            sequence, best_chain, rng, time_budget=cb_budget
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 4: Final MC polish
    remaining = _DEADLINE - time.time() - 5
    if remaining > 8:
        mc_budget2 = min(remaining * 0.8, 30.0)
        best_chain, best_energy = _mc_local_search(
            sequence, best_chain, rng, time_budget=mc_budget2
        )
    
    return [list(p) for p in best_chain]
