"""Rewritten program: simulated annealing on HP lattice folding."""
# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute hydrophobic contact energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _try_move(sequence, chain, occupied, i, rng):
    """
    Attempt a local move at position i. Returns (new_chain, new_occupied, delta_energy)
    or None if no valid move.
    Uses a simple move: rotate the segment from i to i+2 (if exists) by 90 degrees,
    or try moving a single residue to an adjacent free site.
    """
    n = len(sequence)
    if n < 3:
        return None
    
    # Try moving a single residue (not endpoints) to an adjacent free site
    if i < 1 or i > n - 2:
        return None
    
    current = chain[i]
    candidates = []
    for site in _neighbours(current):
        if site in occupied:
            continue
        # Check that moving doesn't create overlap with neighbors
        # The neighbors are chain[i-1] and chain[i+1], which are fixed
        # We just need site != chain[i-1] and site != chain[i+1] (already checked via occupied)
        candidates.append(site)
    
    if not candidates:
        return None
    
    new_site = rng.choice(candidates)
    
    # Compute delta energy
    # Remove contacts involving position i
    old_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(current):
            j = None
            for k in range(n):
                if chain[k] == q:
                    j = k
                    break
            if j is not None and j != i and sequence[j] == 'H':
                # Count if not adjacent along chain
                if abs(j - i) > 1:
                    old_contacts += 1
    
    # Add new contacts at new_site
    new_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(new_site):
            j = None
            for k in range(n):
                if chain[k] == q:
                    j = k
                    break
            if j is not None and j != i and sequence[j] == 'H':
                if abs(j - i) > 1:
                    new_contacts += 1
    
    delta = new_contacts - old_contacts  # energy change = -(new_contacts) - (-(old_contacts)) = old_contacts - new_contacts... wait
    # Energy = -contacts. So delta_E = E_new - E_old = -new_contacts - (-old_contacts) = old_contacts - new_contacts
    # But we want to minimize energy, so if delta < 0 it's better.
    delta_energy = old_contacts - new_contacts
    
    new_chain = list(chain)
    new_chain[i] = new_site
    new_occupied = dict(occupied)
    del new_occupied[current]
    new_occupied[new_site] = i
    
    return new_chain, new_occupied, delta_energy


def _try_pivot(sequence, chain, occupied, i, rng):
    """
    Pivot move: rotate the subchain from i+1 to end by 90 degrees around chain[i].
    """
    n = len(sequence)
    if i < 1 or i >= n - 1:
        return None
    
    pivot = chain[i]
    tail = chain[i+1:]
    
    # Determine the direction from pivot to chain[i+1]
    d = (tail[0][0] - pivot[0], tail[0][1] - pivot[1])
    
    # Rotate direction by 90 degrees clockwise or counterclockwise
    if rng.random() < 0.5:
        # Clockwise: (dx, dy) -> (dy, -dx)
        new_d = (d[1], -d[0])
    else:
        # Counterclockwise: (dx, dy) -> (-dy, dx)
        new_d = (-d[1], d[0])
    
    # Try to place the tail with the new initial direction
    # We need to check if the entire tail fits without overlap
    # This is complex, so let's try a simpler approach: just try to move chain[i+1]
    # Actually, let's implement a proper pivot by trying to reposition the whole tail.
    
    # For simplicity, let's just try moving chain[i+1] to the new direction
    new_pos = (pivot[0] + new_d[0], pivot[1] + new_d[1])
    
    if new_pos in occupied:
        return None
    
    # Check that new_pos is not chain[i-1]
    if new_pos == chain[i-1]:
        return None
    
    # Now we need to check if the rest of the tail can be placed.
    # This is hard to do without actually placing it. Let's try a simpler heuristic:
    # Just try the move and see if it's valid by checking all subsequent positions.
    # Actually, for a proper pivot, we'd need to re-simulate the tail. Let's skip this
    # and rely on single-residue moves and maybe some other moves.
    
    return None


def fold(sequence, dimension):
    """
    Simulated annealing to find low-energy conformations.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Initial conformation: simple spiral or straight line with some turns
    # Start with a straight line, then add some randomness
    rng = random.Random(42 + len(sequence))
    
    # Create initial chain: a simple "snake" pattern
    chain = []
    occupied = {}
    x, y = 0, 0
    chain.append((x, y))
    occupied[(x, y)] = 0
    
    # Simple initial folding: go right, down, left, down, right, etc.
    directions = [(1, 0), (0, 1), (-1, 0), (0, 1)]
    dir_idx = 0
    for i in range(1, n):
        # Occasionally change direction to create a more compact structure
        if i > 0 and i % 3 == 0:
            dir_idx = (dir_idx + 1) % 4
        
        dx, dy = directions[dir_idx]
        new_x, new_y = x + dx, y + dy
        new_pos = (new_x, new_y)
        
        # If blocked, try other directions
        if new_pos in occupied:
            placed = False
            for d_idx in range(4):
                if d_idx == dir_idx:
                    continue
                dx2, dy2 = directions[d_idx]
                nx2, ny2 = x + dx2, y + dy2
                if (nx2, ny2) not in occupied:
                    new_x, new_y = nx2, ny2
                    new_pos = (new_x, new_y)
                    dir_idx = d_idx
                    placed = True
                    break
            if not placed:
                # Just place it even if overlapping (will be fixed by SA)
                # Actually, we need a valid starting point. Let's use a straight line.
                break
        
        x, y = new_x, new_y
        chain.append(new_pos)
        occupied[new_pos] = i
    
    # If we couldn't place all residues, use a straight line
    if len(chain) < n:
        chain = [(i, 0) for i in range(n)]
        occupied = {p: i for i, p in enumerate(chain)}
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing
    t0 = 10.0
    t1 = 0.1
    num_iterations = 5000
    
    for it in range(num_iterations):
        # Temperature schedule
        t = t0 * (t1 / t0) ** (it / num_iterations)
        
        # Choose a random position to move
        i = rng.randint(1, n - 2)
        
        # Try a move
        result = _try_move(sequence, chain, occupied, i, rng)
        if result is None:
            continue
        
        new_chain, new_occupied, delta_energy = result
        
        # Accept or reject
        if delta_energy < 0:
            # Better, always accept
            chain = new_chain
            occupied = new_occupied
            current_energy += delta_energy
        else:
            # Worse, accept with probability exp(-delta/T)
            if t > 0:
                prob = math.exp(-delta_energy / t)
                if rng.random() < prob:
                    chain = new_chain
                    occupied = new_occupied
                    current_energy += delta_energy
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
    
    # Final local search: greedy improvement
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            best_local = None
            best_local_energy = current_energy
            for site in _neighbours(chain[i]):
                if site in occupied:
                    continue
                # Compute energy change
                old_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(chain[i]):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            old_contacts += 1
                
                new_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        # Find which residue is at q, excluding i
                        for k in range(n):
                            if k != i and chain[k] == q:
                                if sequence[k] == 'H' and abs(k - i) > 1:
                                    new_contacts += 1
                                break
                
                delta = old_contacts - new_contacts
                if delta < best_local_energy - current_energy:
                    best_local = site
                    best_local_energy = current_energy + delta
            
            if best_local is not None:
                # Make the move
                old_pos = chain[i]
                del occupied[old_pos]
                chain[i] = best_local
                occupied[best_local] = i
                current_energy = best_local_energy
                improved = True
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
