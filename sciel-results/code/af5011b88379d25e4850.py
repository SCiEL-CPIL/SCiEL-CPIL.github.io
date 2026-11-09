"""HP lattice protein folding: contact-aware memetic search with backbone re-fitting."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 118.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    h_positions = set()
    for i, c in enumerate(sequence):
        if c == "H":
            h_positions.add(i)
    contacts = 0
    for i, p in enumerate(chain):
        if i in h_positions:
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and j in h_positions:
                    contacts += 1
    return -contacts // 2


def _grow_compact(sequence, rng, compactness):
    """Greedy chain growth with a compactness parameter."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        current = chain[-1]
        x, y = current
        options = []
        for site in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if site in occupied:
                continue
            
            score = 0.0
            if sequence[i] == "H":
                sx, sy = site
                for q in ((sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            score += 2.0
                        elif j < i - 1:
                            score += 0.5
            free_count = sum(1 for q in ((site[0] + 1, site[1]), (site[0] - 1, site[1]), (site[0], site[1] + 1), (site[0], site[1] - 1)) if q not in occupied)
            score -= compactness * free_count * 0.3
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        best_site = options[0][1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True


def _apply_reversal(chain, i, j):
    """Attempt a reversal move."""
    n = len(chain)
    if i >= j:
        return None
    
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None
            
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None
            
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    if not _is_valid_chain(new_chain):
        return None
        
    return new_chain


def _apply_pivot(chain, i, angle):
    """Attempt a pivot move."""
    n = len(chain)
    if i <= 0 or i >= n - 2:
        return None
        
    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]
    
    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        rotated_tail.append((pivot[0] + x, pivot[1] + y))
        
    new_chain = prefix + rotated_tail
    
    if not _is_valid_chain(new_chain):
        return None
        
    return new_chain


def _contact_aware_local_search(sequence, chain, time_limit, rng, tabu_len=40):
    """
    Contact-aware local search with tabu list.
    Generates moves that specifically target H-H contact creation/destruction.
    Returns best_chain, best_energy.
    """
    n = len(chain)
    if n < 3:
        return chain[:], _energy(sequence, chain)
    
    current_chain = chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    h_positions = set(i for i, c in enumerate(sequence) if c == "H")
    
    # Build contact set: pairs (i,j) with i<j, both H, lattice-neighbours, |i-j|>1
    def _get_contacts(ch):
        where = {p: i for i, p in enumerate(ch)}
        contacts = set()
        for i, p in enumerate(ch):
            if i not in h_positions:
                continue
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and j in h_positions:
                    contacts.add((min(i, j), max(i, j)))
        return contacts
    
    current_contacts = _get_contacts(current_chain)
    best_contacts = set(current_contacts)
    
    T = 1.5
    cooling_rate = 0.9995
    deadline = time.time() + time_limit
    stall_counter = 0
    max_stall = 200
    tabu = []
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        
        # Generate candidate moves: prioritize moves that could create H-H contacts
        move_type = rng.random()
        new_chain = None
        move_id = None
        
        if move_type < 0.4:
            # Reversal: target segments that contain H residues at both ends
            h_indices = sorted(h_positions)
            if len(h_indices) >= 2:
                # Pick two H positions and try reversal between them
                i_h = rng.choice(h_indices)
                j_h = rng.choice(h_indices)
                if i_h > j_h:
                    i_h, j_h = j_h, i_h
                # Extend slightly
                i = max(0, i_h - rng.randint(0, 3))
                j = min(n - 1, j_h + rng.randint(0, 3))
                if i < j:
                    new_chain = _apply_reversal(current_chain, i, j)
                    move_id = ('rev', i, j)
            if new_chain is None:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = _apply_reversal(current_chain, i, j)
                move_id = ('rev', i, j)
        elif move_type < 0.7:
            # Pivot at an H residue
            h_indices = sorted(h_positions)
            if h_indices:
                i = rng.choice(h_indices)
                if 1 <= i <= n - 2:
                    angle = rng.choice([1, 2, 3])
                    new_chain = _apply_pivot(current_chain, i, angle)
                    move_id = ('pivot', i, angle)
            if new_chain is None:
                i = rng.randint(1, n - 2)
                angle = rng.choice([1, 2, 3])
                new_chain = _apply_pivot(current_chain, i, angle)
                move_id = ('pivot', i, angle)
        else:
            # Random move
            if rng.random() < 0.5:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = _apply_reversal(current_chain, i, j)
                move_id = ('rev', i, j)
            else:
                i = rng.randint(1, n - 2)
                angle = rng.choice([1, 2, 3])
                new_chain = _apply_pivot(current_chain, i, angle)
                move_id = ('pivot', i, angle)
        
        if new_chain is None:
            continue
        
        # Check tabu
        if move_id in tabu:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if improving, or with SA probability
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            current_contacts = _get_contacts(current_chain)
            stall_counter = 0
            
            # Update tabu
            tabu.append(move_id)
            if len(tabu) > tabu_len:
                tabu.pop(0)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                best_contacts = set(current_contacts)
        else:
            stall_counter += 1
        
        T *= cooling_rate
        if T < 0.05:
            if rng.random() < 0.4:
                T = 1.2
            else:
                break
        
        if stall_counter > max_stall:
            # Diversification: try a large reversal
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is not None:
                current_chain = new_chain
                current_energy = _energy(sequence, current_chain)
                current_contacts = _get_contacts(current_chain)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
                    best_contacts = set(current_contacts)
            stall_counter = 0
            tabu = []
        
        if _remaining_time() < 0.5:
            break
    
    return best_chain, best_energy


def _backbone_refit(sequence, target_chain, rng, max_attempts=20):
    """
    Try to re-fit the backbone to a different compact shape that matches
    the contact pattern of target_chain.
    Returns best_chain, best_energy or None if no improvement.
    """
    n = len(sequence)
    if n < 5:
        return None
    
    target_energy = _energy(sequence, target_chain)
    best_chain = None
    best_energy = target_energy
    
    # Extract contact pattern from target
    where = {p: i for i, p in enumerate(target_chain)}
    h_positions = set(i for i, c in enumerate(sequence) if c == "H")
    target_contacts = set()
    for i, p in enumerate(target_chain):
        if i not in h_positions:
            continue
        x, y = p
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            j = where.get(q)
            if j is not None and abs(i - j) > 1 and j in h_positions:
                target_contacts.add((min(i, j), max(i, j)))
    
    for attempt in range(max_attempts):
        if _remaining_time() < 3.0:
            break
        
        # Grow a new chain with bias toward matching target contacts
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        
        for i in range(1, n):
            current = chain[-1]
            x, y = current
            options = []
            for site in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if site in occupied:
                    continue
                
                score = 0.0
                if sequence[i] == "H":
                    sx, sy = site
                    for q in ((sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1:
                                if sequence[j] == "H":
                                    # Check if this contact is in target
                                    if (min(i, j), max(i, j)) in target_contacts:
                                        score += 3.0
                                    else:
                                        score += 1.0
                                else:
                                    score -= 0.5
                
                free_count = sum(1 for q in ((site[0] + 1, site[1]), (site[0] - 1, site[1]), (site[0], site[1] + 1), (site[0], site[1] - 1)) if q not in occupied)
                score -= free_count * 0.2
                score += rng.random() * 0.8
                options.append((score, site))
            
            if not options:
                success = False
                break
            
            options.sort(key=lambda x: x[0], reverse=True)
            best_site = options[0][1]
            occupied[best_site] = i
            chain.append(best_site)
        
        if not success:
            continue
        
        if not _is_valid_chain(chain):
            continue
        
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain[:]
    
    if best_chain is not None and best_energy < target_energy:
        return best_chain, best_energy
    return None


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation using contact-aware memetic search.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains
    compactness_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    growths_per_value = max(3, int(2.0 / len(compactness_values) / 0.015))
    
    initial_chains = []
    
    for compactness in compactness_values:
        if _remaining_time() < 8.0:
            break
        for _ in range(growths_per_value):
            if _remaining_time() < 8.0:
                break
            chain = _grow_compact(sequence, rng, compactness)
            if chain is not None and _is_valid_chain(chain):
                initial_chains.append(chain)
                
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])
        
    # Score and rank initial chains
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))
        
    scored_chains.sort(key=lambda x: x[0])
    
    top_k = min(5, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Phase 2: Contact-aware local search on top candidates
    total_time_remaining = _remaining_time()
    time_per_candidate = total_time_remaining / max(1, len(candidates))
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 2.0:
            break
            
        alloc_time = min(time_per_candidate, _remaining_time() - 1.5)
        if alloc_time < 0.5:
            continue
            
        best_c, best_e = _contact_aware_local_search(sequence, chain, alloc_time, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
    
    if best_chain is None:
        if initial_chains:
            best_chain = initial_chains[0]
        else:
            best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Backbone re-fitting with contact pattern matching
    if _remaining_time() > 5.0:
        refit_result = _backbone_refit(sequence, best_chain, rng, max_attempts=15)
        if refit_result is not None:
            refit_chain, refit_energy = refit_result
            # Polish the refit chain
            polish_time = min(_remaining_time() - 2.0, 8.0)
            if polish_time > 1.0:
                best_c, best_e = _contact_aware_local_search(sequence, refit_chain, polish_time, rng)
                if best_e < best_energy:
                    best_energy = best_e
                    best_chain = best_c
    
    # Phase 4: Final polish
    if _remaining_time() > 3.0 and best_chain is not None:
        polish_time = min(_remaining_time() - 1.5, 10.0)
        if polish_time > 1.0:
            polish_rng = random.Random(99 + len(sequence))
            best_c, best_e = _contact_aware_local_search(sequence, best_chain, polish_time, polish_rng, tabu_len=30)
            if best_e < best_energy:
                best_energy = best_e
                best_chain = best_c
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
