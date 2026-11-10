"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    """Yield the 6 neighbors of a lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _are_adjacent(a, b):
    """Check if two lattice points are adjacent (unit distance)."""
    return sum(abs(a[i] - b[i]) for i in range(len(a))) == 1


def _grow_compact(sequence, dimension, rng):
    """Growth with compactness bias."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
            dist = abs(site[0] - cx) + abs(site[1] - cy) + abs(site[2] - cz)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 3.0 - dist * 0.5 + free * 0.1 + rng.random() * 0.3
            options.append((score, site))
        if not options:
            return None
        options.sort(reverse=True)
        k = min(3, len(options))
        site = options[rng.randint(0, k - 1)][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    """Compute energy = -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _compute_single_move_delta(sequence, chain, occupied, i, new_site):
    """
    Compute energy delta for moving residue i to new_site.
    Returns (delta, valid).
    """
    n = len(chain)
    if i == 0 or i == n - 1:
        return 0, False
    if new_site in occupied:
        return 0, False
    # Check adjacency with neighbors
    if not _are_adjacent(chain[i - 1], new_site):
        return 0, False
    if not _are_adjacent(chain[i + 1], new_site):
        return 0, False
    
    old_site = chain[i]
    if sequence[i] != 'H':
        # No energy change since P residues don't contribute
        # But we need to check self-avoidance is maintained
        return 0, True
    
    # Compute lost contacts at old_site
    lost = 0
    for q in _neighbours(old_site):
        j = occupied.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == 'H':
                lost += 1
    
    # Compute gained contacts at new_site
    gained = 0
    for q in _neighbours(new_site):
        j = occupied.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == 'H':
                gained += 1
    
    delta = gained - lost
    return delta, True


def _compute_reversal_delta(sequence, chain, occupied, i, j):
    """
    Compute energy delta for reversing segment [i..j].
    Returns (delta, valid).
    """
    n = len(chain)
    if i <= 0 or j >= n - 1 or j - i < 1:
        return 0, False
    
    # Check connectivity
    if not _are_adjacent(chain[i - 1], chain[j]):
        return 0, False
    if j + 1 < n and not _are_adjacent(chain[i], chain[j + 1]):
        return 0, False
    
    # Compute energy delta
    # The set of sites is the same, only order changes
    # We need to check which H-H contacts are broken/formed
    # Build set of positions in the segment
    segment_positions = set(range(i, j + 1))
    
    # For each H in the segment, count contacts with other H's not in segment
    # and with H's in segment that are not adjacent in chain
    
    # Easier: compute energy of affected region before and after
    # Affected: residues i-1, i, ..., j, j+1 and their neighbors
    
    # Let's just recompute the relevant contacts
    # Build temp occupied without segment
    temp_occ = dict(occupied)
    for idx in range(i, j + 1):
        temp_occ[chain[idx]] = -1
    
    old_segment = list(chain[i:j + 1])
    new_segment = list(reversed(old_segment))
    
    delta = 0
    for k in range(i, j + 1):
        old_site = old_segment[k - i]
        new_site = new_segment[k - i]
        if old_site == new_site:
            continue
        if sequence[k] == 'H':
            # Lost contacts at old_site
            for q in _neighbours(old_site):
                jj = temp_occ.get(q)
                if jj is not None and jj >= 0 and jj != k:
                    if sequence[jj] == 'H':
                        delta -= 1
            # Gained contacts at new_site
            for q in _neighbours(new_site):
                jj = temp_occ.get(q)
                if jj is not None and jj >= 0 and jj != k:
                    if sequence[jj] == 'H':
                        delta += 1
    
    return delta, True


def _local_search(sequence, chain, rng, deadline, max_stagnation=500):
    """
    Local search with single-site moves and segment reversals.
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    stagnation = 0
    temp = 1.0
    cooling_rate = 0.995
    
    while time.time() < deadline and stagnation < max_stagnation:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Single-site move
            i = rng.randint(1, n - 2)
            candidates = list(_neighbours(current_chain[i]))
            if not candidates:
                continue
            new_site = candidates[rng.randint(0, len(candidates) - 1)]
            
            delta, valid = _compute_single_move_delta(sequence, current_chain, occupied, i, new_site)
            if not valid:
                continue
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                if temp > 0.001:
                    accept = rng.random() < math.exp(-delta / temp)
            
            if accept:
                old_site = current_chain[i]
                occupied[old_site] = -1
                occupied[new_site] = i
                current_chain[i] = new_site
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
        
        elif move_type < 0.8:
            # Segment reversal
            max_seg = min(n // 2, 15)
            seg_len = rng.randint(2, max_seg)
            i = rng.randint(1, n - seg_len - 1)
            j = i + seg_len - 1
            
            delta, valid = _compute_reversal_delta(sequence, current_chain, occupied, i, j)
            if not valid:
                continue
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                if temp > 0.001:
                    accept = rng.random() < math.exp(-delta / temp)
            
            if accept:
                # Apply reversal
                old_segment = current_chain[i:j + 1]
                new_segment = list(reversed(old_segment))
                for k in range(i, j + 1):
                    old_site = old_segment[k - i]
                    new_site = new_segment[k - i]
                    if old_site != new_site:
                        occupied[old_site] = -1
                        occupied[new_site] = k
                    current_chain[k] = new_site
                
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
        else:
            # Do nothing (cooling)
            pass
        
        temp *= cooling_rate
        if temp < 0.005:
            temp = 0.5  # Reheat
    
    return best_chain, best_energy


def _perturb(sequence, chain, rng, num_moves):
    """
    Apply random perturbations to chain.
    Returns new_chain or None if invalid.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    new_chain = list(chain)
    occupied = {p: i for i, p in enumerate(new_chain)}
    
    for _ in range(num_moves):
        # Random single-site move
        i = rng.randint(1, n - 2)
        candidates = list(_neighbours(new_chain[i]))
        if not candidates:
            continue
        new_site = candidates[rng.randint(0, len(candidates) - 1)]
        
        if new_site in occupied:
            continue
        if not _are_adjacent(new_chain[i - 1], new_site):
            continue
        if not _are_adjacent(new_chain[i + 1], new_site):
            continue
        
        old_site = new_chain[i]
        occupied[old_site] = -1
        occupied[new_site] = i
        new_chain[i] = new_site
    
    return new_chain


def fold(sequence, dimension):
    """Best conformation via growth + iterative local search with perturbation."""
    start_time = time.time()
    # 240s total for 26 sequences, so ~9.2s each. Use 8.5s per sequence for safety.
    total_deadline = start_time + 8.5
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate initial chain
    chain = _grow_compact(sequence, dimension, rng)
    if chain is None:
        # Fallback: straight line
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    # Phase 2: Iterative local search with perturbation
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    
    # First local search
    ls_deadline = total_deadline - 0.5
    best_chain, best_energy = _local_search(sequence, chain, rng, ls_deadline)
    
    # Iterative improvement: perturb and local search
    iteration = 0
    while time.time() < total_deadline - 0.5:
        # Perturb the best chain
        num_perturb = min(10, len(sequence) // 4)
        perturbed = _perturb(sequence, best_chain, rng, num_perturb)
        if perturbed is None:
            break
        
        # Local search on perturbed chain
        iter_deadline = min(total_deadline - 0.3, time.time() + 1.5)
        if iter_deadline <= time.time():
            break
        
        new_chain, new_energy = _local_search(sequence, perturbed, rng, iter_deadline)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = list(new_chain)
        
        iteration += 1
        if iteration > 20:  # Safety limit
            break
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
