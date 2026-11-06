"""HP lattice protein folding on the cubic lattice - Pivot-based Monte Carlo approach."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain, occupied):
    """Compute the energy (negative number of H-H contacts)."""
    n = len(chain)
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _initial_walk(sequence, dimension, rng):
    """
    Build an initial self-avoiding walk using a heuristic that tries to
    cluster H residues.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        current = chain[-1]
        candidates = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            # Score this site
            score = rng.random() * 0.1
            if sequence[i] == 'H':
                # Count H neighbours already placed
                for q in _neighbours(site):
                    if q in occupied and sequence[occupied[q]] == 'H':
                        score += 1.0
            # Slight preference for sites with more free space (avoid trapping)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            candidates.append((score, site))
        
        if not candidates:
            # Dead end - try to backtrack or just fail
            return None
        
        # Pick best candidate
        candidates.sort(key=lambda x: -x[0])
        site = candidates[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain


def _try_2_pivot(sequence, chain, occupied, i, rng):
    """
    Try a 2-pivot move: rotate segment [i, i+1].
    Returns (new_chain, new_occupied, new_energy) or None.
    """
    n = len(chain)
    if i < 1 or i + 2 >= n:
        return None
    
    p_prev = chain[i-1]
    p_next = chain[i+2]
    
    candidates = []
    for p_i_new in _neighbours(p_prev):
        if p_i_new == p_next:
            continue
        for p_i1_new in _neighbours(p_i_new):
            if p_i1_new == p_prev or p_i1_new == p_next:
                continue
            dist = abs(p_i1_new[0]-p_next[0]) + abs(p_i1_new[1]-p_next[1]) + abs(p_i1_new[2]-p_next[2])
            if dist == 1:
                valid = True
                for pos in (p_i_new, p_i1_new):
                    if pos in occupied:
                        idx = occupied[pos]
                        if idx != i and idx != i+1:
                            valid = False
                            break
                if valid:
                    candidates.append((p_i_new, p_i1_new))
    
    if not candidates:
        return None
    
    p_i_new, p_i1_new = rng.choice(candidates)
    
    # Create new chain and occupied dict
    new_chain = list(chain)
    new_occupied = dict(occupied)
    
    old_p_i = chain[i]
    old_p_i1 = chain[i+1]
    
    new_chain[i] = p_i_new
    new_chain[i+1] = p_i1_new
    
    del new_occupied[old_p_i]
    del new_occupied[old_p_i1]
    new_occupied[p_i_new] = i
    new_occupied[p_i1_new] = i+1
    
    new_energy = _compute_energy(sequence, new_chain, new_occupied)
    
    return new_chain, new_occupied, new_energy


def _try_3_pivot(sequence, chain, occupied, i, rng):
    """
    Try a 3-pivot move: rotate segment [i, i+1, i+2].
    Returns (new_chain, new_occupied, new_energy) or None.
    """
    n = len(chain)
    if i < 1 or i + 3 >= n:
        return None
    
    p_prev = chain[i-1]
    p_next = chain[i+3]
    
    candidates = []
    for p_i_new in _neighbours(p_prev):
        if p_i_new == p_next:
            continue
        for p_i1_new in _neighbours(p_i_new):
            if p_i1_new in (p_prev, p_next):
                continue
            for p_i2_new in _neighbours(p_i1_new):
                if p_i2_new in (p_prev, p_next, p_i_new, p_i1_new):
                    continue
                dist = abs(p_i2_new[0]-p_next[0]) + abs(p_i2_new[1]-p_next[1]) + abs(p_i2_new[2]-p_next[2])
                if dist == 1:
                    valid = True
                    for pos in (p_i_new, p_i1_new, p_i2_new):
                        if pos in occupied:
                            idx = occupied[pos]
                            if idx not in (i, i+1, i+2):
                                valid = False
                                break
                    if valid:
                        candidates.append((p_i_new, p_i1_new, p_i2_new))
    
    if not candidates:
        return None
    
    p_i_new, p_i1_new, p_i2_new = rng.choice(candidates)
    
    new_chain = list(chain)
    new_occupied = dict(occupied)
    
    old_p_i = chain[i]
    old_p_i1 = chain[i+1]
    old_p_i2 = chain[i+2]
    
    new_chain[i] = p_i_new
    new_chain[i+1] = p_i1_new
    new_chain[i+2] = p_i2_new
    
    del new_occupied[old_p_i]
    del new_occupied[old_p_i1]
    del new_occupied[old_p_i2]
    new_occupied[p_i_new] = i
    new_occupied[p_i1_new] = i+1
    new_occupied[p_i2_new] = i+2
    
    new_energy = _compute_energy(sequence, new_chain, new_occupied)
    
    return new_chain, new_occupied, new_energy


def _simulated_annealing_pivot(sequence, dimension, time_budget, rng):
    """
    Simulated annealing using pivot moves.
    """
    n = len(sequence)
    
    chain = _initial_walk(sequence, dimension, rng)
    if chain is None:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain, occupied)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    t0 = 100.0
    t_final = 0.01
    max_iters = min(200000, int(time_budget * 20000))
    if max_iters < 1000:
        max_iters = 1000
    
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.95:
            break
        
        alpha = it / max_iters
        temp = t0 * (t_final / t0) ** alpha
        
        move_type = rng.random()
        if move_type < 0.5:
            i = rng.randint(1, n - 3)
            result = _try_2_pivot(sequence, chain, occupied, i, rng)
        else:
            i = rng.randint(1, n - 4)
            result = _try_3_pivot(sequence, chain, occupied, i, rng)
        
        if result is None:
            continue
            
        new_chain, new_occupied, new_energy = result
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            chain = new_chain
            occupied = new_occupied
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
    
    return best_chain, best_energy


def _try_multiple_starts(sequence, dimension, time_budget, rng):
    """Try multiple initializations and pick the best."""
    best_chain = None
    best_energy = 0
    
    num_restarts = max(1, min(5, int(time_budget / 2)))
    time_per_restart = time_budget / num_restarts
    
    for r in range(num_restarts):
        remaining = time_budget - (time.time() - _try_multiple_starts._start_time) if hasattr(_try_multiple_starts, '_start_time') else time_budget
        if remaining < 0.1:
            break
        chain, energy = _simulated_annealing_pivot(sequence, dimension, time_per_restart, rng)
        if chain is not None and energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = 0
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice using pivot-based simulated annealing.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    base_time = 5.0
    time_budget = base_time + (n / 100.0) * 3.0
    time_budget = min(time_budget, 20.0)
    
    _try_multiple_starts._start_time = time.time()
    chain, energy = _try_multiple_starts(sequence, dimension, time_budget, rng)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
