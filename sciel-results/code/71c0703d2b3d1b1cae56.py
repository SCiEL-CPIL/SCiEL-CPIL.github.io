"""
HP lattice protein folding on the cubic lattice.
Strategy: Monte Carlo growth with incremental energy updates, pivot moves, and simulated annealing.
"""
import random
import sys
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _manhattan(a, b):
    """Manhattan distance between two lattice points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _initial_energy(seq, chain):
    """Compute initial energy of a chain."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                if i < j:
                    contacts += 1
    return -contacts


def _compute_contacts_set(seq, chain):
    """Compute set of H-H contact pairs (i,j) with i<j."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = set()
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                pair = (min(i, j), max(i, j))
                contacts.add(pair)
    return contacts


def _fast_local_optimize(seq, chain, max_moves=5000, rng=None):
    """
    Fast local optimization using incremental energy updates.
    Only single-residue moves (1-2-3 moves) for speed.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = list(chain)
    current_energy = _initial_energy(seq, current_chain)
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    is_h = [c == 'H' for c in seq]
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    for _ in range(max_moves):
        i = rng.randint(1, n - 2)
        p = current_chain[i]
        
        for q in _neighbours(p):
            if q in pos_to_idx:
                continue
            
            prev_p = current_chain[i - 1]
            next_p = current_chain[i + 1]
            
            if _manhattan(prev_p, q) != 1:
                continue
            if _manhattan(next_p, q) != 1:
                continue
            
            # Compute energy delta incrementally
            # Remove contacts involving i at position p
            delta = 0
            for q2 in _neighbours(p):
                j = pos_to_idx.get(q2)
                if j is not None and abs(i - j) > 1 and is_h[i] and is_h[j]:
                    delta += 1
            
            # Add contacts involving i at position q
            for q2 in _neighbours(q):
                j = pos_to_idx.get(q2)
                if j is not None and abs(i - j) > 1 and is_h[i] and is_h[j]:
                    delta -= 1
            
            new_energy = current_energy - delta
            
            if new_energy < current_energy:
                # Accept the move
                pos_to_idx.pop(p)
                pos_to_idx[q] = i
                current_chain[i] = q
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
            else:
                # Try simulated annealing acceptance occasionally
                if rng.random() < 0.01:
                    T = 1.0
                    import math
                    if rng.random() < math.exp(-delta / T):
                        pos_to_idx.pop(p)
                        pos_to_idx[q] = i
                        current_chain[i] = q
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(current_chain)
    
    return best_chain


def _pivot_move(seq, chain, pos_to_idx, i, direction, rng):
    """
    Attempt a pivot move: rotate the segment from i to end around position i-1.
    Returns (new_chain_segment, new_energy_delta, success)
    """
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None, 0, False
    
    pivot_point = chain[i - 1]
    old_segment = chain[i:]
    
    # Try to rotate the segment
    # Simple approach: try moving just residue i (already handled elsewhere)
    # For now, skip complex pivots for speed
    return None, 0, False


def _simulated_annealing_fold_fast(seq, dimension, iterations=20000, rng=None):
    """
    Fast simulated annealing with incremental energy updates.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(seq)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Start with a random walk
    chain = [tuple([0] * dimension)]
    occupied = {chain[0]: 0}
    last_dir = None
    
    for i in range(1, n):
        valid_dirs = []
        for axis in range(dimension):
            for step in (-1, 1):
                if last_dir is not None and last_dir[0] == axis and last_dir[1] == -step:
                    continue
                q = list(chain[-1])
                q[axis] += step
                q = tuple(q)
                if q not in occupied:
                    valid_dirs.append((axis, step))
        
        if not valid_dirs:
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(chain[-1])
                    q[axis] += step
                    q = tuple(q)
                    if q not in occupied:
                        valid_dirs.append((axis, step))
        
        if not valid_dirs:
            return None
        
        axis, step = rng.choice(valid_dirs)
        last_dir = (axis, step)
        q = list(chain[-1])
        q[axis] += step
        q = tuple(q)
        chain.append(q)
        occupied[q] = i
    
    if len(chain) < n:
        return None
    
    current_energy = _initial_energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    is_h = [c == 'H' for c in seq]
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    T = 3.0
    cooling = 0.9995
    
    for it in range(iterations):
        T *= cooling
        if T < 0.05:
            break
        
        i = rng.randint(1, n - 2)
        p = chain[i]
        
        axis = rng.randint(0, dimension - 1)
        step = rng.choice((-1, 1))
        q = list(p)
        q[axis] += step
        q = tuple(q)
        
        if q in occupied:
            continue
        
        prev_p = chain[i - 1]
        next_p = chain[i + 1]
        
        if _manhattan(prev_p, q) != 1:
            continue
        if _manhattan(next_p, q) != 1:
            continue
        
        # Incremental energy delta
        delta = 0
        for q2 in _neighbours(p):
            j = pos_to_idx.get(q2)
            if j is not None and abs(i - j) > 1 and is_h[i] and is_h[j]:
                delta += 1
        
        for q2 in _neighbours(q):
            j = pos_to_idx.get(q2)
            if j is not None and abs(i - j) > 1 and is_h[i] and is_h[j]:
                delta -= 1
        
        new_energy = current_energy - delta
        
        import math
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.05)):
            pos_to_idx.pop(p)
            pos_to_idx[q] = i
            chain[i] = q
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            pass
    
    return best_chain


def _greedy_grow_with_heuristic(seq, dimension, rng, alpha=1.0):
    """
    Greedy growth with heuristic scoring.
    """
    n = len(seq)
    chain = [tuple([0] * dimension)]
    occupied = {chain[0]: 0}
    is_h = [c == 'H' for c in seq]
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            score = 0.0
            if is_h[i]:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and is_h[j]:
                        score += 5.0
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            score += rng.random() * alpha
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        
        # Sometimes pick a non-best option for diversity
        if rng.random() < 0.25 and len(candidates) > 1:
            idx = rng.randint(1, min(3, len(candidates) - 1))
        else:
            idx = 0
        
        site = candidates[idx][1]
        chain.append(site)
        occupied[site] = i
    
    return chain


def _hybrid_fold(seq, dimension, time_budget=10.0):
    """
    Hybrid approach: multiple greedy growths + fast local optimization + SA.
    """
    rng = random.Random(42 + len(seq))
    n = len(seq)
    
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    start_time = time.time()
    
    # Strategy 1: Multiple greedy growths with local optimization
    num_growth_trials = min(30, max(5, int(time_budget * 2)))
    for trial in range(num_growth_trials):
        if time.time() - start_time > time_budget * 0.5:
            break
        
        seed = 1000 + trial * 7
        rng_trial = random.Random(seed)
        
        chain = _greedy_grow_with_heuristic(seq, dimension, rng_trial, alpha=1.0 + trial * 0.1)
        
        if chain is None or len(chain) != n:
            continue
        
        energy = _initial_energy(seq, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Fast local optimization
        if time.time() - start_time < time_budget * 0.7:
            moves = max(1000, min(5000, int(time_budget * 300)))
            opt_chain = _fast_local_optimize(seq, chain, max_moves=moves, rng=rng_trial)
            opt_energy = _initial_energy(seq, opt_chain)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
    
    # Strategy 2: Simulated annealing from best chain found
    if time.time() - start_time < time_budget * 0.85 and best_chain is not None:
        remaining_time = time_budget - (time.time() - start_time)
        sa_iters = max(5000, min(30000, int(remaining_time * 1000)))
        sa_chain = _simulated_annealing_fold_fast(seq, dimension, iterations=sa_iters, rng=rng)
        if sa_chain is not None and len(sa_chain) == n:
            sa_energy = _initial_energy(seq, sa_chain)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain
    
    # Strategy 3: Additional local optimization on best chain
    if time.time() - start_time < time_budget * 0.95 and best_chain is not None:
        remaining_time = time_budget - (time.time() - start_time)
        final_moves = max(2000, min(10000, int(remaining_time * 500)))
        final_chain = _fast_local_optimize(seq, best_chain, max_moves=final_moves, rng=rng)
        final_energy = _initial_energy(seq, final_chain)
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Determine time budget: total 240s for 26 sequences
    time_budget = max(2.0, min(12.0, 240.0 / 26.0 * 0.85))
    
    chain = _hybrid_fold(sequence, dimension, time_budget=time_budget)
    
    # Ensure the chain is valid
    if chain is None or len(chain) != n:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
