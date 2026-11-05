"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(pt, perm):
    """Compute makespan for a given permutation. O(n*m)."""
    n, m = pt.shape
    if len(perm) == 0:
        return 0.0
    # Vectorized makespan computation
    # C[j, k] = completion time of job perm[j] on machine k
    # We can compute this incrementally
    jobs = pt[perm]  # (n, m)
    # C[j, 0] = C[j-1, 0] + jobs[j, 0]
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + jobs[j, k]
    
    # Use a simple loop for clarity, but we can vectorize the machine dimension
    # Actually for small n*m this is fine
    c_prev = np.zeros(m)
    c_cur = np.zeros(m)
    for j in range(n):
        c_cur[0] = c_prev[0] + jobs[j, 0]
        for k in range(1, m):
            c_cur[k] = max(c_prev[k], c_cur[k-1]) + jobs[j, k]
        c_prev, c_cur = c_cur, c_prev
        # Wait, this swap logic is wrong. Let me redo.
    
    # Let me just do it properly
    c = np.zeros(m)
    for j in range(n):
        new_c = np.zeros(m)
        new_c[0] = c[0] + jobs[j, 0]
        for k in range(1, m):
            new_c[k] = max(c[k], new_c[k-1]) + jobs[j, k]
        c = new_c
    return c[-1]


def _makespan_fast(pt, perm):
    """Faster makespan using numpy where possible."""
    n, m = pt.shape
    if n == 0:
        return 0.0
    jobs = pt[perm]
    # For small instances, the loop is fine. Let's keep it simple.
    c = np.zeros(m)
    for j in range(n):
        new_c = np.empty(m)
        new_c[0] = c[0] + jobs[j, 0]
        if m > 1:
            # new_c[k] = max(c[k], new_c[k-1]) + jobs[j, k]
            # This has a sequential dependency along k, so can't fully vectorize
            # But for typical m (small), this is fine
            for k in range(1, m):
                new_c[k] = max(c[k], new_c[k-1]) + jobs[j, k]
        c = new_c
    return c[-1]


def _neh_order(pt):
    """NEH heuristic for flow shop."""
    n, m = pt.shape
    totals = pt.sum(axis=1)
    # Sort jobs by total processing time descending
    order = np.argsort(-totals)
    
    perm = []
    for j in order:
        # Insert job j at the position that minimizes makespan
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            mk = _makespan(pt, candidate)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        perm = perm[:best_pos] + [j] + perm[best_pos:]
    
    return perm


def _swap_neighbor(perm, i, j):
    """Return permutation with positions i and j swapped."""
    p = list(perm)
    p[i], p[j] = p[j], p[i]
    return p


def _insert_neighbor(perm, i, j):
    """Remove element at position i and insert it at position j."""
    p = list(perm)
    el = p.pop(i)
    p.insert(j, el)
    return p


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 5s before 120s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n <= 1:
        return list(range(n))
    
    # Step 1: NEH heuristic to get a good initial solution
    perm = _neh_order(pt)
    best_perm = list(perm)
    best_mk = _makespan(pt, best_perm)
    
    if n <= 2:
        return best_perm
    
    # Step 2: Local search with simulated annealing
    rng = random.Random(42)
    
    # Temperature schedule
    T_init = max(best_mk * 0.1, 1.0)
    T_min = 0.01
    cooling_rate = 0.995
    
    T = T_init
    current_perm = list(best_perm)
    current_mk = best_mk
    
    # Precompute for speed: we'll use swap and insert moves
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Generate a random neighbor
        move_type = rng.random()
        
        if move_type < 0.5:
            # Swap move
            i, j = sorted(rng.sample(range(n), 2))
            new_perm = _swap_neighbor(current_perm, i, j)
        else:
            # Insert move
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            new_perm = _insert_neighbor(current_perm, i, j)
        
        new_mk = _makespan(pt, new_perm)
        delta = new_mk - current_mk
        
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            current_perm = new_perm
            current_mk = new_mk
            no_improve_count = 0
            
            if new_mk < best_mk:
                best_mk = new_mk
                best_perm = list(new_perm)
        else:
            no_improve_count += 1
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If we've been stuck too long and temperature is low, restart from best
        if no_improve_count > max_no_improve and T <= T_min * 2:
            # Restart from best solution with new temperature
            current_perm = list(best_perm)
            current_mk = best_mk
            T = T_init * 0.5
            no_improve_count = 0
            max_no_improve = 300
    
    return best_perm
# EVOLVE-BLOCK-END
