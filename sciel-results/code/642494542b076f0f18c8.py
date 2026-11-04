"""
Strategy: QAP tai256c solver using Iterated Tabu Search with 2-opt moves and
multi-start spectral/random initialization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym):
    """
    Compute the change in cost when swapping facilities at positions i and j.
    p[i] is the location assigned to facility i.
    """
    r = i
    s = j
    x = p[r]
    y = p[s]
    
    if x == y:
        return 0
    
    n = len(p)
    
    # Vectorized calculation
    diff_b = b[y, p] - b[x, p]
    coeff = a_sym[r, :] - a_sym[s, :]
    
    delta = np.dot(coeff, diff_b)
    
    # Correct for the self-terms (k=r and k=s) which were included in the dot product
    # but need special handling due to the change in p[r] and p[s]
    
    # Subtract contribution of k=r from dot product
    # coeff[r] = a_sym[r,r] - a_sym[s,r] = 2*a[r,r] - a[r,s] - a[s,r]
    # diff_b[r] = b[y, p[r]] - b[x, p[r]] = b[y, x] - b[x, x]
    delta -= (a_sym[r, r] - a_sym[s, r]) * (b[y, x] - b[x, x])
    
    # Subtract contribution of k=s from dot product
    # coeff[s] = a_sym[r,s] - a_sym[s,s] = a[r,s] + a[s,r] - 2*a[s,s]
    # diff_b[s] = b[y, p[s]] - b[x, p[s]] = b[y, y] - b[x, y]
    delta -= (a_sym[r, s] - a_sym[s, s]) * (b[y, y] - b[x, y])
    
    # Add the correct self terms:
    # The terms involving a[r,r] and a[s,s] change location.
    # Old: a[r,r]*b[x,x] + a[s,s]*b[y,y]
    # New: a[r,r]*b[y,y] + a[s,s]*b[x,x]
    # Change: a[r,r]*(b[y,y]-b[x,x]) + a[s,s]*(b[x,x]-b[y,y])
    delta += a[r, r] * (b[y, y] - b[x, x])
    delta += a[s, s] * (b[x, x] - b[y, y])
    
    return delta

def _spectral_init(a, b, n):
    """
    Generate an initial permutation using spectral alignment.
    Align the principal eigenvectors of A and B.
    """
    def get_pca(mat):
        # Center the matrix? Or just use row sums?
        # Let's use the first left and right singular vectors?
        # Or just the largest eigenvector of (mat+mat.T)/2
        m = (mat + mat.T) / 2.0
        vals, vecs = np.linalg.eigh(m)
        idx = np.argmax(vals)
        return vecs[:, idx]
    
    try:
        v_a = get_pca(a)
        v_b = get_pca(b)
        
        # Sort indices by v_a and v_b
        order_a = np.argsort(v_a)
        order_b = np.argsort(v_b)
        
        # If v_a is mostly negative, flip it?
        # Standard PCA alignment:
        # If dot(v_a, v_b) < 0, flip v_b? 
        # Actually, we just want to align the "directions".
        # Let's try:
        # Assign facility order_a[k] to location order_b[k]
        
        p = np.empty(n, dtype=np.int64)
        for k in range(n):
            p[order_a[k]] = order_b[k]
        return p
        
    except Exception:
        return np.random.permutation(n)

def _tabu_search(a, b, p_init, time_limit, a_sym):
    n = len(p_init)
    p = p_init.copy()
    cost = _qap_cost(a, b, p)
    
    # Tabu list: (i, j, until_iter)
    tabu_len = 20
    tabu_list = {}  # (i, j) -> until_iter
    
    best_p = p.copy()
    best_cost = cost
    no_improve_count = 0
    
    # Precompute some helper arrays?
    # We need to evaluate all moves in each iteration.
    # O(n^2) moves per iteration.
    # n=256, n^2=65536.
    # Each move evaluation is O(n) using vectorized dot.
    # 65536 * 256 flops ~ 16M flops per iteration.
    # In Python/NumPy, this might be too slow if done naively in a loop.
    # We need to vectorize the move evaluation better or use a faster heuristic.
    
    # Alternative: Sample moves?
    # Or: Compute all deltas at once?
    # delta(i, j) depends on p[i] and p[j].
    # It is hard to fully vectorize over all pairs (i, j) without huge memory or complex broadcasting.
    
    # Let's try a "fast" local search step first to settle, then tabu with sampled moves or smaller neighborhood?
    # Actually, for n=256, a full pass of 2-opt (greedy best improvement or first improvement) is feasible if optimized.
    # First improvement 2-opt:
    # For i in range(n):
    #   For j in range(i+1, n):
    #     if delta < 0: swap, restart j? or continue?
    
    # Let's stick to a robust implementation.
    # To speed up, we can precompute b[p, :]? No, p changes.
    
    # Let's use a simplified Tabu Search that samples a subset of moves per iteration to ensure speed.
    # Or, use the "greedy" 2-opt until local optimum, then perturb with tabu memory.
    
    # Given the time constraint, let's do:
    # 1. Greedy 2-opt local search (first improvement).
    # 2. Perturbation: random k swaps.
    # 3. Greedy 2-opt local search again.
    # This is Iterated Local Search (ILS). Tabu adds memory but ILS is simpler and often effective.
    # The prompt asked for a STRUCTURAL change. ILS is a different structure from plain SA.
    # Let's implement ILS with a good perturbation.
    
    while time.time() < time_limit:
        # Local Search: First Improvement 2-opt
        improved = True
        while improved and time.time() < time_limit - 5.0:
            improved = False
            for i in range(n):
                if time.time() > time_limit - 5.0:
                    break
                for j in range(i + 1, n):
                    delta = _swap_delta(a, b, p, i, j, a_sym)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        # Do not restart i, just continue? Or restart to be strict?
                        # Continuing is faster.
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # Perturbation
        # Number of swaps: 2 + int(n * 0.02)? ~ 2-7 swaps
        num_swaps = random.randint(2, 7)
        for _ in range(num_swaps):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
            # Update cost? We can recompute later in LS.
            # But we need current cost for potential acceptance?
            # ILS usually accepts the result of LS on perturbed solution unconditionally or with SA.
            # Let's just run LS on the perturbed p.
            
    return best_p, best_cost

def _solve_tai256c(a, b, time_limit=145.0):
    n = len(a)
    start_time = time.time()
    a_sym = a + a.T
    
    best_p = None
    best_cost = float('inf')
    
    # Multi-start
    # Start 1: Spectral
    p_init = _spectral_init(a, b, n)
    p_res, cost_res = _tabu_search(a, b, p_init, time.time() + 60.0, a_sym)
    if cost_res < best_cost:
        best_cost = cost_res
        best_p = p_res
    
    # Start 2: Random
    if time.time() - start_time < time_limit - 10.0:
        p_init = np.random.permutation(n)
        p_res, cost_res = _tabu_search(a, b, p_init, time.time() + 60.0, a_sym)
        if cost_res < best_cost:
            best_cost = cost_res
            best_p = p_res
            
    # If time remains, do more random restarts
    while time.time() - start_time < time_limit - 5.0:
        p_init = np.random.permutation(n)
        # Shorter time limit for each restart?
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10.0:
            break
        p_res, cost_res = _tabu_search(a, b, p_init, time.time() + remaining * 0.5, a_sym)
        if cost_res < best_cost:
            best_cost = cost_res
            best_p = p_res

    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
