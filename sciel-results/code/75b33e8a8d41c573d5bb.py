"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    def calc_delta(i, j, p):
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        
        direct = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        
        diff_b_row = b[v, p] - b[u, p]
        diff_b_col = b[p, v] - b[p, u]
        
        s1 = np.dot(diff_a_row, diff_b_row)
        s2 = np.dot(diff_a_col, diff_b_col)
        
        s1 -= diff_a_row[i] * (b[v, u] - b[u, u])
        s1 -= diff_a_row[j] * (b[v, v] - b[u, v])
        
        s2 -= diff_a_col[i] * (b[u, v] - b[u, u])
        s2 -= diff_a_col[j] * (b[v, v] - b[v, u])
        
        return s1 + s2 + direct

    t0 = time.time()
    time_limit = 115.0
    
    best_p = None
    best_cost = np.iinfo(np.int64).max
    
    rng = np.random.default_rng(42)
    
    # Start with a random permutation
    p = rng.permutation(n)
    current_cost = total_cost(p)
    
    # Track stagnation
    stagnation = 0
    
    while time.time() - t0 < time_limit:
        # Local Search Phase (First Improvement)
        improved_in_pass = False
        # Use a fixed order or random order? Random order helps escape symmetries?
        # Let's use a shuffled order for i, and sequential for j.
        i_order = list(range(n))
        rng.shuffle(i_order)
        
        for i in i_order:
            if time.time() - t0 > time_limit:
                break
            for j in range(i + 1, n):
                delta = calc_delta(i, j, p)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved_in_pass = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        stagnation = 0
                    # Do not break, continue to find more improvements in this pass
            if time.time() - t0 > time_limit:
                break
        
        if not improved_in_pass:
            # Converged
            stagnation += 1
            
            # Perturb the best solution found so far
            if best_p is not None:
                p = best_p.copy()
                current_cost = best_cost
            else:
                p = rng.permutation(n)
                current_cost = total_cost(p)
                
            # Adaptive perturbation strength
            # If stuck for a long time, perturb more
            K = min(5 + stagnation * 2, 20)
            for _ in range(K):
                i = rng.integers(n)
                j = rng.integers(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = total_cost(p)
            
            # If the perturbed solution is significantly worse, 
            # we might want to reset to best_p immediately? 
            # No, let LS try to recover.
            
    if best_p is None:
        return p.tolist()
    return best_p.tolist()
# EVOLVE-BLOCK-END
