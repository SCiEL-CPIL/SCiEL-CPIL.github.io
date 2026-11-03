"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # --- Greedy construction for a good initial solution
    order = np.argsort(-a.sum(axis=1), kind="stable")
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    for idx in range(n):
        f = order[idx]
        best_loc = -1
        best_cost = np.iinfo(np.int64).max
        for loc in range(n):
            if used[loc]:
                continue
            inc = 0
            for k in range(idx):
                g = order[k]
                inc += a[f, g] * b[loc, p[g]] + a[g, f] * b[p[g], loc]
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[f] = best_loc
        used[best_loc] = True

    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = total_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    t0 = time.time()
    time_limit = 110.0

    # Simulated Annealing
    T = 100000.0
    T_min = 1.0
    cooling_rate = 0.999
    
    # Precompute b rows and cols for faster access? 
    # b[p, v] is just indexing.
    
    rng = np.random.default_rng(42)
    
    while time.time() - t0 < time_limit:
        # Pick random i, j
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        
        u = p[i]
        v = p[j]
        
        # Calculate Delta
        # diff_b_vec = b[v, p] - b[u, p]
        # diff_b_col = b[p, v] - b[p, u]
        
        diff_b_vec = b[v, p] - b[u, p]
        diff_b_col = b[p, v] - b[p, u]
        
        # Sum 1: a[i, :] . diff_b_vec
        s1 = np.dot(a[i, :], diff_b_vec)
        # Exclude k=i, k=j
        s1 -= a[i, i] * (b[v, u] - b[u, u])
        s1 -= a[i, j] * (b[v, v] - b[u, v])
        
        # Sum 2: a[:, i] . diff_b_col
        s2 = np.dot(a[:, i], diff_b_col)
        # Exclude k=i, k=j
        s2 -= a[i, i] * (b[u, v] - b[u, u])
        s2 -= a[j, i] * (b[v, v] - b[v, u])
        
        # Sum 3: - a[j, :] . diff_b_vec
        s3 = -np.dot(a[j, :], diff_b_vec)
        # Exclude k=i, k=j
        s3 += a[j, i] * (b[v, u] - b[u, u])
        s3 += a[j, j] * (b[v, v] - b[u, v])
        
        # Sum 4: - a[:, j] . diff_b_col
        s4 = -np.dot(a[:, j], diff_b_col)
        # Exclude k=i, k=j
        s4 += a[i, j] * (b[u, v] - b[u, u])
        s4 += a[j, j] * (b[v, v] - b[v, u])
        
        # Direct terms
        direct = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        delta = s1 + s2 + s3 + s4 + direct
        
        # Acceptance
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            # Swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_p.tolist()
# EVOLVE-BLOCK-END
