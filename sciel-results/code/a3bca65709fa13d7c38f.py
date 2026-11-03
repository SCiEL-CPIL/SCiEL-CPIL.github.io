"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Time limit
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    def get_initial_permutation():
        # Rank-matching heuristic
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def compute_cost(p):
        b_p = b[np.ix_(p, p)]
        return np.sum(a * b_p)

    def local_search(p):
        current_cost = compute_cost(p)
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    
                    # S1 = sum_{k != i, j} (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
                    diff_a_row = a[i, :] - a[j, :]
                    diff_b_col = b[v, p] - b[u, p]
                    
                    dot1 = np.dot(diff_a_row, diff_b_col)
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    s1 = dot1 - term_i - term_j
                    
                    # S2 = sum_{k != i, j} (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
                    diff_a_col = a[:, i] - a[:, j]
                    diff_b_row = b[p, v] - b[p, u]
                    
                    dot2 = np.dot(diff_a_col, diff_b_row)
                    term_i2 = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                    term_j2 = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                    s2 = dot2 - term_i2 - term_j2
                    
                    # Direct term
                    direct = (a[i, j] * b[v, u] + a[j, i] * b[u, v]) - (a[i, j] * b[u, v] + a[j, i] * b[v, u])
                    
                    delta = s1 + s2 + direct
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
                        improved = True
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
        return p, current_cost

    # Main Loop: Multiple Restarts
    best_p = get_initial_permutation()
    best_cost, _ = local_search(best_p)
    
    # If time remains, try random restarts
    while time.time() - start_time < time_limit:
        p_rand = np.random.permutation(n)
        p_new, cost_new = local_search(p_rand)
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
