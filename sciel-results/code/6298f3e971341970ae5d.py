"""Rewritten QAP solver: fast local search with optimized swap delta + Hungarian refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching initial permutation.
    2. Fast local search with O(n) swap delta (no mask creation).
    3. Multiple Hungarian linearization refinement rounds.
    4. Final high-precision polish.
    """
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # --- Fast swap delta: O(n) without mask/where ---
    def swap_delta_fast(i, j, p):
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        # br_diff[l] = b[v,l] - b[u,l]
        br_diff = b[v] - b[u]
        # bc_diff[l] = b[l,v] - b[l,u]
        bc_diff = b[:, v] - b[:, u]
        
        # Corner terms (i,i), (i,j), (j,i), (j,j)
        delta = (a[i, i] * (b[v, v] - b[u, u]) +
                 a[i, j] * (b[v, u] - b[u, v]) +
                 a[j, i] * (b[u, v] - b[v, u]) +
                 a[j, j] * (b[u, u] - b[v, v]))
        
        # Row i: sum_{k!=i,j} a[i,k]*br_diff[p[k]]
        # = sum_k a[i,k]*br_diff[p[k]] - a[i,i]*br_diff[u] - a[i,j]*br_diff[v]
        s_ri = np.dot(a[i], br_diff[p])
        delta += s_ri - a[i, i] * br_diff[u] - a[i, j] * br_diff[v]
        
        # Row j: sum_{k!=i,j} a[j,k]*(-br_diff[p[k]])
        # = -sum_k a[j,k]*br_diff[p[k]] + a[j,i]*br_diff[u] + a[j,j]*br_diff[v]
        s_rj = np.dot(a[j], br_diff[p])
        delta -= s_rj - a[j, i] * br_diff[u] - a[j, j] * br_diff[v]
        
        # Col i: sum_{k!=i,j} a[k,i]*bc_diff[p[k]]
        # = sum_k a[k,i]*bc_diff[p[k]] - a[i,i]*bc_diff[u] - a[j,i]*bc_diff[v]
        s_ci = np.dot(a[:, i], bc_diff[p])
        delta += s_ci - a[i, i] * bc_diff[u] - a[j, i] * bc_diff[v]
        
        # Col j: sum_{k!=i,j} a[k,j]*(-bc_diff[p[k]])
        # = -sum_k a[k,j]*bc_diff[p[k]] + a[i,j]*bc_diff[u] + a[j,j]*bc_diff[v]
        s_cj = np.dot(a[:, j], bc_diff[p])
        delta -= s_cj - a[i, j] * bc_diff[u] - a[j, j] * bc_diff[v]
        
        return delta
    
    def compute_cost(p):
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    # --- Initial solution: rank-matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # --- Local search with fast swap delta ---
    def local_search(p, cost, time_limit):
        """Hill climbing with random 2-swap moves until no improvement or time limit."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Number of trials per pass: scale with n
            num_trials = max(5000, n * 100)
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p)
                if d < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cost += d
                    improved = True
        return p, cost
    
    # Phase 1: Initial local search
    best_p, best_cost = local_search(best_p, best_cost, deadline - 40.0)
    
    # Phase 2: Hungarian refinement (multiple rounds)
    for round_num in range(3):
        if time.time() > deadline - 30.0:
            break
        
        # Build linear cost matrix
        b_col_p = b[:, best_p]  # [j, k] = b[j, p[k]]
        b_row_p = b[best_p, :]  # [k, j] = b[p[k], j]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
        
        # Local search after Hungarian
        if time.time() < deadline - 25.0:
            best_p, best_cost = local_search(best_p, best_cost, deadline - 25.0)
    
    # Phase 3: Final polish - more iterations, tighter
    if time.time() < deadline - 5.0:
        best_p, best_cost = local_search(best_p, best_cost, deadline - 3.0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
