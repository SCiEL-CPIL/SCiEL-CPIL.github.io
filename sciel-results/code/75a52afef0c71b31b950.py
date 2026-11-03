"""QAP solver: fast local search with optimized swap delta + Hungarian refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching + Hungarian initial permutation.
    2. Fast local search with O(n) swap delta and batch evaluation.
    3. Multiple Hungarian linearization refinement rounds.
    4. Final high-precision polish with best-first moves.
    """
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s limit
    np.random.seed(42)
    
    def swap_delta_fast(i, j, p):
        """Compute cost change for swapping facilities i and j in permutation p."""
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
        """Compute exact QAP cost for permutation p."""
        total = 0.0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    def local_search(p, cost, time_limit, batch_size=500):
        """Hill climbing with batch-evaluated 2-swap moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Evaluate a batch of random swaps
            for _ in range(batch_size):
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
    
    def best_first_local_search(p, cost, time_limit, num_candidates=200):
        """Evaluate many candidates, apply the best improving move."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Generate candidate moves
            candidates = []
            for _ in range(num_candidates):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p)
                if d < -1e-9:
                    candidates.append((d, i, j))
            
            if not candidates:
                break
            
            # Apply the best move
            candidates.sort(key=lambda x: x[0])
            d, i, j = candidates[0]
            p[i], p[j] = p[j], p[i]
            cost += d
            improved = True
        return p, cost
    
    # --- Initial solution: rank-matching + Hungarian ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Refine with Hungarian
    b_col_p = b[:, p]
    b_row_p = b[p, :]
    C = a @ b_col_p.T + a.T @ b_row_p
    row_ind, col_ind = linear_sum_assignment(C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    cost_hung = compute_cost(p_hung)
    cost_rank = compute_cost(p)
    
    best_p = p_hung.copy() if cost_hung < cost_rank else p.copy()
    best_cost = min(cost_hung, cost_rank)
    
    # Phase 1: Initial local search (longer)
    phase1_deadline = deadline - 50.0
    best_p, best_cost = local_search(best_p, best_cost, phase1_deadline, batch_size=300)
    
    # Phase 2: Hungarian refinement (2 rounds)
    for round_num in range(2):
        if time.time() > deadline - 30.0:
            break
        
        # Build linear cost matrix
        b_col_p = b[:, best_p]
        b_row_p = b[best_p, :]
        
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
            best_p, best_cost = local_search(best_p, best_cost, deadline - 25.0, batch_size=300)
    
    # Phase 3: Final polish with best-first moves
    if time.time() < deadline - 5.0:
        best_p, best_cost = best_first_local_search(best_p, best_cost, deadline - 3.0, num_candidates=100)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
