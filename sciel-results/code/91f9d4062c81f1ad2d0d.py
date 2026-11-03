import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    """
    Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching + Hungarian initial permutation.
    2. Fast local search with O(n) swap delta and best-of-batch move selection.
    3. Multiple Hungarian linearization refinement rounds to escape local minima.
    4. Final high-precision polish.
    """
    n = len(a)
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Precompute useful arrays
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    def swap_delta_fast(i, j, p):
        """Compute cost change for swapping facilities i and j in permutation p."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0.0
        
        # Extract rows/cols of b for u and v
        b_u_row = b[u, :]
        b_v_row = b[v, :]
        b_u_col = b[:, u]
        b_v_col = b[:, v]
        
        # Diagonal terms
        buu = b[u, u]
        bvv = b[v, v]
        buv = b[u, v]
        bvu = b[v, u]
        
        # Dot products with current permutation p
        # S1: sum_k a[i,k] * b[v, p[k]]
        S1 = np.dot(a[i], b_v_row[p])
        # S2: sum_k a[i,k] * b[u, p[k]]
        S2 = np.dot(a[i], b_u_row[p])
        # S3: sum_k a[j,k] * b[u, p[k]]
        S3 = np.dot(a[j], b_u_row[p])
        # S4: sum_k a[j,k] * b[v, p[k]]
        S4 = np.dot(a[j], b_v_row[p])
        
        # S5: sum_k a[k,i] * b[p[k], v]
        S5 = np.dot(a[:, i], b_v_col[p])
        # S6: sum_k a[k,i] * b[p[k], u]
        S6 = np.dot(a[:, i], b_u_col[p])
        # S7: sum_k a[k,j] * b[p[k], u]
        S7 = np.dot(a[:, j], b_u_col[p])
        # S8: sum_k a[k,j] * b[p[k], v]
        S8 = np.dot(a[:, j], b_v_col[p])
        
        # Total change from row/col swaps
        delta = (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8)
        
        # Subtract corrections for k=i and k=j which were included in the sums
        # but should be handled by the direct diagonal/interaction terms or excluded.
        
        # Correction for k=i:
        # In S1, S2: a[i,i] * (b[v,u] - b[u,u]) = a[i,i] * (bvu - buu)
        # In S3, S4: a[j,i] * (b[u,u] - b[v,u]) = a[j,i] * (buu - bvu)
        # In S5, S6: a[i,i] * (b[u,v] - b[u,u]) = a[i,i] * (buv - buu)
        # In S7, S8: a[j,i] * (b[u,u] - b[u,v]) = a[j,i] * (buu - buv)
        
        # Correction for k=j:
        # In S1, S2: a[i,j] * (b[v,v] - b[u,v]) = a[i,j] * (bvv - buv)
        # In S3, S4: a[j,j] * (b[u,v] - b[v,v]) = a[j,j] * (buv - bvv)
        # In S5, S6: a[i,j] * (b[v,v] - b[v,u]) = a[i,j] * (bvv - bvu)
        # In S7, S8: a[j,j] * (b[v,u] - b[v,v]) = a[j,j] * (bvu - bvv)
        
        corr_i = a[i,i]*(bvu - buu) + a[j,i]*(buu - bvu) + a[i,i]*(buv - buu) + a[j,i]*(buu - buv)
        corr_j = a[i,j]*(bvv - buv) + a[j,j]*(buv - bvv) + a[i,j]*(bvv - bvu) + a[j,j]*(bvu - bvv)
        
        delta -= corr_i
        delta -= corr_j
        
        # Add the direct interaction terms for (i,i), (j,j), (i,j), (j,i)
        # The swap changes:
        # b[p[i], p[i]] from buu to bvv -> a[i,i]*(bvv-buu)
        # b[p[j], p[j]] from bvv to buu -> a[j,j]*(buu-bvv)
        # b[p[i], p[j]] from buv to bvu -> a[i,j]*(bvu-buv)
        # b[p[j], p[i]] from bvu to buv -> a[j,i]*(buv-bvu)
        
        delta += a[i,i]*(bvv - buu)
        delta += a[j,j]*(buu - bvv)
        delta += a[i,j]*(bvu - buv)
        delta += a[j,i]*(buv - bvu)
        
        return delta

    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        total = 0.0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    def local_search(p, cost, time_limit, batch_size=200):
        """
        Hill climbing with best-of-batch move selection.
        """
        p = p.copy()
        improved = True
        no_improve_count = 0
        max_no_improve = 5000 # If no improvement in this many batches, break
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Evaluate a batch of random swaps and pick the best one
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            for _ in range(batch_size):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
            
            if best_i != -1 and best_delta < -1e-9:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        return p, cost

    # --- Initial solution: rank-matching + Hungarian ---
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
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

    # Phase 1: Initial local search
    phase1_deadline = deadline - 40.0
    best_p, best_cost = local_search(best_p, best_cost, phase1_deadline, batch_size=100)
    
    # Phase 2: Hungarian refinement rounds
    for round_num in range(3):
        if time.time() > deadline - 15.0:
            break
        
        # Build linear cost matrix based on current best_p
        b_col_p = b[:, best_p]
        b_row_p = b[best_p, :]
        
        # Linearization: C[i, j] = sum_k a[i, k] b[j, p[k]] + sum_k a[k, i] b[p[k], j]
        # This is an approximation of the marginal cost of assigning facility i to location j
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
        
        # Local search after Hungarian
        if time.time() < deadline - 10.0:
            best_p, best_cost = local_search(best_p, best_cost, deadline - 10.0, batch_size=100)
            
    # Phase 3: Final polish with smaller batches for finer tuning
    if time.time() < deadline - 2.0:
        best_p, best_cost = local_search(best_p, best_cost, deadline - 1.0, batch_size=20)

    return best_p.tolist()
