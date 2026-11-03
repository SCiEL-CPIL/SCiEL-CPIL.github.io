"""QAP solver: spectral init + vectorized 2-opt local search with restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP using spectral initialization, Hungarian refinement, and vectorized local search."""
    n = len(a)
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return int(np.sum(a[:, :, None, None] * b[p[:, None], p[None, :]]))
    
    def spectral_init():
        """Generate initial permutation using spectral methods on flow/distance matrices."""
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        try:
            eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
            eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
            
            # Top eigenvector of flow, bottom eigenvector of distance
            spec_fac = eigvecs_a[:, np.argmax(eigvals_a)]
            spec_loc = eigvecs_b[:, np.argmin(eigvals_b)]
            
            fac_order = np.argsort(-spec_fac, kind="stable")
            loc_order = np.argsort(spec_loc, kind="stable")
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p
        except:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    def rank_init():
        """Rank-based initialization: match heavy facilities to peripheral locations."""
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        peripheral = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = peripheral
        return p
    
    def hungarian_refine(p):
        """Use Hungarian algorithm to refine the assignment."""
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        # Linearized cost matrix: C[i,l] approximates marginal cost of putting facility i at location l
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new
    
    def vectorized_local_search(p, cost, time_limit):
        """
        Local search with correct O(n) delta computation per swap.
        
        Key formula for delta when swapping p[i]=u and p[j]=v:
        delta = sum_{k not in {i,j}} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[k,i]*(b[p[k],v]-b[p[k],u])
                                     + a[j,k]*(b[u,p[k]]-b[v,p[k]]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
              + a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v])
              + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
        
        We maintain:
          S_row[i, l] = sum_k a[i,k] * b[l, p[k]]  for all i, l
          S_col[i, l] = sum_k a[k,i] * b[p[k], l]  for all i, l
        
        Then:
          sum_{k not in {i,j}} a[i,k]*(b[v,p[k]]-b[u,p[k]]) = S_row[i,v] - S_row[i,u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
          sum_{k not in {i,j}} a[k,i]*(b[p[k],v]-b[p[k],u]) = S_col[i,v] - S_col[i,u] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
          sum_{k not in {i,j}} a[j,k]*(b[u,p[k]]-b[v,p[k]]) = S_row[j,u] - S_row[j,v] - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
          sum_{k not in {i,j}} a[k,j]*(b[p[k],u]-b[p[k],v]) = S_col[j,u] - S_col[j,v] - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        """
        p = p.copy()
        
        # Initialize S_row and S_col
        b_at_p = b[p, :]       # b_at_p[k, l] = b[p[k], l]
        b_at_p_T = b[:, p]     # b_at_p_T[l, k] = b[l, p[k]]
        
        # S_row[i, l] = sum_k a[i,k] * b[l, p[k]] = (a @ b_at_p_T.T)[i, l]
        S_row = a @ b_at_p_T.T   # shape (n, n): [i, l]
        # S_col[i, l] = sum_k a[k,i] * b[p[k], l] = (a.T @ b_at_p)[i, l]
        S_col = a.T @ b_at_p     # shape (n, n): [i, l]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try a batch of random swaps
            n_trials = 50000
            for _ in range(n_trials):
                if time.time() > time_limit:
                    break
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Compute delta
                # Cross terms
                aii = a[i, i]
                aij = a[i, j]
                aji = a[j, i]
                ajj = a[j, j]
                buu = b[u, u]
                buv = b[u, v]
                bvu = b[v, u]
                bvv = b[v, v]
                
                # Diagonal and direct cross terms
                diag_cross = aii * (bvv - buu) + aij * (bvu - buv) + aji * (buv - bvu) + ajj * (buu - bvv)
                
                # Off-diagonal via S_row and S_col
                sr_iv = S_row[i, v]
                sr_iu = S_row[i, u]
                sc_iv = S_col[i, v]
                sc_iu = S_col[i, u]
                sr_ju = S_row[j, u]
                sr_jv = S_row[j, v]
                sc_ju = S_col[j, u]
                sc_jv = S_col[j, v]
                
                off_diag = (sr_iv - sr_iu - aii * (bvv - buu) - aij * (bvu - buv)) \
                         + (sc_iv - sc_iu - aii * (bvv - buu) - aji * (bvu - buv)) \
                         + (sr_ju - sr_jv - aji * (buv - bvu) - ajj * (buu - bvv)) \
                         + (sc_ju - sc_jv - aij * (buv - bvu) - ajj * (buu - bvv))
                
                delta = diag_cross + off_diag
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = v, u
                    cost += delta
                    improved = True
                    
                    # Update S_row and S_col incrementally
                    # S_row[i, :] changes because p[i] changed from u to v
                    # S_row[i, l] = sum_k a[i,k]*b[l,p[k]]
                    # Only k=i and k=j terms change
                    # S_row[i, l] += a[i,i]*(b[l,v]-b[l,u]) + a[i,j]*(b[l,u]-b[l,v])
                    #             = (a[i,i]-a[i,j])*(b[l,v]-b[l,u])
                    S_row[i, :] += (aii - aij) * (b[:, v] - b[:, u])
                    
                    # S_row[j, :] changes because p[j] changed from v to u
                    # S_row[j, l] += (a[j,j]-a[j,i])*(b[l,u]-b[l,v])
                    S_row[j, :] += (ajj - aji) * (b[:, u] - b[:, v])
                    
                    # For all k, S_row[k, u] and S_row[k, v] change because
                    # b[u, p[k]] and b[v, p[k]] are affected when k=i or k=j
                    # S_row[k, u] = sum_m a[k,m]*b[u,p[m]]
                    # After swap: S_row_new[k, u] = S_row_old[k, u] - a[k,i]*b[u,u] - a[k,j]*b[u,v] + a[k,i]*b[u,v] + a[k,j]*b[u,u]
                    #            = S_row_old[k, u] + (a[k,j]-a[k,i])*(b[u,u]-b[u,v])
                    diff_a = a[:, j] - a[:, i]
                    S_row[:, u] += diff_a * (buu - buv)
                    S_row[:, v] += diff_a * (bvv - bvu)
                    
                    # S_col[i, :] changes because p[i] changed from u to v
                    # S_col[i, l] = sum_k a[k,i]*b[p[k],l]
                    # S_col[i, l] += (a[i,i]-a[j,i])*(b[v,l]-b[u,l])
                    S_col[i, :] += (aii - aji) * (b[v, :] - b[u, :])
                    
                    # S_col[j, :] changes because p[j] changed from v to u
                    # S_col[j, l] += (a[j,j]-a[i,j])*(b[u,l]-b[v,l])
                    S_col[j, :] += (ajj - aij) * (b[u, :] - b[v, :])
                    
                    # For all k, S_col[k, u] and S_col[k, v] change
                    # S_col[k, u] = sum_m a[m,k]*b[p[m],u]
                    # S_col_new[k, u] = S_col_old[k, u] + (a[j,k]-a[i,k])*(b[u,u]-b[v,u])
                    # S_col_new[k, v] = S_col_old[k, v] + (a[j,k]-a[i,k])*(b[v,v]-b[u,v])
                    S_col[:, u] += diff_a * (buu - bvu)
                    S_col[:, v] += diff_a * (bvv - buv)
        
        return p, cost
    
    # --- Generate initial solutions ---
    best_p = None
    best_cost = np.inf
    
    # Spectral init
    p1 = spectral_init()
    c1 = compute_cost(p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
    
    # Rank init
    if time.time() < deadline - 30:
        p2 = rank_init()
        c2 = compute_cost(p2)
        if c2 < best_cost:
            best_cost = c2
            best_p = p2.copy()
    
    # Hungarian refinement of best
    if time.time() < deadline - 20:
        p3 = hungarian_refine(best_p)
        c3 = compute_cost(p3)
        if c3 < best_cost:
            best_cost = c3
            best_p = p3.copy()
    
    # --- Local search ---
    if time.time() < deadline - 10:
        best_p, best_cost = vectorized_local_search(best_p, best_cost, deadline - 10)
    
    # --- Hungarian refinement again ---
    if time.time() < deadline - 5:
        p4 = hungarian_refine(best_p)
        c4 = compute_cost(p4)
        if c4 < best_cost:
            best_cost = c4
            best_p = p4.copy()
    
    # --- Final local search ---
    if time.time() < deadline - 2:
        best_p, best_cost = vectorized_local_search(best_p, best_cost, deadline - 1)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
