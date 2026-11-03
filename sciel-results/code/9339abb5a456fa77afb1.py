"""Rewritten QAP solver: spectral init + Hungarian + fast incremental local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Hungarian-based linearization refinement.
    3. Fast local search with O(n) incremental updates per swap.
    4. Multi-start with diverse initial solutions.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    np.random.seed(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    # --- Spectral initialization ---
    def spectral_init(method='flow'):
        """Generate initial permutation using spectral methods."""
        if method == 'flow':
            a_centered = a - a.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(a_centered)
                idx = np.argsort(-np.abs(eigvals))
                spec_coords = eigvecs[:, idx[0]]
                fac_rank = np.argsort(spec_coords)
                loc_rank = np.argsort(b.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'distance':
            b_centered = b - b.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(b_centered)
                idx = np.argsort(-np.abs(eigvals))
                spec_coords = eigvecs[:, idx[0]]
                loc_rank = np.argsort(spec_coords)
                fac_rank = np.argsort(a.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'combined':
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            try:
                eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
                eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
                idx_a = np.argsort(-np.abs(eigvals_a))
                idx_b = np.argsort(-np.abs(eigvals_b))
                spec_a = eigvecs_a[:, idx_a[0]]
                spec_b = eigvecs_b[:, idx_b[0]]
                fac_rank = np.argsort(spec_a)
                loc_rank = np.argsort(spec_b)
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        else:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Local search with O(n) incremental updates ---
    def local_search(p, cost, time_limit):
        """Perform local search with fast incremental delta updates."""
        p = p.copy()
        
        # Precompute:
        # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]  = (a @ b[:, p].T)[i, l]
        # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]  = (a.T @ b[p, :])[i, l]
        
        def recompute_precomp(p):
            b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
            b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
            dot_row = a @ b_col_p.T
            dot_col = a.T @ b_row_p
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        # Precompute constant terms for faster delta
        # b_ii[l] = b[l, l] for all l
        b_diag = np.diag(b)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            num_trials = 20000
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Compute delta using precomputed dot products (O(1) lookups + O(1) arithmetic)
                buu = b_diag[u]
                bvv = b_diag[v]
                buv = b[u, v]
                bvu = b[v, u]
                
                delta = 0.0
                # Terms 1-4
                delta += a[i, i] * (bvv - buu)
                delta += a[i, j] * (bvu - buv)
                delta += a[j, i] * (buv - bvu)
                delta += a[j, j] * (buu - bvv)
                
                # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
                # = dot_row[i, v] - dot_row[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvv - buu) - a[i,j]*(bvu - buv)
                
                # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
                # = dot_row[j, u] - dot_row[j, v] - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(buv - bvu) - a[j,j]*(buu - bvv)
                
                # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
                # = dot_col[i, v] - dot_col[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(bvv - buu) - a[j,i]*(bvu - buv)
                
                # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
                # = dot_col[j, u] - dot_col[j, v] - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(buv - bvu) - a[j,j]*(buu - bvv)
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Incremental update of dot_row and dot_col in O(n)
                    # When p[i] and p[j] swap (u <-> v):
                    #
                    # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
                    # Only p[i] and p[j] changed.
                    # For k != i,j: dot_row[k, l] changes by:
                    #   a[k,i]*(b[l,v] - b[l,u]) + a[k,j]*(b[l,u] - b[l,v])
                    #   = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    # For k = i: dot_row[i, l] changes by:
                    #   a[i,i]*(b[l,v] - b[l,u])  [p[i] changed from u to v]
                    #   + a[i,j]*(b[l,u] - b[l,v])  [p[j] changed from v to u]
                    #   = a[i,i]*(b[l,v] - b[l,u]) - a[i,j]*(b[l,v] - b[l,u])
                    #   = (a[i,i] - a[i,j]) * (b[l,v] - b[l,u])
                    # For k = j: dot_row[j, l] changes by:
                    #   a[j,i]*(b[l,u] - b[l,v]) + a[j,j]*(b[l,v] - b[l,u])
                    #   = (a[j,j] - a[j,i]) * (b[l,v] - b[l,u])
                    #
                    # So for ALL k: dot_row[k, l] += (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    # Wait, let me re-derive more carefully.
                    #
                    # Before swap: p[i]=u, p[j]=v
                    # After swap:  p[i]=v, p[j]=u
                    #
                    # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
                    # Change = a[k,i]*(b[l,v] - b[l,u]) + a[k,j]*(b[l,u] - b[l,v])
                    #        = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    #
                    # So: dot_row[:, :] += outer(a[:, i] - a[:, j], b[:, v] - b[:, u])
                    # This is O(n^2) which is fine.
                    
                    diff_a = a[:, i] - a[:, j]  # shape (n,)
                    diff_b = b[:, v] - b[:, u]  # shape (n,)
                    dot_row += np.outer(diff_a, diff_b)
                    
                    # Similarly for dot_col:
                    # dot_col[k, l] = sum_m a[m,k] * b[p[m], l]
                    # Change = a[i,k]*(b[v,l] - b[u,l]) + a[j,k]*(b[u,l] - b[v,l])
                    #        = (a[i,k] - a[j,k]) * (b[v,l] - b[u,l])
                    # So: dot_col[:, :] += outer(a[i, :] - a[j, :], b[v, :] - b[u, :])
                    
                    diff_a2 = a[i, :] - a[j, :]  # shape (n,)
                    diff_b2 = b[v, :] - b[u, :]  # shape (n,)
                    dot_col += np.outer(diff_a2, diff_b2)
        
        return p, cost
    
    # --- Try multiple initial solutions and keep the best ---
    init_methods = ['flow', 'distance', 'combined', 'default']
    best_p = None
    best_cost = np.inf
    
    for method in init_methods:
        if time.time() > deadline - 40.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # Also try a random initialization
    p_rand = np.random.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost:
        best_cost = cost_rand
        best_p = p_rand.copy()
    
    # --- Run local search on best initial solution ---
    time_limit = deadline - 25.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Hungarian-based refinement ---
    if time.time() < deadline - 20.0:
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
    
    # --- Second local search pass ---
    if time.time() < deadline - 10.0:
        time_limit2 = deadline - 5.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit2)
    
    # --- Another Hungarian refinement if time allows ---
    if time.time() < deadline - 8.0:
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
    
    # --- Final local search ---
    if time.time() < deadline - 3.0:
        time_limit3 = deadline - 1.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit3)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
