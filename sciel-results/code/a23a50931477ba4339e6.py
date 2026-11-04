"""Rewritten QAP solver: spectral initialization + Hungarian + optimized local search with incremental updates."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Hungarian-based linearization refinement.
    3. Fast local search with O(n^2) incremental updates for precomputed dot products.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        # Vectorized computation: sum_i a[i, :] dot b[p[i], p]
        # b[p, p] is the matrix B' where B'[i, j] = b[p[i], p[j]]
        # Cost = sum_i sum_j a[i, j] B'[i, j] = trace(a.T @ B') or sum(a * B')
        # But B' depends on p.
        # Efficient way:
        # b_p_p = b[np.ix_(p, p)]
        # cost = np.sum(a * b_p_p)
        # This creates an n x n array, which is fine for n=150.
        b_p_p = b[np.ix_(p, p)]
        return np.sum(a * b_p_p)
    
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
        else:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Try multiple initial solutions and keep the best ---
    init_methods = ['flow', 'distance', 'default']
    best_p = None
    best_cost = np.inf
    
    for method in init_methods:
        if time.time() > deadline - 30.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # --- Local search with precomputed dot products and incremental updates ---
    def local_search(p, cost, time_limit):
        """Perform local search starting from p with initial cost."""
        p = p.copy()
        
        # Precompute dot_row and dot_col
        # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        
        def recompute_precomp(p):
            b_col_p = b[:, p]  # [l, k] = b[l, p[k]]
            b_row_p = b[p, :]  # [k, l] = b[p[k], l]
            dot_row = a @ b_col_p.T  # [i, l] = sum_k a[i,k] b[l, p[k]]
            dot_col = a.T @ b_row_p  # [i, l] = sum_k a[k,i] b[p[k], l]
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Random sampling of pairs
            num_trials = 10000
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
                
                # Compute delta using precomputed dot products
                delta = 0.0
                # Diagonal and cross terms
                delta += a[i, i] * (b[v, v] - b[u, u])
                delta += a[i, j] * (b[v, u] - b[u, v])
                delta += a[j, i] * (b[u, v] - b[v, u])
                delta += a[j, j] * (b[u, u] - b[v, v])
                
                # Off-diagonal terms using precomputed dots
                # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                
                # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                
                # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Incremental update of dot_row and dot_col
                    # dot_row[k, l] += (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    # dot_col[k, l] += (a[i,k] - a[j,k]) * (b[v,l] - b[u,l])
                    
                    # Precompute difference vectors
                    diff_a_col_i_j = a[:, i] - a[:, j]  # shape (n,)
                    diff_b_row_v_u = b[:, v] - b[:, u]  # shape (n,)
                    # Update dot_row: outer product
                    dot_row += np.outer(diff_a_col_i_j, diff_b_row_v_u)
                    
                    diff_a_row_i_j = a[i, :] - a[j, :]  # shape (n,)
                    diff_b_col_v_u = b[v, :] - b[u, :]  # shape (n,)
                    # Update dot_col: outer product
                    dot_col += np.outer(diff_a_row_i_j, diff_b_col_v_u)
        
        return p, cost
    
    # Run local search
    time_limit = deadline - 20.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Hungarian-based refinement ---
    if time.time() < deadline - 15.0:
        # Build linear cost matrix
        b_col_p = b[:, best_p]  # shape (n, n), [j, k] = b[j, p[k]]
        b_row_p = b[best_p, :]  # shape (n, n), [k, j] = b[p[k], j]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # --- Final local search pass ---
    if time.time() < deadline - 5.0:
        time_limit = deadline - 3.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
