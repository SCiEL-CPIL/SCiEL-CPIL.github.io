"""QAP solver: spectral init + Hungarian + simulated annealing local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Hungarian-based linearization refinement.
    3. Simulated annealing local search with O(n) incremental updates per swap.
    4. Multi-start with diverse initial solutions.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    np.random.seed(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        b_p = b[p]
        return np.dot(a.ravel(), (b_p * a[:, :, None].sum(axis=0)).ravel())
    
    def compute_cost_v2(p):
        """Compute QAP cost using matrix multiplication."""
        b_p = b[p]
        return np.sum(a * b_p)
    
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
    
    # --- Simulated Annealing Local Search with O(n) incremental updates ---
    def sa_local_search(p, cost, time_limit, T0=None, T_end=None):
        """Perform simulated annealing with fast incremental delta updates."""
        p = p.copy()
        
        # Precompute:
        # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        dot_row = a @ b_col_p.T
        dot_col = a.T @ b_row_p
        
        # Temperature schedule
        if T0 is None:
            T0 = abs(cost) * 0.001 + 1.0
        if T_end is None:
            T_end = 0.001
        
        T = T0
        cooling_rate = 0.9995
        min_T = T_end
        
        # Precompute difference vectors for faster updates
        a_cols = a  # (n, n)
        b_rows = b  # (n, n)
        
        elapsed_start = time.time()
        total_time = time_limit - elapsed_start
        
        while time.time() < time_limit:
            batch_size = 5000
            for _ in range(batch_size):
                if time.time() > time_limit:
                    break
                
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Correct delta computation:
                # Delta = sum_{k} a[i,k]*(b[v,p[k]]-b[u,p[k]]) + sum_{k} a[j,k]*(b[u,p[k]]-b[v,p[k]])
                #       + sum_{k} a[k,i]*(b[p[k],v]-b[p[k],u]) + sum_{k} a[k,j]*(b[p[k],u]-b[p[k],v])
                #       - 2*(a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) 
                #              + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v]))
                # 
                # Using dot_row and dot_col:
                # dot_row[i,v] - dot_row[i,u] = sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]])
                # dot_row[j,u] - dot_row[j,v] = sum_k a[j,k]*(b[u,p[k]]-b[v,p[k]])
                # dot_col[i,v] - dot_col[i,u] = sum_k a[k,i]*(b[p[k],v]-b[p[k],u])
                # dot_col[j,u] - dot_col[j,v] = sum_k a[k,j]*(b[p[k],u]-b[p[k],v])
                #
                # But these include k=i and k=j terms that need correction.
                # For k=i in dot_row[i,*]: a[i,i]*(b[*,u] vs b[*,v]) - when p[i]=u, term is a[i,i]*b[*,u]
                # After swap p[i]=v, so we need to subtract the old and add new.
                #
                # Correct formula:
                delta = (dot_row[i, v] - dot_row[i, u]) + (dot_row[j, u] - dot_row[j, v])
                delta += (dot_col[i, v] - dot_col[i, u]) + (dot_col[j, u] - dot_col[j, v])
                # Subtract double-counted diagonal terms:
                buv = b[u, v]
                bvu = b[v, u]
                buu = b[u, u]
                bvv = b[v, v]
                delta -= 2.0 * (a[i, i] * (bvv - buu) + a[i, j] * (bvu - buv)
                               + a[j, i] * (buv - bvu) + a[j, j] * (buu - bvv))
                
                # Accept/reject
                if delta < 0:
                    accept = True
                elif T > min_T:
                    accept = np.random.random() < np.exp(-delta / T)
                else:
                    accept = False
                
                if accept:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    
                    # Incremental update: rank-1 updates
                    diff_a = a[:, i] - a[:, j]
                    diff_b = b[:, v] - b[:, u]
                    dot_row += np.outer(diff_a, diff_b)
                    
                    diff_a_T = a[i, :] - a[j, :]
                    diff_b_T = b[v, :] - b[u, :]
                    dot_col += np.outer(diff_a_T, diff_b_T)
            
            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        return p, cost
    
    # --- Try multiple initial solutions ---
    best_p = None
    best_cost = np.inf
    
    inits = []
    
    # Spectral init - flow
    if time.time() < deadline - 30:
        p1 = spectral_init('flow')
        inits.append(p1)
    
    # Spectral init - distance
    if time.time() < deadline - 30:
        p2 = spectral_init('distance')
        inits.append(p2)
    
    # Spectral init - combined
    if time.time() < deadline - 30:
        p3 = spectral_init('combined')
        inits.append(p3)
    
    # Rank matching
    if time.time() < deadline - 30:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p4 = np.empty(n, dtype=np.int64)
        p4[heavy] = central
        inits.append(p4)
    
    # Random init
    if time.time() < deadline - 30:
        p5 = np.random.permutation(n).astype(np.int64)
        inits.append(p5)
    
    # Evaluate all inits
    for p_init in inits:
        c = compute_cost_v2(p_init)
        if c < best_cost:
            best_cost = c
            best_p = p_init.copy()
    
    # --- Simulated Annealing on best initial solution ---
    if time.time() < deadline - 25:
        best_p, best_cost = sa_local_search(best_p, best_cost, deadline - 25)
    
    # --- Hungarian refinement ---
    if time.time() < deadline - 15:
        b_col_p = b[:, best_p]
        b_row_p = b[best_p, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        cost_hung = compute_cost_v2(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # --- Final SA polish ---
    if time.time() < deadline - 5:
        best_p, best_cost = sa_local_search(best_p, best_cost, deadline - 3, T0=abs(best_cost)*0.0001+0.1, T_end=0.0001)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
