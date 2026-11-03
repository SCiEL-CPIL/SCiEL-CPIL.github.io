"""Combined QAP solver: spectral init + fast incremental local search + Hungarian refinement + multi-restart."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Fast local search with O(n) incremental updates per swap.
    3. Hungarian-based linearization refinement.
    4. Multi-start with diverse initial solutions and perturbation.
    """
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s limit
    np.random.seed(42)
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        b_sub = b[np.ix_(p, p)]
        return int(np.sum(a * b_sub))
    
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
    
    # --- Fast local search with O(n) incremental updates ---
    def local_search(p, cost, time_limit):
        """Perform local search with fast incremental delta updates."""
        p = p.copy()
        
        def recompute_precomp(p):
            b_col_p = b[:, p]
            b_row_p = b[p, :]
            dot_row = a @ b_col_p.T
            dot_col = a.T @ b_row_p
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        b_diag = np.diag(b)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            num_trials = 30000
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
                
                buu = b_diag[u]
                bvv = b_diag[v]
                buv = b[u, v]
                bvu = b[v, u]
                
                delta = 0.0
                delta += a[i, i] * (bvv - buu)
                delta += a[i, j] * (bvu - buv)
                delta += a[j, i] * (buv - bvu)
                delta += a[j, j] * (buu - bvv)
                
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvv - buu) - a[i,j]*(bvu - buv)
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(buv - bvu) - a[j,j]*(buu - bvv)
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(bvv - buu) - a[j,i]*(bvu - buv)
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(buv - bvu) - a[j,j]*(buu - bvv)
                
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cost += int(delta)
                    improved = True
                    
                    diff_a = a[:, i] - a[:, j]
                    diff_b = b[:, v] - b[:, u]
                    dot_row += np.outer(diff_a, diff_b)
                    
                    diff_a2 = a[i, :] - a[j, :]
                    diff_b2 = b[v, :] - b[u, :]
                    dot_col += np.outer(diff_a2, diff_b2)
        
        return p, cost
    
    # --- Hungarian-based refinement ---
    def hungarian_refinement(p):
        """Use Hungarian algorithm on linearized cost matrix."""
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        return p_hung
    
    # --- Main search loop ---
    best_p = None
    best_cost = np.inf
    
    # Try multiple initial solutions
    init_methods = ['flow', 'distance', 'combined', 'default']
    for method in init_methods:
        if time.time() > deadline - 60.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # Also try random initialization
    p_rand = np.random.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost:
        best_cost = cost_rand
        best_p = p_rand.copy()
    
    # Run local search on best initial solution
    time_limit = deadline - 40.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # Hungarian refinement
    if time.time() < deadline - 30.0:
        p_hung = hungarian_refinement(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # Second local search pass
    if time.time() < deadline - 20.0:
        time_limit2 = deadline - 15.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit2)
    
    # Multi-restart with perturbation
    num_restarts = 0
    max_restarts = 20
    while time.time() < deadline - 10.0 and num_restarts < max_restarts:
        # Create perturbed version
        p_restart = best_p.copy()
        num_swaps = max(2, n // 15)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
        
        # Calculate cost
        current_cost_r = compute_cost(p_restart)
        
        # Run local search on perturbed solution
        time_left = deadline - 10.0 - time.time()
        if time_left < 5.0:
            break
        p_restart, current_cost_r = local_search(p_restart, current_cost_r, time_left)
        
        if current_cost_r < best_cost:
            best_cost = current_cost_r
            best_p = p_restart.copy()
        
        num_restarts += 1
    
    # Final Hungarian refinement if time allows
    if time.time() < deadline - 5.0:
        p_hung = hungarian_refinement(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # Final local search
    if time.time() < deadline - 2.0:
        time_limit3 = deadline - 1.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit3)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
