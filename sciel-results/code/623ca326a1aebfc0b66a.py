"""QAP solver: spectral init + systematic 2-opt local search with incremental updates."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Multiple diverse initializations (spectral, rank-based, random).
    2. Systematic 2-opt local search with O(n) incremental delta updates.
    3. Hungarian-based refinement.
    4. Final polish with tighter local search.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    np.random.seed(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        b_p_p = b[np.ix_(p, p)]
        return int(np.sum(a * b_p_p))
    
    def time_left():
        return deadline - time.time()
    
    # --- Initialization methods ---
    def spectral_init_flow():
        """Spectral init: match flow eigenvector to location centrality."""
        a_centered = a - a.mean()
        eigvals, eigvecs = np.linalg.eigh(a_centered)
        idx = np.argsort(-np.abs(eigvals))
        spec_coords = eigvecs[:, idx[0]]
        fac_rank = np.argsort(spec_coords)
        loc_rank = np.argsort(b.sum(axis=1))
        p = np.empty(n, dtype=np.int64)
        p[fac_rank] = loc_rank
        return p
    
    def spectral_init_dist():
        """Spectral init: match distance eigenvector to flow centrality."""
        b_centered = b - b.mean()
        eigvals, eigvecs = np.linalg.eigh(b_centered)
        idx = np.argsort(-np.abs(eigvals))
        spec_coords = eigvecs[:, idx[0]]
        loc_rank = np.argsort(spec_coords)
        fac_rank = np.argsort(a.sum(axis=1))
        p = np.empty(n, dtype=np.int64)
        p[fac_rank] = loc_rank
        return p
    
    def rank_init():
        """Rank-based: heavy flows to central locations."""
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p
    
    def random_init(seed):
        """Random permutation."""
        np.random.seed(seed)
        p = np.random.permutation(n).astype(np.int64)
        return p
    
    # --- Local search with systematic 2-opt passes ---
    def local_search(p, cost, time_limit, systematic=True):
        """Perform local search with fast incremental delta updates.
        
        If systematic=True, do full passes over all pairs (i,j) with i<j.
        Otherwise, use random sampling.
        """
        p = p.copy()
        
        # Precompute dot products
        def recompute_precomp(p):
            b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
            b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
            dot_row = a @ b_col_p.T
            dot_col = a.T @ b_row_p
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        b_diag = np.diag(b).copy()
        
        # Precompute swap deltas more efficiently
        # For swap (i,j) with u=p[i], v=p[j]:
        # delta = sum of 8 terms as in original
        
        def compute_swap_delta(i, j, p, dot_row, dot_col):
            u = p[i]
            v = p[j]
            if u == v:
                return 0.0
            
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
            
            return delta
        
        def apply_swap(i, j, p, dot_row, dot_col):
            u = p[i]
            v = p[j]
            p[i], p[j] = v, u
            
            diff_a_col = a[:, i] - a[:, j]
            diff_b_row = b[:, v] - b[:, u]
            dot_row += np.outer(diff_a_col, diff_b_row)
            
            diff_a_row = a[i, :] - a[j, :]
            diff_b_col = b[v, :] - b[u, :]
            dot_col += np.outer(diff_a_row, diff_b_col)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            if systematic:
                # Systematic pass: try all pairs (i, j) with i < j
                for i in range(n):
                    if time.time() > time_limit:
                        return p, cost
                    for j in range(i + 1, n):
                        if time.time() > time_limit:
                            return p, cost
                        
                        u = p[i]
                        v = p[j]
                        if u == v:
                            continue
                        
                        delta = compute_swap_delta(i, j, p, dot_row, dot_col)
                        
                        if delta < -1e-9:
                            apply_swap(i, j, p, dot_row, dot_col)
                            cost += delta
                            improved = True
            else:
                # Random sampling
                num_trials = 50000
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
                    
                    delta = compute_swap_delta(i, j, p, dot_row, dot_col)
                    
                    if delta < -1e-9:
                        apply_swap(i, j, p, dot_row, dot_col)
                        cost += delta
                        improved = True
        
        return p, cost
    
    # --- Generate diverse initial solutions ---
    init_solutions = []
    
    # Spectral inits
    for init_func in [spectral_init_flow, spectral_init_dist, rank_init]:
        if time_left() < 30.0:
            break
        p_init = init_func()
        cost_init = compute_cost(p_init)
        init_solutions.append((p_init, cost_init))
    
    # Random inits
    for seed in [0, 1, 2, 3, 4]:
        if time_left() < 30.0:
            break
        p_init = random_init(seed)
        cost_init = compute_cost(p_init)
        init_solutions.append((p_init, cost_init))
    
    # Sort by cost
    init_solutions.sort(key=lambda x: x[1])
    
    # --- Run local search on top diverse solutions ---
    # Keep solutions that are sufficiently different from each other
    selected = []
    for p_init, cost_init in init_solutions:
        if time_left() < 25.0:
            break
        # Check diversity: Hamming distance from selected solutions
        diverse = True
        for p_sel, _ in selected:
            hamming = np.sum(p_init != p_sel)
            if hamming < n * 0.3:  # Less than 30% different
                diverse = False
                break
        if diverse:
            selected.append((p_init, cost_init))
        if len(selected) >= 3:
            break
    
    # If no diverse solutions, take the best
    if not selected:
        selected = [init_solutions[0]]
    
    best_p, best_cost = None, np.inf
    
    # Run local search on each selected solution
    for idx, (p_init, cost_init) in enumerate(selected):
        if time_left() < 20.0:
            break
        
        # Allocate time proportional to remaining budget
        remaining_solutions = len(selected) - idx
        time_per_solution = min(time_left() - 5.0, 40.0) / remaining_solutions
        time_limit = time.time() + time_per_solution
        
        p_ls, cost_ls = local_search(p_init, cost_init, time_limit, systematic=True)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # --- Hungarian-based refinement ---
    if best_p is not None and time_left() > 10.0:
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
    
    # --- Final polish with systematic local search ---
    if best_p is not None and time_left() > 5.0:
        time_limit = time.time() + min(time_left() - 3.0, 15.0)
        best_p, best_cost = local_search(best_p, best_cost, time_limit, systematic=True)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
