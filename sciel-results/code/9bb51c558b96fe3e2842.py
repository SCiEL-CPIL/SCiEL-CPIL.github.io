"""QAP solver: spectral init + linearized greedy construction + incremental local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors.
    2. Linearized greedy construction for diverse starting points.
    3. Fast local search with O(n) incremental updates per swap.
    4. Hungarian-based refinement if time permits.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    np.random.seed(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
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
    
    # --- Linearized Greedy Construction ---
    def greedy_construction(seed=None):
        """Build permutation greedily using dynamic cost matrix."""
        if seed is not None:
            np.random.seed(seed)
        
        p = np.full(n, -1, dtype=np.int64)
        unassigned_fac = set(range(n))
        unassigned_loc = set(range(n))
        
        # C[i, j] = marginal cost of assigning facility i to location j
        # Given partial assignment, C[i,j] = sum_{k assigned} a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j]
        # For unassigned i, j: C[i,j] = sum_{k in assigned} (a[i,k] + a[k,i]) * b[j, p[k]]
        # We maintain this incrementally.
        
        # Start with a random or heuristic first assignment
        # Pick the facility with highest total flow
        if unassigned_fac:
            # Choose first facility-location pair by a heuristic
            # Use the facility with highest row sum and location with lowest column sum
            fac_sums = a.sum(axis=1)
            loc_sums = b.sum(axis=1)
            
            # Pick facility with max flow
            first_fac = max(unassigned_fac, key=lambda x: fac_sums[x])
            # Pick location with min distance sum
            first_loc = min(unassigned_loc, key=lambda x: loc_sums[x])
            
            p[first_fac] = first_loc
            unassigned_fac.remove(first_fac)
            unassigned_loc.remove(first_loc)
            
            # Initialize cost matrix for remaining
            # C[i, j] for i in unassigned_fac, j in unassigned_loc
            # C[i, j] = (a[i, first_fac] + a[first_fac, i]) * b[j, first_loc]
            # But we need to be careful: the full cost contribution when we assign i to j
            # and later assign k to l includes terms like a[i,k]*b[j,l] and a[k,i]*b[l,j]
            # The greedy approach approximates this by only considering interactions with already assigned.
            
            # Maintain C as a dict or sparse structure
            # For efficiency, maintain C as a 2D array for unassigned
            # But n=150, so we can use a full array and mask
            
            C = np.zeros((n, n), dtype=np.float64)
            # C[i, j] = sum_{k assigned} (a[i,k] + a[k,i]) * b[j, p[k]]
            # For first_fac assigned to first_loc:
            for i in unassigned_fac:
                C[i, :] = (a[i, first_fac] + a[first_fac, i]) * b[:, first_loc]
            # Only care about unassigned locations
            for j in unassigned_loc:
                pass  # C[:, j] is already set for unassigned i
            
            # Now greedily assign remaining
            for step in range(1, n):
                if time.time() > deadline - 30.0:
                    break
                if not unassigned_fac or not unassigned_loc:
                    break
                
                # Find best (i, j) pair in unassigned
                # Use a simple scan for now
                best_val = np.inf
                best_i = -1
                best_j = -1
                for i in unassigned_fac:
                    for j in unassigned_loc:
                        val = C[i, j]
                        if val < best_val:
                            best_val = val
                            best_i = i
                            best_j = j
                
                if best_i == -1:
                    break
                
                # Assign best_i to best_j
                p[best_i] = best_j
                unassigned_fac.remove(best_i)
                unassigned_loc.remove(best_j)
                
                # Update C for remaining unassigned facilities and locations
                # C[i, j] += (a[i, best_i] + a[best_i, i]) * b[j, best_j]
                for i in unassigned_fac:
                    diff = (a[i, best_i] + a[best_i, i]) * b[:, best_j]
                    for j in unassigned_loc:
                        C[i, j] += diff[j]
        
        # If we broke early, fill in randomly
        if len(unassigned_fac) > 0:
            remaining_locs = list(unassigned_loc)
            np.random.shuffle(remaining_locs)
            for i in unassigned_fac:
                p[i] = remaining_locs.pop()
        
        return p
    
    # --- Local search with O(n) incremental updates ---
    def local_search(p, cost, time_limit):
        """Perform local search with fast incremental delta updates."""
        p = p.copy()
        
        # Precompute:
        # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        
        def recompute_precomp(p):
            b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
            b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
            dot_row = a @ b_col_p.T
            dot_col = a.T @ b_row_p
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        # Precompute diagonal of b
        b_diag = np.diag(b).copy()
        
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
                
                # Compute delta using precomputed dot products
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
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvv - buu) - a[i,j]*(bvu - buv)
                
                # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(buv - bvu) - a[j,j]*(buu - bvv)
                
                # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(bvv - buu) - a[j,i]*(bvu - buv)
                
                # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(buv - bvu) - a[j,j]*(buu - bvv)
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Incremental update of dot_row and dot_col in O(n)
                    # dot_row[k, l] += (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    diff_a_col = a[:, i] - a[:, j]  # shape (n,)
                    diff_b_row = b[:, v] - b[:, u]  # shape (n,)
                    dot_row += np.outer(diff_a_col, diff_b_row)
                    
                    # dot_col[k, l] += (a[i,k] - a[j,k]) * (b[v,l] - b[u,l])
                    diff_a_row = a[i, :] - a[j, :]  # shape (n,)
                    diff_b_col = b[v, :] - b[u, :]  # shape (n,)
                    dot_col += np.outer(diff_a_row, diff_b_col)
        
        return p, cost
    
    # --- Try multiple initial solutions and keep the best ---
    best_p = None
    best_cost = np.inf
    
    # Spectral inits
    init_methods = ['flow', 'distance', 'default']
    for method in init_methods:
        if time.time() > deadline - 40.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # Greedy constructions with different seeds
    for seed in [0, 1, 2]:
        if time.time() > deadline - 30.0:
            break
        p_init = greedy_construction(seed=seed)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # Run local search on the best initial solution
    if best_p is not None:
        time_limit = deadline - 20.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Hungarian-based refinement ---
    if time.time() < deadline - 15.0:
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
    
    # --- Final local search pass ---
    if time.time() < deadline - 5.0:
        time_limit = deadline - 3.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
