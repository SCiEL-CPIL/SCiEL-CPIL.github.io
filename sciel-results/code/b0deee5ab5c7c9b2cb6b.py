"""Combined QAP solver: MDS/Procrustes init + efficient local search + SA polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Convert to float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        """Compute exact cost."""
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        """Classical MDS embedding."""
        n_local = D.shape[0]
        J = np.eye(n_local) - np.ones((n_local, n_local)) / n_local
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        vals = np.maximum(vals, 0)
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y):
        """Align X to Y and solve assignment problem."""
        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        X_aligned = X @ Q
        
        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross
        
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def efficient_local_search(p, current_cost, end_time, seed=12345):
        """
        Efficient 2-swap local search with precomputed row/col contributions.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        # List of all pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > end_time:
                    break
                
                li = curr_p[i]
                lj = curr_p[j]
                
                drow = b_f[lj, curr_p] - b_f[li, curr_p]
                dcol = b_f[curr_p, lj] - b_f[curr_p, li]
                
                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]
                
                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
                
                if delta < -1e-10:
                    curr_p[i], curr_p[j] = lj, li
                    curr_cost += delta
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True
            
            if not improved:
                break
                
        return best_p, best_cost

    # --- Initialization Strategies ---
    inits = []
    costs = []

    # 1. MDS + Procrustes Initialization
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0
    
    for dim in [2, 3, 5]:
        if time.time() > deadline - 50.0:
            break
        try:
            X = mds_embedding(a_sym, dim)
            Y = mds_embedding(b_sym, dim)
            p_try = procrustes_alignment(X, Y)
            cost_try = compute_cost(p_try)
            inits.append(p_try)
            costs.append(cost_try)
        except Exception:
            pass

    # 2. Rank Matching Initialization
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    inits.append(p_rank)
    costs.append(cost_rank)

    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    cost_rank_rev = compute_cost(p_rank_rev)
    inits.append(p_rank_rev)
    costs.append(cost_rank_rev)

    # 3. Hungarian Algorithm Initialization (Linear Relaxation)
    try:
        if time.time() < deadline - 5.0:
            C = a_f @ b_f
            row_ind, col_ind = linear_sum_assignment(C)
            p_hung = np.empty(n, dtype=np.int64)
            p_hung[row_ind] = col_ind
            cost_hung = compute_cost(p_hung)
            inits.append(p_hung)
            costs.append(cost_hung)
    except Exception:
        pass

    # 4. Spectral initialization
    try:
        a_centered = a_f - a_f.mean()
        b_centered = b_f - b_f.mean()
        
        a_scale = np.sqrt(np.mean(a_centered ** 2)) + 1e-10
        b_scale = np.sqrt(np.mean(b_centered ** 2)) + 1e-10
        
        a_norm = a_centered / a_scale
        b_norm = b_centered / b_scale
        
        C = 0.5 * (a_norm + a_norm.T) + 0.5 * (b_norm + b_norm.T)
        C = 0.5 * (C + C.T)
        
        k = 5
        if time.time() < deadline - 30.0:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigvecs = eigenvectors[:, n-k:]
            
            for idx in range(k):
                eigvec = eigvecs[:, idx]
                facility_order = np.argsort(eigvec, kind="stable")
                location_order = np.argsort(-eigvec, kind="stable")
                
                p = np.empty(n, dtype=np.int64)
                p[facility_order] = location_order
                cost_p = compute_cost(p)
                inits.append(p)
                costs.append(cost_p)
    except Exception:
        pass

    # 5. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(3):
        p_rand = rng.permutation(n)
        cost_rand = compute_cost(p_rand)
        inits.append(p_rand)
        costs.append(cost_rand)

    # --- Main Search Loop ---
    
    best_cost = np.inf
    best_p = None
    
    # Run local search from each initialization
    for idx, p_init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        p_work = p_init.copy()
        cost = costs[idx]
        
        p_work, cost = efficient_local_search(p_work, cost, deadline - 5.0)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = costs[0]

    # --- Iterated Local Search (ILS) with Adaptive Perturbation ---
    
    no_improve_count = 0
    base_perturb = 3
    max_perturb = 15
    
    while time.time() < deadline - 10.0:
        p = best_p.copy()
        
        # Adaptive perturbation size
        perturb_size = base_perturb + min(no_improve_count, 10)
        perturb_size = min(perturb_size, max_perturb)
        
        # Occasionally do a large jump (20% chance)
        if rng.rand() < 0.2:
            perturb_size = max_perturb + rng.randint(0, 10)
        
        num_swaps = max(2, perturb_size)
        for _ in range(num_swaps):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        p, current_cost = efficient_local_search(p, current_cost, deadline - 5.0)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Reset counter if it gets too high to allow exploration
        if no_improve_count > 20:
            no_improve_count = 10

    # --- Final SA Polish if time remains ---
    if time.time() < deadline - 3.0:
        curr_p = best_p.copy()
        curr_cost = best_cost
        best_p_sa = best_p.copy()
        best_cost_sa = best_cost
        
        rng_sa = np.random.RandomState(111)
        T = 1000.0
        cooling = 0.999
        
        end_time_sa = deadline - 1.0
        while time.time() < end_time_sa and T > 1.0:
            i, j = rng_sa.choice(n, 2, replace=False)
            li = curr_p[i]
            lj = curr_p[j]
            
            drow = b_f[lj, curr_p] - b_f[li, curr_p]
            dcol = b_f[curr_p, lj] - b_f[curr_p, li]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
            
            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = lj, li
                curr_cost += delta
                
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p_sa = curr_p.copy()
            
            T *= cooling
        
        if best_cost_sa < best_cost:
            best_p = best_p_sa
            best_cost = best_cost_sa

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
