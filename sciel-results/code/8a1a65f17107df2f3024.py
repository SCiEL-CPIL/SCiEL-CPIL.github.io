"""QAP solver for tho150 using MDS/Procrustes initialization with efficient 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
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
        from scipy.optimize import linear_sum_assignment
        
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
        Uses the delta formula:
        delta = (a[i,:] - a[j,:]) @ (b[old_j, p] - b[old_i, p]) 
              + (a[:,i] - a[:,j]) @ (b[p, old_j] - b[p, old_i])
        where p is the current permutation (facility -> location).
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        # Precompute:
        # b_row_loc[l] = b_f[l, :]  (row l of b)
        # b_col_loc[l] = b_f[:, l]  (col l of b)
        # We can just index b_f directly.
        
        # For fast delta computation, we maintain:
        # RC[k] = b_f[curr_p[k], :]  -> shape (n, n)
        # CC[k] = b_f[:, curr_p[k]]  -> shape (n, n)
        # Actually, let's just use b_f indexing, it's fast enough in numpy.
        
        # Let's precompute a_f rows and cols as views
        # a_f[i, :] and a_f[:, i]
        
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
                
                # delta = sum_k a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + sum_k a[k,i]*(b[p[k],lj] - b[p[k],li])
                #       + sum_k a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + sum_k a[k,j]*(b[p[k],li] - b[p[k],lj])
                
                # Let drow = b_f[lj, curr_p] - b_f[li, curr_p]
                # Let dcol = b_f[curr_p, lj] - b_f[curr_p, li]
                
                drow = b_f[lj, curr_p] - b_f[li, curr_p]
                dcol = b_f[curr_p, lj] - b_f[curr_p, li]
                
                # delta = a_f[i, :] @ drow + a_f[:, i] @ dcol + a_f[j, :] @ (-drow) + a_f[:, j] @ (-dcol)
                #       = (a_f[i, :] - a_f[j, :]) @ drow + (a_f[:, i] - a_f[:, j]) @ dcol
                
                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]
                
                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
                
                if delta < -1e-10:  # Accept if strictly better
                    # Perform swap
                    curr_p[i], curr_p[j] = lj, li
                    curr_cost += delta
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True
            
            if not improved:
                break
                
        return best_p, best_cost

    # 1. MDS Initialization
    # Use symmetrized matrices for MDS
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0
    
    # Try different embedding dimensions
    best_p_init = None
    best_cost_init = np.inf
    
    for dim in [2, 3, 5]:
        if time.time() > deadline - 50.0:
            break
            
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try
    
    # 2. Rank Matching Initialization
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    
    if cost_rank < best_cost_init:
        best_cost_init = cost_rank
        best_p_init = p_rank
    
    # 3. Random Initialization
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    # 4. Local Search Polish
    end_time_lk = deadline - 10.0
    best_p, best_cost = efficient_local_search(best_p_init, best_cost_init, end_time_lk, seed=12345)
    
    # 5. Perturb and Local Search if time remains
    if time.time() < deadline - 15.0:
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        # Swap 3-5 random pairs
        for _ in range(4):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost(p_pert)
        end_time_lk2 = deadline - 5.0
        p_lk2, cost_lk2 = efficient_local_search(p_pert, cost_pert, end_time_lk2, seed=456)
        
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    # 6. Final SA Polish if time remains
    if time.time() < deadline - 3.0:
        # Simple SA to escape local minima
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

    return best_p.tolist()
# EVOLVE-BLOCK-END
