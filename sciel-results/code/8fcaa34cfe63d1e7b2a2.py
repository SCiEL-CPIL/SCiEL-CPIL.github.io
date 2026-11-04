"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

    def cyclic_rotation_dp(a_f, b_f):
        """
        Cyclic rotation DP: derive two cyclic orderings from eigenvectors,
        then optimize relative rotation using DP.
        """
        # Get top eigenvector of symmetrized matrices
        a_sym = (a_f + a_f.T) / 2.0
        b_sym = (b_f + b_f.T) / 2.0
        
        # Center the matrices
        J_a = np.eye(n) - np.ones((n, n)) / n
        C_A = J_a @ a_sym @ J_a
        C_B = J_a @ b_sym @ J_a
        
        # Get top eigenvector
        vals_a, vecs_a = np.linalg.eigh(C_A)
        vals_b, vecs_b = np.linalg.eigh(C_B)
        
        idx_a = np.argsort(vals_a)[::-1]
        idx_b = np.argsort(vals_b)[::-1]
        
        # Top eigenvector gives a cyclic ordering
        v_a = vecs_a[:, idx_a[0]]
        v_b = vecs_b[:, idx_b[0]]
        
        # Sort by eigenvector value to get cyclic order
        order_a = np.argsort(v_a)
        order_b = np.argsort(v_b)
        
        # For each relative rotation k, compute the cost
        # p[i] = order_b[(order_a.index(i) + k) % n]
        # This is O(n^2) to compute all rotations
        
        best_k = 0
        best_cost = np.inf
        
        for k in range(n):
            # Build permutation for rotation k
            p = np.empty(n, dtype=np.int64)
            for i in range(n):
                pos_a = np.where(order_a == i)[0][0]
                pos_b = (pos_a + k) % n
                p[i] = order_b[pos_b]
            
            cost = compute_cost(p)
            if cost < best_cost:
                best_cost = cost
                best_k = k
        
        # Build final permutation for best_k
        p_best = np.empty(n, dtype=np.int64)
        for i in range(n):
            pos_a = np.where(order_a == i)[0][0]
            pos_b = (pos_a + best_k) % n
            p_best[i] = order_b[pos_b]
        
        return p_best, best_cost

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
        
        # Maintain permuted B matrix for faster access
        # B_perm[i, j] = b_f[curr_p[i], curr_p[j]]
        B_perm = b_f[np.ix_(curr_p, curr_p)].copy()
        
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
                
                # delta = (a[i,:] - a[j,:]) @ (b[lj, p] - b[li, p]) 
                #       + (a[:,i] - a[:,j]) @ (b[p, lj] - b[p, li])
                
                # Using B_perm:
                # b[lj, p] corresponds to B_perm[j, :] (since curr_p[j] = lj)
                # b[li, p] corresponds to B_perm[i, :] (since curr_p[i] = li)
                # b[p, lj] corresponds to B_perm[:, j]
                # b[p, li] corresponds to B_perm[:, i]
                
                drow = B_perm[j, :] - B_perm[i, :]
                dcol = B_perm[:, j] - B_perm[:, i]
                
                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]
                
                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
                
                if delta < -1e-10:  # Accept if strictly better
                    # Perform swap
                    curr_p[i], curr_p[j] = lj, li
                    curr_cost += delta
                    
                    # Update B_perm efficiently
                    # Swap rows i and j
                    B_perm[[i, j], :] = B_perm[[j, i], :]
                    # Swap cols i and j
                    B_perm[:, [i, j]] = B_perm[:, [j, i]]
                    
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
    
    # 2. Cyclic Rotation DP Initialization
    if time.time() < deadline - 45.0:
        p_cyc, cost_cyc = cyclic_rotation_dp(a_f, b_f)
        if cost_cyc < best_cost_init:
            best_cost_init = cost_cyc
            best_p_init = p_cyc
    
    # 3. Rank Matching Initialization
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
    
    # 4. Random Initialization
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    # 5. Local Search Polish
    end_time_lk = deadline - 10.0
    best_p, best_cost = efficient_local_search(best_p_init, best_cost_init, end_time_lk, seed=12345)
    
    # 6. Perturb and Local Search if time remains
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

    # 7. Final SA Polish if time remains
    if time.time() < deadline - 3.0:
        # Simple SA to escape local minima
        curr_p = best_p.copy()
        curr_cost = best_cost
        best_p_sa = best_p.copy()
        best_cost_sa = best_cost
        
        # Maintain permuted B matrix for SA
        B_perm = b_f[np.ix_(curr_p, curr_p)].copy()
        
        rng_sa = np.random.RandomState(111)
        T = 1000.0
        cooling = 0.999
        
        end_time_sa = deadline - 1.0
        while time.time() < end_time_sa and T > 1.0:
            i, j = rng_sa.choice(n, 2, replace=False)
            li = curr_p[i]
            lj = curr_p[j]
            
            drow = B_perm[j, :] - B_perm[i, :]
            dcol = B_perm[:, j] - B_perm[:, i]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
            
            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = lj, li
                curr_cost += delta
                
                # Update B_perm
                B_perm[[i, j], :] = B_perm[[j, i], :]
                B_perm[:, [i, j]] = B_perm[:, [j, i]]
                
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p_sa = curr_p.copy()
            
            T *= cooling
        
        if best_cost_sa < best_cost:
            best_p = best_p_sa
            best_cost = best_cost_sa

    return best_p.tolist()
# EVOLVE-BLOCK-END
