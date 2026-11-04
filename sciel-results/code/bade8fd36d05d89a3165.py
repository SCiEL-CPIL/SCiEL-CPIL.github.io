"""QAP solver for tho150 using MDS/Procrustes init with BCD + multi-restart 2-opt polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        nn = D.shape[0]
        J = np.eye(nn) - np.ones((nn, nn)) / nn
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

    def local_search(p, current_cost, end_time, seed=12345):
        """Vectorized 2-swap local search using B_perm matrix."""
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        # Build B_perm: B_perm[i, j] = b[curr_p[i], curr_p[j]]
        B_perm = b_f[np.ix_(curr_p, curr_p)]
        
        rng = np.random.RandomState(seed)
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        
        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)
            
            for idx in range(len(pairs)):
                if time.time() > end_time:
                    break
                
                i, j = pairs[idx]
                li = curr_p[i]
                lj = curr_p[j]
                
                # Vectorized delta: a[i,j] * (b[lj,li] - b[li,li] - b[lj,lj] + b[li,lj])
                # + sum_k a[i,k]*(B_perm[k,lj]-B_perm[k,li]) + sum_k a[j,k]*(B_perm[k,li]-B_perm[k,lj])
                # Simplified: delta = a[i,j]*(b[lj,li]+b[li,lj]-b[li,li]-b[lj,lj])
                #            + (a[i,:]-a[j,:]) @ (B_perm[:,lj]-B_perm[:,li])
                #            + (a[:,i]-a[:,j]) @ (B_perm[lj,:]-B_perm[li,:])
                
                dcol = B_perm[:, lj] - B_perm[:, li]  # n-vector
                drow = B_perm[lj, :] - B_perm[li, :]  # n-vector
                
                delta = (a_f[i, j] * (b_f[lj, li] + b_f[li, lj] - b_f[li, li] - b_f[lj, lj])
                         + np.dot(a_f[i, :] - a_f[j, :], dcol)
                         + np.dot(a_f[:, i] - a_f[:, j], drow))
                
                if delta < -1e-10:
                    # Swap i and j in curr_p
                    curr_p[i], curr_p[j] = lj, li
                    curr_cost += delta
                    # Update B_perm: swap rows and cols i, j
                    B_perm[[i, j], :] = B_perm[[j, i], :]
                    B_perm[:, [i, j]] = B_perm[:, [j, i]]
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True
            
            if not improved:
                break
                
        return best_p, best_cost

    def sa_polish(p, current_cost, end_time, seed=111):
        """SA polish with B_perm acceleration and geometric cooling."""
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        # Build B_perm
        B_perm = b_f[np.ix_(curr_p, curr_p)]
        
        rng_sa = np.random.RandomState(seed)
        
        # Estimate typical delta magnitude with a few samples
        deltas = []
        for _ in range(10):
            i0, j0 = rng_sa.choice(n, 2, replace=False)
            li0, lj0 = curr_p[i0], curr_p[j0]
            dcol0 = B_perm[:, lj0] - B_perm[:, li0]
            drow0 = B_perm[lj0, :] - B_perm[li0, :]
            d0 = (a_f[i0, j0] * (b_f[lj0, li0] + b_f[li0, lj0] - b_f[li0, li0] - b_f[lj0, lj0])
                  + np.dot(a_f[i0, :] - a_f[j0, :], dcol0)
                  + np.dot(a_f[:, i0] - a_f[:, j0], drow0))
            deltas.append(abs(d0))
        
        T_start = max(np.median(deltas) * 3.0, 200.0)
        T_end = 1.0
        
        # Estimate total iterations to set cooling rate
        # Assume ~20000 iterations in available time
        elapsed = time.time() - start_time
        remaining = max(end_time - time.time(), 1.0)
        est_iters = int(remaining * 15000)  # ~15k iters/sec estimate
        est_iters = max(est_iters, 100)
        
        if est_iters > 1:
            cooling = (T_end / T_start) ** (1.0 / est_iters)
        else:
            cooling = 0.999
        
        T = T_start
        
        while time.time() < end_time:
            i, j = rng_sa.choice(n, 2, replace=False)
            li, lj = curr_p[i], curr_p[j]
            
            dcol = B_perm[:, lj] - B_perm[:, li]
            drow = B_perm[lj, :] - B_perm[li, :]
            delta = (a_f[i, j] * (b_f[lj, li] + b_f[li, lj] - b_f[li, li] - b_f[lj, lj])
                     + np.dot(a_f[i, :] - a_f[j, :], dcol)
                     + np.dot(a_f[:, i] - a_f[:, j], drow))
            
            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = lj, li
                curr_cost += delta
                B_perm[[i, j], :] = B_perm[[j, i], :]
                B_perm[:, [i, j]] = B_perm[:, [j, i]]
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling
            if T < T_end:
                T = T_end
        
        return best_p, best_cost

    # 1. MDS Initialization - try multiple dims
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0
    
    best_p_init = None
    best_cost_init = np.inf
    
    for dim in [2, 3, 5, 8]:
        if time.time() > deadline - 60.0:
            break
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try
    
    # 2. Rank matching
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

    # 3. Initial local search (longer budget since B_perm is faster)
    end_time_lk1 = deadline - 35.0
    best_p, best_cost = local_search(best_p_init, best_cost_init, end_time_lk1, seed=12345)
    
    # 4. Multiple perturbation + local search restarts (more restarts with faster search)
    rng_perturb = np.random.RandomState(789)
    num_restarts = 7
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        p_pert = best_p.copy()
        # Random number of swaps: 2 to 8
        n_swaps = rng_perturb.randint(2, 9)
        for _ in range(n_swaps):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        cost_pert = compute_cost(p_pert)
        end_time_lk = deadline - 15.0 - r * 2.0
        if end_time_lk < time.time() + 3:
            break
        p_lk, cost_lk = local_search(p_pert, cost_pert, end_time_lk, seed=456 + r)
        
        if cost_lk < best_cost:
            best_p = p_lk
            best_cost = cost_lk
    
    # 5. SA polish if time remains
    if time.time() < deadline - 5.0:
        end_time_sa = deadline - 3.0
        p_sa, cost_sa = sa_polish(best_p, best_cost, end_time_sa, seed=111)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa
    
    # 6. Final short local search
    if time.time() < deadline - 1.0:
        end_time_final = deadline - 0.5
        p_final, cost_final = local_search(best_p, best_cost, end_time_final, seed=999)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
