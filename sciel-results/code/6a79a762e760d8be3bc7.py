"""QAP tai256c solver: iterated local search with vectorized deltas, Hungarian init/reopt, adaptive ILS."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def vectorized_deltas_for_i(p, i):
        """Compute swap deltas for all j > i at once."""
        pi = p[i]
        # For each j, delta = change in cost when swapping positions i and j
        # We compute for all j in vectorized form
        # d_uv[j] = b[p[j], :] - b[pi, :]
        # d_vu[j] = b[:, p[j]] - b[:, pi]
        # This is complex to fully vectorize, so we do a semi-vectorized approach:
        # Precompute differences involving position i
        a_diff_row = a[i, :]  # a[i, k] for all k
        a_diff_col = a[:, i]  # a[k, i] for all k
        
        # For each j:
        # s1_j = dot(a[i,:]-a[j,:], b[p[j],:]-b[pi,:])[p] with corrections
        # s2_j = dot(a[:,i]-a[:,j], b[:,p[j]]-b[:,pi])[p] with corrections
        # cross_j = (a[i,j]-a[j,i])*(b[p[j],pi]-b[pi,p[j]])
        
        # Vectorize over j:
        pj = p  # all positions
        # b[pj, :] shape (n, n), b[pi, :] shape (n,)
        # d_uv for all j: b[pj, :] - b[pi, :] -> shape (n, n)
        d_uv_all = b[pj, :] - b[pi, :]  # (n, n)
        d_vu_all = b[:, pj] - b[:, pi]  # (n, n)
        
        # s1_j = dot(a[i,:]-a[j,:], d_uv_all[j])[p]
        # = sum_k (a[i,k]-a[j,k]) * d_uv_all[j,k,p[k]]
        # d_uv_all[j,k,p[k]]: for each j, take row j, then index columns by p
        # This is: for each j, sum_k (a[i,k]-a[j,k]) * (b[p[j],k]-b[pi,k]) * (p[k] == k? no)
        # Actually d_uv_all[j, k] = b[p[j], k] - b[pi, k]
        # We need d_uv_all[j, p[j]]? No, we need sum over k of (a[i,k]-a[j,k])*(b[p[j],k]-b[pi,k])
        # But with the [p] indexing in the original: d_uv[p] means d_uv[p[k]] for each k
        # So s1_j = sum_k (a[i,k]-a[j,k]) * (b[p[j], p[k]] - b[pi, p[k]])
        
        # Let's compute: for each j, s1_j = sum_k (a[i,k]-a[j,k]) * (b[p[j], p[k]] - b[pi, p[k]])
        # b[p[j], p[k]] for all j,k: this is b[np.ix_(p, p)] which is fixed!
        # Let Bpp = b[np.ix_(p, p)]
        # Then b[p[j], p[k]] = Bpp[j, k]
        # b[pi, p[k]] = b[pi, p[k]] -> row pi of Bpp? No, Bpp[j,k] = b[p[j],p[k]]
        # b[pi, p[k]] is not directly Bpp unless pi = p[j] for some j
        # b[pi, p[k]] = row pi of b, indexed by p[k]
        
        # This is getting complex. Let's use a different approach:
        # Precompute Bpp = b[np.ix_(p,p)]
        # Then for swap (i,j):
        # s1_j = sum_k (a[i,k]-a[j,k]) * (Bpp[j,k] - b[pi, p[k]])
        # b[pi, p[k]] for all k: this is b[pi, p] which is a row vector
        
        Bpp = b[np.ix_(p, p)]
        b_pi_p = b[pi, p]  # shape (n,)
        
        # s1_j = sum_k (a[i,k]-a[j,k]) * (Bpp[j,k] - b_pi_p[k])
        # = sum_k a[i,k]*(Bpp[j,k]-b_pi_p[k]) - sum_k a[j,k]*(Bpp[j,k]-b_pi_p[k])
        # = (a[i,:] * (Bpp[j,:]-b_pi_p)).sum() - (a[j,:] * (Bpp[j,:]-b_pi_p)).sum()
        # For all j: 
        # term1_j = (a[i,:] * (Bpp[j,:]-b_pi_p)).sum() -> a[i,:] @ (Bpp - b_pi_p)[:, j]? 
        # Actually: sum_k a[i,k] * (Bpp[j,k] - b_pi_p[k]) = a[i,:] dot (Bpp[j,:] - b_pi_p)
        # For all j: this is (Bpp - b_pi_p[None,:]) @ a[i,:]  -> shape (n,)
        term1 = (Bpp - b_pi_p[None, :]) @ a[i, :]  # (n,)
        
        # term2_j = sum_k a[j,k] * (Bpp[j,k] - b_pi_p[k]) = row j of A times (Bpp[j,:]-b_pi_p)
        # For all j: this is element-wise: A * (Bpp - b_pi_p) then sum along axis 1
        term2 = (a * (Bpp - b_pi_p[None, :])).sum(axis=1)  # (n,)
        
        s1_all = term1 - term2  # (n,)
        
        # Now s2_j = sum_k (a[k,i]-a[k,j]) * (b[p[k], p[j]] - b[p[k], pi])
        # b[p[k], p[j]] = Bpp[k, j]
        # b[p[k], pi] = Bpp[k, ?] where ? is such that p[?]=pi... not straightforward
        # b[p[k], pi] for all k: this is column pi of b, indexed by p[k]
        # = b[p, pi] -> shape (n,)
        b_p_pi = b[p, pi]  # shape (n,)
        
        # s2_j = sum_k (a[k,i]-a[k,j]) * (Bpp[k,j] - b_p_pi[k])
        # = sum_k a[k,i]*(Bpp[k,j]-b_p_pi[k]) - sum_k a[k,j]*(Bpp[k,j]-b_p_pi[k])
        # = (a[:,i] * (Bpp[:,j]-b_p_pi)).sum() - (a[:,j] * (Bpp[:,j]-b_p_pi)).sum()
        
        # For all j:
        # term3_j = sum_k a[k,i] * (Bpp[k,j] - b_p_pi[k]) = a[:,i] dot (Bpp[:,j]-b_p_pi)
        # = (Bpp.T - b_p_pi[:,None]) @ a[:,i]  -> shape (n,)
        term3 = (Bpp.T - b_p_pi[:, None]) @ a[:, i]  # (n,)
        
        # term4_j = sum_k a[k,j] * (Bpp[k,j] - b_p_pi[k])
        # For all j: element-wise A.T * (Bpp.T - b_p_pi) then sum along axis 0? 
        # a[k,j] * (Bpp[k,j] - b_p_pi[k]) summed over k for each j
        term4 = (a.T * (Bpp.T - b_p_pi[:, None])).sum(axis=0)  # (n,)
        
        s2_all = term3 - term4  # (n,)
        
        # cross_j = (a[i,j]-a[j,i]) * (b[p[j], pi] - b[pi, p[j]])
        # b[p[j], pi] = b_p_pi[j]? No, b_p_pi[k] = b[p[k], pi], so b[p[j], pi] = b_p_pi[j]
        # b[pi, p[j]] = b_pi_p[j]
        cross_all = (a[i, :] - a[:, i]) * (b_p_pi - b_pi_p)  # (n,)
        
        deltas = s1_all + s2_all + cross_all  # (n,)
        
        # Corrections for self-interactions (when j == i or when k involves i or j)
        # The original formula has corrections. Let me add them:
        # s1 correction: -(a[i,i]-a[j,i])*d_uv[pi] - (a[i,j]-a[j,j])*d_uv[p[j]]
        # d_uv[pi] = b[p[j], pi] - b[pi, pi] = b_p_pi[j] - b[pi,pi]
        # d_uv[p[j]] = b[p[j], p[j]] - b[pi, p[j]] = Bpp[j,j] - b_pi_p[j]
        corr1 = -(a[i, :] - a[:, i]) * (b_p_pi - b[pi, pi])  # this is for the first correction term
        # Wait, the correction is: -(a[i,i]-a[j,i])*d_uv[pi] where d_uv[pi] = b[p[j],pi]-b[pi,pi]
        # For all j: -(a[i,i]-a[j,i]) * (b_p_pi[j] - b[pi,pi])
        corr1 = -(a[i, i] - a[:, i]) * (b_p_pi - b[pi, pi])  # (n,)
        
        # Second correction: -(a[i,j]-a[j,j])*d_uv[p[j]]
        # d_uv[p[j]] = b[p[j],p[j]] - b[pi,p[j]] = Bpp[j,j] - b_pi_p[j]
        corr2 = -(a[i, :] - np.diag(a)) * (np.diag(Bpp) - b_pi_p)  # (n,)
        
        # s2 corrections:
        # -(a[i,i]-a[i,j])*d_vu[pi] where d_vu[pi] = b[pi,pi]-b[pi,p[j]]... 
        # d_vu[k] = b[k, p[j]] - b[k, pi]
        # d_vu[pi] = b[pi, p[j]] - b[pi, pi] = b_pi_p[j] - b[pi,pi]
        corr3 = -(a[i, i] - a[i, :]) * (b_pi_p - b[pi, pi])  # (n,)
        
        # -(a[j,i]-a[j,j])*d_vu[p[j]]
        # d_vu[p[j]] = b[p[j], p[j]] - b[p[j], pi] = Bpp[j,j] - b_p_pi[j]
        corr4 = -(a[:, i] - np.diag(a)) * (np.diag(Bpp) - b_p_pi)  # (n,)
        
        deltas = s1_all + corr1 + corr2 + s2_all + corr3 + corr4 + cross_all
        
        # Mask out j == i (no swap with self)
        deltas[i] = 0
        return deltas

    def local_search(p, cost, max_passes=20):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                # Compute all deltas for swaps involving position i
                deltas = vectorized_deltas_for_i(p, i)
                # Find best improving swap with j > i
                j_candidates = np.where(deltas[i+1:] < 0)[0]
                if len(j_candidates) > 0:
                    best_j = j_candidates[np.argmin(deltas[i+1:][j_candidates])] + i + 1
                    delta = deltas[best_j]
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += delta
                    improved = True
                    break
        return p, cost

    def polish_2opt(p, cost, time_budget=5.0):
        """High-precision polish: best-improvement 2-opt with vectorized deltas."""
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            for i in range(n):
                if (time.time() - start_time) > time_budget:
                    break
                deltas = vectorized_deltas_for_i(p, i)
                # Only consider j > i to avoid double-counting
                valid = np.arange(i + 1, n)
                if len(valid) == 0:
                    continue
                sub_deltas = deltas[valid]
                min_idx = np.argmin(sub_deltas)
                if sub_deltas[min_idx] < best_delta:
                    best_delta = sub_deltas[min_idx]
                    best_i = i
                    best_j = valid[min_idx]
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def hungarian_init():
        """Create initial solution using Hungarian algorithm on linearized costs."""
        # Use row sums of a and column sums of b as a linear surrogate
        # C[i,j] = a_row_sums[i] * b_col_sums[j] is too simple
        # Better: C[i,j] = sum_k a[i,k] * b[j,k] (approximate interaction)
        C = a @ b.T
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # Initial solution: Hungarian-based
    p = hungarian_init()
    best_p = p.copy()
    best_cost = compute_cost(best_p)

    # First local search
    current_p, current_cost = local_search(p.copy(), best_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with adaptive perturbation and periodic Hungarian
    iteration = 0
    no_improve_count = 0
    while time.time() < deadline - 25:  # reserve 25s for final polish
        iteration += 1

        # Adaptive perturbation: more swaps when stuck
        base_swaps = 3
        if no_improve_count > 10:
            base_swaps = 6
        if no_improve_count > 30:
            base_swaps = 10
        n_swaps = np.random.randint(base_swaps, base_swaps + 4)
        
        p = best_p.copy()
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 4 == 0 and time.time() < deadline - 35:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            no_improve_count = 0
        else:
            no_improve_count += 1

    # Final high-precision polish pass
    if time.time() < deadline - 5:
        p = best_p.copy()
        p, cost = polish_2opt(p, best_cost, time_budget=min(15.0, deadline - time.time() - 5))
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

        # Try a few more local searches from slightly perturbed best
        remaining = deadline - time.time() - 5
        if remaining > 5:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                p = best_p.copy()
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
                p, cost = local_search(p, compute_cost(p), max_passes=30)
                if cost < best_cost:
                    best_p = p.copy()
                    best_cost = cost
                if time.time() > deadline - 5:
                    break

    return best_p.tolist()
