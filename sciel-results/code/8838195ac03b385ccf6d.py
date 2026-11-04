"""QAP solver for tho150 combining MDS/Procrustes initialization with vectorized 2-opt local search."""
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

    def fast_local_search(p, current_cost, end_time, seed=12345):
        """
        Fast 2-swap local search.
        Uses vectorized delta computation where possible, but falls back to per-pair
        if needed. For n=150, we can afford O(n^2) per pass with vectorized ops.
        
        The key insight: for a swap (i, j), the delta can be computed in O(n) 
        using precomputed row/col differences.
        
        We will do multiple passes. In each pass, we shuffle the order of pairs
        and accept the first improving move (first-improvement strategy) or 
        the best move (best-improvement). Best-improvement is slower but 
        converges faster in terms of number of swaps. Given n=150, 
        n^2/2 = ~11k pairs. Each delta is O(n) ~ 150 ops. 
        Total per pass ~ 1.7M ops. This is fast enough to do many passes.
        
        Let's use a hybrid: compute all deltas for a fixed i against all j>i 
        in a vectorized way? 
        Actually, the formula delta = (a[i,:]-a[j,:]) @ (b[lj,p]-b[li,p]) + ... 
        makes it hard to fully vectorize over j for a fixed i without expanding.
        
        However, we can precompute:
        B_perm = b_f[p, p]  # The permuted distance matrix
        Then b[lj, p] is just B_perm[lj, :]? No.
        b_f[lj, p] is row lj of b_f, indexed by p.
        B_perm[k, l] = b_f[p[k], p[l]].
        So b_f[lj, p[k]] is NOT directly in B_perm unless lj is in p.
        
        Let's stick to the explicit formula but optimize the inner loop.
        For n=150, a pure Python loop over 11k pairs with numpy dot products 
        inside might be too slow if we do many passes.
        
        Alternative: Vectorize the delta computation for ALL pairs at once?
        delta[i, j] = (a[i,:] - a[j,:]) @ (b[p[j], p] - b[p[i], p]) 
                   + (a[:,i] - a[:,j]) @ (b[p, p[j]] - b[p, p[i]])
        
        Let's define:
        D1[i, j] = (a[i,:] - a[j,:]) @ (b[p[j], p] - b[p[i], p])
        This is hard to vectorize directly as a matrix product because the 
        vectors depend on i and j in a non-linear way.
        
        However, we can use the following identity:
        Let A = a_f
        Let B = b_f
        Let P = p (permutation)
        
        Term 1: sum_k (A[i,k] - A[j,k]) * (B[P[j], P[k]] - B[P[i], P[k]])
        = sum_k A[i,k] B[P[j], P[k]] - sum_k A[i,k] B[P[i], P[k]]
          - sum_k A[j,k] B[P[j], P[k]] + sum_k A[j,k] B[P[i], P[k]]
        
        Let C1[i, l] = sum_k A[i,k] B[l, P[k]]
        Then Term 1 = C1[i, P[j]] - C1[i, P[i]] - C1[j, P[j]] + C1[j, P[i]]
        
        Similarly for Term 2:
        Term 2: sum_k (A[k,i] - A[k,j]) * (B[P[k], P[j]] - B[P[k], P[i]])
        = sum_k A[k,i] B[P[k], P[j]] - sum_k A[k,i] B[P[k], P[i]]
          - sum_k A[k,j] B[P[k], P[j]] + sum_k A[k,j] B[P[k], P[i]]
        
        Let C2[i, l] = sum_k A[k,i] B[P[k], l]
        Then Term 2 = C2[i, P[j]] - C2[i, P[i]] - C2[j, P[j]] + C2[j, P[i]]
        
        So delta[i, j] = (C1[i, P[j]] - C1[i, P[i]] - C1[j, P[j]] + C1[j, P[i]]) 
                       + (C2[i, P[j]] - C2[i, P[i]] - C2[j, P[j]] + C2[j, P[i]])
        
        This allows O(1) lookup per pair after O(n^3) precomputation of C1 and C2.
        
        C1[i, l] = sum_k A[i,k] B[l, P[k]]
        C2[i, l] = sum_k A[k,i] B[P[k], l]
        
        We can compute C1 and C2 using matrix multiplication.
        C1 = A @ (B[:, P]).T ?
        Let M1 = B[:, P] (columns of B permuted by P). M1[l, k] = B[l, P[k]].
        Then C1[i, l] = sum_k A[i,k] M1[l, k] = (A @ M1.T)[i, l].
        So C1 = A @ B[:, P].T
        
        C2[i, l] = sum_k A[k,i] B[P[k], l]
        Let M2 = B[P, :] (rows of B permuted by P). M2[k, l] = B[P[k], l].
        Then C2[i, l] = sum_k A[k,i] M2[k, l] = (A.T @ M2)[i, l].
        So C2 = A.T @ B[P, :]
        
        This is O(n^3) per local search pass. For n=150, 150^3 = 3.3M multiplications.
        This is very fast in NumPy (milliseconds).
        
        Then we can compute the full delta matrix in O(n^2) using vectorized ops.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        while time.time() < end_time:
            # Precompute C1 and C2
            # C1[i, l] = sum_k a_f[i,k] * b_f[l, curr_p[k]]
            C1 = a_f @ b_f[:, curr_p].T
            # C2[i, l] = sum_k a_f[k,i] * b_f[curr_p[k], l]
            C2 = a_f.T @ b_f[curr_p, :]
            
            # Compute delta matrix
            # delta[i, j] = (C1[i, P[j]] - C1[i, P[i]] - C1[j, P[j]] + C1[j, P[i]]) 
            #             + (C2[i, P[j]] - C2[i, P[i]] - C2[j, P[j]] + C2[j, P[i]])
            
            # Let's vectorize this.
            # We want a matrix D where D[i, j] is the delta for swap (i, j).
            # P is curr_p.
            
            # C1[i, P[j]] is C1 indexed by (i, P[j]).
            # This is C1[:, P] -> shape (n, n) where (C1[:, P])[i, j] = C1[i, P[j]].
            C1_ipj = C1[:, curr_p]
            C1_jpi = C1[curr_p, :] # C1[j, P[i]] where j is row index? No.
            # C1[j, P[i]]: for fixed i, this is a vector of length n where entry j is C1[j, P[i]].
            # This is C1[:, P[i]] (column P[i] of C1).
            # So C1_jpi[i, j] = C1[j, P[i]].
            # This is C1.T[P[i], :]? No.
            # C1.T[P[i], j] = C1[j, P[i]]. Yes.
            C1_jpi = C1.T[curr_p, :] # Shape (n, n), entry [i, j] = C1[j, P[i]]
            
            # C1[i, P[i]] is a vector of length n, entry i is C1[i, P[i]].
            C1_ii = C1[np.arange(n), curr_p]
            # C1[j, P[j]] is a vector of length n, entry j is C1[j, P[j]].
            C1_jj = C1[curr_p, np.arange(n)] # Wait. C1[j, P[j]].
            # Entry j of this vector should be C1[j, P[j]].
            # C1[curr_p, np.arange(n)] has entry j = C1[P[j], j]. That's not it.
            # We want a vector v where v[j] = C1[j, P[j]].
            # This is np.einsum('ij,j->i', C1, curr_p)? Or C1[np.arange(n), curr_p].
            # C1[np.arange(n), curr_p] has entry i = C1[i, curr_p[i]].
            # So C1_jj_vec = C1[np.arange(n), curr_p].
            C1_jj_vec = C1[np.arange(n), curr_p]
            
            # Similarly for C2.
            C2_ipj = C2[:, curr_p]
            C2_jpi = C2.T[curr_p, :]
            C2_ii_vec = C2[np.arange(n), curr_p]
            C2_jj_vec = C2[np.arange(n), curr_p] # Same formula, just different matrix.
            
            # Delta[i, j] = (C1_ipj[i, j] - C1_ii_vec[i] - C1_jj_vec[j] + C1_jpi[i, j])
            #             + (C2_ipj[i, j] - C2_ii_vec[i] - C2_jj_vec[j] + C2_jpi[i, j])
            
            # Let's build the matrix.
            Delta = (C1_ipj - C1_ii_vec[:, None] - C1_jj_vec[None, :] + C1_jpi) + \
                    (C2_ipj - C2_ii_vec[:, None] - C2_jj_vec[None, :] + C2_jpi)
            
            # We only care about i < j.
            # Set diagonal to 0 (no swap).
            np.fill_diagonal(Delta, 0)
            
            # Find the best improving move in the upper triangle.
            # We can use np.triu to zero out lower triangle, then find min.
            Delta_upper = np.triu(Delta, k=1)
            
            min_delta = np.min(Delta_upper)
            
            if min_delta < -1e-10:
                # Find the index
                idx = np.unravel_index(np.argmin(Delta_upper), Delta_upper.shape)
                i, j = idx
                
                # Perform swap
                li = curr_p[i]
                lj = curr_p[j]
                curr_p[i] = lj
                curr_p[j] = li
                curr_cost += min_delta
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            else:
                # No improving move found, local optimum reached.
                break
                
        return best_p, best_cost

    # 1. MDS Initialization
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0
    
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
    
    # 3. Local Search Polish
    end_time_lk = deadline - 10.0
    best_p, best_cost = fast_local_search(best_p_init, best_cost_init, end_time_lk, seed=12345)
    
    # 4. Perturb and Local Search if time remains
    if time.time() < deadline - 15.0:
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        # Swap 3-5 random pairs
        for _ in range(4):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost(p_pert)
        end_time_lk2 = deadline - 5.0
        p_lk2, cost_lk2 = fast_local_search(p_pert, cost_pert, end_time_lk2, seed=456)
        
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    # 5. Final SA Polish if time remains
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
            
            # Compute delta for this specific swap
            # Using the C1/C2 formula is overkill for a single swap.
            # Just compute directly.
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
