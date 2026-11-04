<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p, b):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        # Efficient delta calculation using precomputed difference vectors
        diff_b_row = b[lj, :] - b[li, :]
        diff_b_col = b[:, lj] - b[:, li]
        
        # Vectorized delta calculation
        term1 = a[i, :] @ diff_b_row[p]
        term2 = a[:, i] @ diff_b_col[p]
        term3 = -a[j, :] @ diff_b_row[p]
        term4 = -a[:, j] @ diff_b_col[p]
        
        delta = term1 + term2 + term3 + term4
        
        # Exclude i and j from the sum (they were included in the dot products)
        # We need to subtract the contributions involving i and j
        # a[i,i]*diff_b_row[p[i]] + a[i,j]*diff_b_row[p[j]]
        # a[j,i]*diff_b_row[p[i]] + a[j,j]*diff_b_row[p[j]]
        # a[i,i]*diff_b_col[p[i]] + a[i,j]*diff_b_col[p[j]]
        # a[j,i]*diff_b_col[p[i]] + a[j,j]*diff_b_col[p[j]]
        
        # Actually, the formula above is correct for the sum over all k != i, j
        # but the dot products include k=i and k=j. Let me re-derive.
        
        # Delta = sum_{k != i,j} a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[j,k]*(b[li,p[k]] - b[lj,p[k]])
        #       + sum_{k != i,j} a[k,i]*(b[p[k],lj] - b[p[k],li]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
        #       + a[i,i]*(b[lj,lj] - b[li,li]) + a[j,j]*(b[li,li] - b[lj,lj])
        #       + a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
        
        # The first sum can be written as a[i,:]*diff_b_row[p] - a[i,i]*diff_b_row[p[i]] - a[i,j]*diff_b_row[p[j]]
        # The second sum can be written as a[j,:]*(-diff_b_row[p]) + a[j,i]*diff_b_row[p[i]] + a[j,j]*diff_b_row[p[j]]
        # The third sum can be written as a[:,i]*diff_b_col[p] - a[i,i]*diff_b_col[p[i]] - a[j,i]*diff_b_col[p[j]]
        # The fourth sum can be written as a[:,j]*(-diff_b_col[p]) + a[i,j]*diff_b_col[p[i]] + a[j,j]*diff_b_col[p[j]]
        
        # So delta = (a[i,:] @ diff_b_row[p] - a[i,i]*diff_b_row[p[i]] - a[i,j]*diff_b_row[p[j]])
        #         + (-a[j,:] @ diff_b_row[p] + a[j,i]*diff_b_row[p[i]] + a[j,j]*diff_b_row[p[j]])
        #         + (a[:,i] @ diff_b_col[p] - a[i,i]*diff_b_col[p[i]] - a[j,i]*diff_b_col[p[j]])
        #         + (-a[:,j] @ diff_b_col[p] + a[i,j]*diff_b_col[p[i]] + a[j,j]*diff_b_col[p[j]])
        #         + a[i,i]*(b[lj,lj] - b[li,li]) + a[j,j]*(b[li,li] - b[lj,lj])
        #         + a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
        
        # This is getting complicated. Let me use the simpler approach from the original code
        # which was correct, just slower.
        
        # Actually, let me just use the original correct formula
        delta_k = (a[i, :] - a[j, :]) * diff_b_row[p] + (a[:, i] - a[:, j]) * diff_b_col[p]
        delta_k[i] = 0
        delta_k[j] = 0
        delta = np.sum(delta_k)
        
        # Diagonal and i-j terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        return delta

    def local_search(p, best_cost, b):
        """Perform 2-opt local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    delta = compute_swap_delta(i, j, p, b)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        improved = True
        return p, best_cost

    def mds_initialize(a, b):
        """MDS-based initialization."""
        def double_center(M):
            n = len(M)
            J = np.eye(n) - np.ones((n, n)) / n
            return J @ M @ J

        C_A = double_center(a.astype(float))
        C_B = double_center(b.astype(float))
        
        k = 10
        
        eigvals_A, eigvecs_A = np.linalg.eigh(C_A)
        eigvals_B, eigvecs_B = np.linalg.eigh(C_B)
        
        idx_A = np.argsort(eigvals_A)[::-1]
        idx_B = np.argsort(eigvals_B)[::-1]
        
        eigvals_A = eigvals_A[idx_A]
        eigvecs_A = eigvecs_A[:, idx_A]
        
        eigvals_B = eigvals_B[idx_B]
        eigvecs_B = eigvecs_B[:, idx_B]
        
        X_A = eigvecs_A[:, :k] * np.sqrt(np.maximum(eigvals_A[:k], 0))
        X_B = eigvecs_B[:, :k] * np.sqrt(np.maximum(eigvals_B[:k], 0))
        
        M = X_A.T @ X_B
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        
        X_A_rot = X_A @ R
        
        norm_A = np.sum(X_A_rot**2, axis=1)
        norm_B = np.sum(X_B**2, axis=1)
        dists = norm_A[:, None] + norm_B[None, :] - 2 * (X_A_rot @ X_B.T)
        
        row_ind, col_ind = linear_sum_assignment(dists)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    # 1. MDS Initialization
    p_mds = mds_initialize(a, b)
    best_p = p_mds.copy()
    best_cost = compute_cost_fast(p_mds)
    
    # 2. Local Search
    best_p, best_cost = local_search(best_p, best_cost, b)
    
    # 3. Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        p_rand = rng.permutation(n)
        p_rand, cost_rand = local_search(p_rand, compute_cost_fast(p_rand), b)
        if cost_rand < best_cost:
            best_p = p_rand.copy()
            best_cost = cost_rand
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def local_search_2opt(p, best_cost, a, b):
        """Perform 2-opt local search using permuted B for fast deltas."""
        # Maintain B_perm where B_perm[i] = b[p[i]]
        B_perm = b[p]  # shape (n, n), B_perm[i] = b[p[i]]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return p, best_cost
                ai = a[i]
                a_i_col = a[:, i]
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, best_cost
                    # Delta for swapping facilities i and j
                    # diff = B_perm[j] - B_perm[i]  (row difference)
                    # col_diff = B_perm[:, j] - B_perm[:, i] (column difference)
                    diff_row = B_perm[j] - B_perm[i]
                    diff_col = B_perm[:, j] - B_perm[:, i]
                    
                    # delta = sum_k (a[i,k]-a[j,k]) * diff_row[k] + sum_k (a[k,i]-a[k,j]) * diff_col[k]
                    # but exclude k=i and k=j from the sums, then add explicit i-j and diagonal terms
                    # Simpler: compute full sum then subtract i and j contributions
                    delta = (ai - a[j]) @ diff_row + (a_i_col - a[:, j]) @ diff_col
                    # Subtract the i and j terms that were included
                    delta -= ai[i] * diff_row[i] + ai[j] * diff_row[j]
                    delta -= a[j, i] * diff_row[i] + a[j, j] * diff_row[j]
                    delta -= a_i_col[i] * diff_col[i] + a_i_col[j] * diff_col[j]
                    delta -= a[:, j, i] * diff_col[i] + a[:, j, j] * diff_col[j]
                    # Wait, that's wrong. Let me be more careful.
                    # The full dot products include all k. We need to exclude k=i and k=j.
                    # Actually the formula (ai - a[j]) @ diff_row includes k=i and k=j.
                    # We want sum_{k != i,j} (ai[k]-a[j,k])*diff_row[k]
                    # = (ai - a[j]) @ diff_row - ai[i]*diff_row[i] - ai[j]*diff_row[j] + a[j,i]*diff_row[i] + a[j,j]*diff_row[j]
                    # Similarly for columns.
                    # Let me just recompute properly.
                    
                    # Redo: 
                    # delta = sum_{k!=i,j} [(a[i,k]-a[j,k])*diff_row[k] + (a[k,i]-a[k,j])*diff_col[k]]
                    #       + a[i,i]*(b[p[j],p[j]]-b[p[i],p[i]]) + a[j,j]*(b[p[i],p[i]]-b[p[j],p[j]])
                    #       + a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
                    
                    # The first sum:
                    # sum_{k!=i,j} (a[i,k]-a[j,k])*diff_row[k]
                    # = (a[i,:]-a[j,:]) @ diff_row - (a[i,i]-a[j,i])*diff_row[i] - (a[i,j]-a[j,j])*diff_row[j]
                    
                    # sum_{k!=i,j} (a[k,i]-a[k,j])*diff_col[k]
                    # = (a[:,i]-a[:,j]) @ diff_col - (a[i,i]-a[i,j])*diff_col[i] - (a[j,i]-a[j,j])*diff_col[j]
                    
                    d1 = (ai - a[j]) @ diff_row
                    d1 -= (ai[i] - a[j, i]) * diff_row[i]
                    d1 -= (ai[j] - a[j, j]) * diff_row[j]
                    
                    d2 = (a_i_col - a[:, j]) @ diff_col
                    d2 -= (a_i_col[i] - a[i, j]) * diff_col[i]
                    d2 -= (a_i_col[j] - a[j, j]) * diff_col[j]
                    
                    # Diagonal terms
                    diag_i = b[p[i], p[i]]
                    diag_j = b[p[j], p[j]]
                    cross_ij = b[p[i], p[j]]
                    cross_ji = b[p[j], p[i]]
                    
                    delta = d1 + d2
                    delta += ai[i] * (diag_j - diag_i)
                    delta += a[j, j] * (diag_i - diag_j)
                    delta += ai[j] * (cross_ji - cross_ij)
                    delta += a[j, i] * (cross_ij - cross_ji)
                    
                    if delta < 0:
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        B_perm[i], B_perm[j] = B_perm[j], B_perm[i]
                        best_cost += delta
                        improved = True
        return p, best_cost

    def local_search_3opt(p, best_cost, a, b):
        """Perform 3-opt local search: cycle three facilities."""
        # For 3-opt, we try cycling p[i], p[j], p[k] -> p[j], p[k], p[i]
        # This is more expensive but can escape 2-opt local optima
        B_perm = b[p]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return p, best_cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, best_cost
                    for k in range(j + 1, n):
                        if time.time() > deadline:
                            return p, best_cost
                        # Compute cost delta for cycling i->j, j->k, k->i
                        # This means facility i goes to location p[k], j to p[i], k to p[j]
                        # New assignment: p_new[i]=p[k], p_new[j]=p[i], p_new[k]=p[j]
                        # Delta = new_cost - old_cost
                        # We compute this by evaluating the change
                        li, lj, lk = p[i], p[j], p[k]
                        
                        # For each facility, compute its new contribution minus old
                        # This is O(n) per triple, so O(n^4) total - too slow for full 3-opt
                        # Instead, use a smarter delta computation
                        # Actually for n=150, n^4 = 506M which is too much.
                        # Skip full 3-opt, just do targeted 3-opt on best 2-opt neighbors
                        pass
        return p, best_cost

    def mds_initialize(a, b, k=15):
        """MDS-based initialization with configurable dimensions."""
        def double_center(M):
            n = len(M)
            J = np.eye(n) - np.ones((n, n)) / n
            return J @ M @ J

        C_A = double_center(a.astype(float))
        C_B = double_center(b.astype(float))
        
        eigvals_A, eigvecs_A = np.linalg.eigh(C_A)
        eigvals_B, eigvecs_B = np.linalg.eigh(C_B)
        
        idx_A = np.argsort(eigvals_A)[::-1]
        idx_B = np.argsort(eigvals_B)[::-1]
        
        eigvals_A = eigvals_A[idx_A]
        eigvecs_A = eigvecs_A[:, idx_A]
        
        eigvals_B = eigvals_B[idx_B]
        eigvecs_B = eigvecs_B[:, idx_B]
        
        X_A = eigvecs_A[:, :k] * np.sqrt(np.maximum(eigvals_A[:k], 0))
        X_B = eigvecs_B[:, :k] * np.sqrt(np.maximum(eigvals_B[:k], 0))
        
        M = X_A.T @ X_B
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        
        X_A_rot = X_A @ R
        
        norm_A = np.sum(X_A_rot**2, axis=1)
        norm_B = np.sum(X_B**2, axis=1)
        dists = norm_A[:, None] + norm_B[None, :] - 2 * (X_A_rot @ X_B.T)
        
        row_ind, col_ind = linear_sum_assignment(dists)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    # 1. MDS Initialization with higher dimensions
    p_mds = mds_initialize(a, b, k=15)
    best_p = p_mds.copy()
    best_cost = compute_cost_fast(p_mds)
    
    # 2. 2-opt Local Search with B_perm acceleration
    best_p, best_cost = local_search_2opt(best_p, best_cost, a, b)
    
    # 3. Random restarts with 2-opt local search
    rng = np.random.RandomState(42)
    for _ in range(10):
        if time.time() > deadline - 3:
            break
        p_rand = rng.permutation(n)
        p_rand, cost_rand = local_search_2opt(p_rand, compute_cost_fast(p_rand), a, b)
        if cost_rand < best_cost:
            best_p = p_rand.copy()
            best_cost = cost_rand
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error in the delta computation. Let me fix it with a cleaner approach:

