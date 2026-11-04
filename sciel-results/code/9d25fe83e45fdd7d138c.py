"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

    def compute_swap_delta(i, j, p):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        # Vectorized delta calculation
        diff_b_row = b[lj, :] - b[li, :]
        diff_b_col = b[:, lj] - b[:, li]
        
        term1 = a[i, :] * diff_b_row[p]
        term2 = a[:, i] * diff_b_col[p]
        term3 = a[j, :] * (-diff_b_row[p])
        term4 = a[:, j] * (-diff_b_col[p])
        
        delta_k = term1 + term2 + term3 + term4
        
        # Exclude i and j
        delta_k[i] = 0
        delta_k[j] = 0
        
        delta = np.sum(delta_k)
        
        # Diagonal and i-j terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        return delta

    def local_search(p, best_cost):
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
                    delta = compute_swap_delta(i, j, p)
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
    best_p, best_cost = local_search(best_p, best_cost)
    
    # 3. Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        p_rand = rng.permutation(n)
        p_rand, cost_rand = local_search(p_rand, compute_cost_fast(p_rand))
        if cost_rand < best_cost:
            best_p = p_rand.copy()
            best_cost = cost_rand
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
