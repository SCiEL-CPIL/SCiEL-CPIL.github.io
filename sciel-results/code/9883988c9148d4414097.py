"""Bootstrap MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Strategy: Use block-Strassen (rank 189) as baseline, then attempt rank reduction
using vectorized verification and tensor-entry-based elimination.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses block-Strassen as baseline and attempts rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build block-Strassen decomposition (rank 189)
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    best_U, best_V, best_W = U_bs.copy(), V_bs.copy(), W_bs.copy()
    best_rank = rank_bs
    
    # Fast vectorized verification using multiple random matrices
    def verify(U, V, W, tol=1e-9, n_tests=3):
        """
        Check decomposition correctness:
        C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        
        Vectorized: For each r, compute:
          u_r = U[:,r] reshaped to (n,m) -> contract with A along j
          v_r = V[:,r] reshaped to (m,p) -> contract with B along j  
          contribution = W[k*n+i, r] * (U_r @ A @ V_r^T elementwise)
        """
        for seed in range(n_tests):
            rng = np.random.default_rng(seed)
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(U.shape[1]):
                wr = W[:, r]
                if np.all(np.abs(wr) < 1e-15):
                    continue
                # u_r: (n*m,) -> (n, m)
                ur = U[:, r].reshape(n, m)
                # v_r: (m*p,) -> (m, p)
                vr = V[:, r].reshape(m, p)
                # w_r: (p*n,) -> (p, n)
                wr_mat = wr.reshape(p, n)
                
                # Compute: sum_j ur[i,j] * A[i,j] for each i -> n-vector
                a_contra = np.sum(ur * A, axis=1)  # shape (n,)
                # Compute: sum_j vr[j,k] * B[j,k] for each k -> p-vector
                b_contra = np.sum(vr * B, axis=0)  # shape (p,)
                
                # C[i,k] += wr_mat[k,i] * a_contra[i] * b_contra[k]
                C_computed += wr_mat * np.outer(b_contra, a_contra)
            
            if not np.allclose(C_computed, C_expected, atol=tol):
                return False
        return True
    
    # Verify block-Strassen
    if not verify(U_bs, V_bs, W_bs):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_schoolbook
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt rank reduction via tensor-entry-based elimination
    # A column r is redundant if the rank-1 tensor u_r ⊗ v_r ⊗ w_r is in the span
    # of the other rank-1 tensors. We check this by verifying that removing column r
    # still yields the correct decomposition (using a fast check).
    
    remaining = list(range(best_rank))
    elim_count = 0
    max_elim = 50
    
    # Pre-compute the target tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Actually, the tensor is: for all i,j,k, the entry is 1 (it's the identity tensor for matmul)
    # The decomposition must satisfy: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Fast check: compute the "residual" tensor when removing column r
    # T_residual[i,j,k] = T[i,j,k] - sum_{s != r} U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    # If T_residual == U[:,r]*V[:,r]*W[:,r] (as a tensor), then r is needed.
    # If T_residual == 0, then r is redundant.
    
    # Even faster: check if the rank-1 tensor u_r⊗v_r⊗w_r can be expressed
    # as a linear combination of others. Use a necessary and sufficient condition:
    # The tensor u_r⊗v_r⊗w_r - sum_s c_s * u_s⊗v_s⊗w_s = 0
    # This is equivalent to: for all (i,j,k), the equation holds.
    
    # Practical approach: try eliminating columns one at a time, checking via
    # the full verification (but with fewer random tests for speed).
    
    # Sort by norm to try eliminating small-norm columns first
    col_norms = (np.linalg.norm(best_U, axis=0) * 
                 np.linalg.norm(best_V, axis=0) * 
                 np.linalg.norm(best_W, axis=0))
    remaining.sort(key=lambda r: col_norms[r])
    
    for round_num in range(max_elim):
        if time.time() > deadline - 20:
            break
        
        eliminated_this_round = False
        
        for idx_r in range(len(remaining)):
            if time.time() > deadline - 20:
                break
            
            r = remaining[idx_r]
            
            # Try removing column r and verify with a quick check
            mask = np.ones(best_rank, dtype=bool)
            mask[r] = False
            
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            # Quick verify with 1 random test
            if verify(U_test, V_test, W_test, tol=1e-8, n_tests=1):
                # Column r is redundant
                remaining.remove(r)
                elim_count += 1
                eliminated_this_round = True
                break
        
        if not eliminated_this_round:
            break
    
    if elim_count > 0:
        final_U = best_U[:, remaining]
        final_V = best_V[:, remaining]
        final_W = best_W[:, remaining]
        
        # Full verification of the reduced decomposition
        if verify(final_U, final_V, final_W, tol=1e-9, n_tests=5):
            best_U, best_V, best_W = final_U, final_V, final_W
            best_rank = len(remaining)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
