"""Program for matrix multiplication tensor rank minimization (3x4, 4x5).

Approach:
1. Construct the explicit matrix multiplication tensor T of shape (n*m, m*p, p*n).
2. Use Alternating Least Squares (ALS) to attempt to find a rank-R decomposition.
3. Iterate R from the schoolbook rank (n*m*p) downwards.
4. For each R, perform multiple random restarts of ALS.
5. If an exact decomposition (residual ~ 0) is found, return it.
6. If no lower rank is found, return the schoolbook decomposition.

This approach uses optimization (ALS) rather than combinatorial merging.
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # 1. Construct the tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Helper to check if decomposition is exact
    def check_exact(U, V, W, tol=1e-5):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = np.abs(T_recon - T)
        return np.max(diff) < tol

    # ALS Step
    def als_step(U, V, W, T):
        R = U.shape[1]
        
        # Update U
        # T_mat[i, j*pn + k] = T[i, j, k]
        # T_mat[i, :] = sum_r U[i, r] * (V[j, r] * W[k, r])
        # Let M_VW[r, j*pn + k] = V[j, r] * W[k, r]
        # Then T_mat = U @ M_VW
        # We solve U @ M_VW = T_mat for U.
        # This is equivalent to M_VW.T @ U.T = T_mat.T
        M_VW = np.einsum('jr,kr->rjk', V, W).reshape(R, mp * pn)
        T_mat = T.reshape(nm, mp * pn)
        try:
            U_T, _, _, _ = np.linalg.lstsq(M_VW.T, T_mat.T, rcond=None)
            U_new = U_T.T
        except:
            U_new = U

        # Update V
        # T_mat2[j, i*pn + k] = T[i, j, k]
        # T_mat2[j, :] = sum_r V[j, r] * (U[i, r] * W[k, r])
        # Let M_UW[r, i*pn + k] = U[i, r] * W[k, r]
        # Then T_mat2 = V @ M_UW
        M_UW = np.einsum('ir,kr->rik', U_new, W).reshape(R, nm * pn)
        T_mat2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        try:
            V_T, _, _, _ = np.linalg.lstsq(M_UW.T, T_mat2.T, rcond=None)
            V_new = V_T.T
        except:
            V_new = V

        # Update W
        # T_mat3[k, i*mp + j] = T[i, j, k]
        # T_mat3[k, :] = sum_r W[k, r] * (U[i, r] * V[j, r])
        # Let M_UV[r, i*mp + j] = U[i, r] * V[j, r]
        # Then T_mat3 = W @ M_UV
        M_UV = np.einsum('ir,jr->rij', U_new, V_new).reshape(R, nm * mp)
        T_mat3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        try:
            W_T, _, _, _ = np.linalg.lstsq(M_UV.T, T_mat3.T, rcond=None)
            W_new = W_T.T
        except:
            W_new = W

        # Calculate residual
        T_recon = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        residual = np.linalg.norm(T_recon - T)
        
        return U_new, V_new, W_new, residual

    # Initialize with schoolbook
    U_init = np.zeros((nm, schoolbook_rank))
    V_init = np.zeros((mp, schoolbook_rank))
    W_init = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i * m + j, col] = 1.0
                V_init[j * p + k, col] = 1.0
                W_init[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_init, V_init, W_init
    best_rank = schoolbook_rank

    # Try to reduce rank
    np.random.seed(42)
    max_restarts = 5
    max_als_iters = 10
    
    for R in range(best_rank - 1, 19, -1):
        found = False
        for restart in range(max_restarts):
            if restart == 0:
                U = best_U[:, :R].copy()
                V = best_V[:, :R].copy()
                W = best_W[:, :R].copy()
            else:
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
            
            res = float('inf')
            for _ in range(max_als_iters):
                U, V, W, res = als_step(U, V, W, T)
                if res < 1e-6:
                    break
                    
            if res < 1e-6 and check_exact(U, V, W, tol=1e-4):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                found = True
                break
        
        if found:
            continue
        else:
            break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
