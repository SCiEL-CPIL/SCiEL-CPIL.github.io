"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target Tensor: T[i, j, k] = 1 if (i, j, k) is a valid term in the multiplication?
    # The tensor for matrix multiplication A (n x m) * B (m x p) -> C (p x n)
    # C[k, i] = sum_j A[i, j] B[j, k]
    # Tensor T is in A^* x B^* x C^*
    # T[i*m + j, k*p + l, ...] ? No.
    # Indices:
    # A index: i*m + j (size n*m)
    # B index: j*p + k (size m*p)
    # C index: k*n + i (size p*n)
    # The tensor has 1s at (i*m+j, j*p+k, k*n+i) for all valid i,j,k.
    
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build target tensor
    T = np.zeros((dim_A, dim_B, dim_C), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Schoolbook Rank
    rank_school = n * m * p
    U_school = np.zeros((dim_A, rank_school))
    V_school = np.zeros((dim_B, rank_school))
    W_school = np.zeros((dim_C, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    # Try to find a lower rank decomposition using ALS
    # Target rank: 55 (based on 23+23+9)
    target_rank = 55
    best_U, best_V, best_W = None, None, None
    
    # ALS Implementation
    def als_decompose(T, rank, max_iters=50, tol=1e-9):
        # Initialize
        U = np.random.randn(T.shape[0], rank)
        V = np.random.randn(T.shape[1], rank)
        W = np.random.randn(T.shape[2], rank)
        
        for it in range(max_iters):
            # Update U
            # T_unfolded_1: (dim_A, dim_B * dim_C)
            # T[i, :, :] reshaped
            T_unfold_1 = T.reshape(T.shape[0], -1)
            # U = T_unfold_1 @ (V * W)^T ?
            # T = sum_r U[:, r] V[:, r] W[:, r]^T
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # T_unfold_1[i, (j, k)] = sum_r U[i, r] (V[j, r] W[k, r])
            # Let X = V * W (element-wise product, shape dim_B x rank)
            # Actually, we need the Khatri-Rao product of V and W.
            # V is dim_B x rank, W is dim_C x rank.
            # Khatri-Rao product V * W is (dim_B * dim_C) x rank.
            # Row (j, k) of V * W is V[j, :] * W[k, :] (element-wise).
            # We can compute this efficiently.
            # V_kr = np.einsum('jr,kr->jkr', V, W).reshape(-1, rank)
            # This is large. dim_B=21, dim_C=21, rank=55. 441x55.
            
            # Optimization: Use einsum or loop
            # V_kr = np.zeros((T.shape[1] * T.shape[2], rank))
            # for j in range(T.shape[1]):
            #     for k in range(T.shape[2]):
            #         V_kr[j*T.shape[2]+k, :] = V[j, :] * W[k, :]
            
            # Better:
            # V_kr = np.einsum('jr,kr->jkr', V, W).reshape(-1, rank)
            V_kr = np.einsum('jr,kr->jkr', V, W).reshape(-1, rank)
            # U = T_unfold_1 @ V_kr^+ (pseudo-inverse)
            # U = np.linalg.lstsq(V_kr, T_unfold_1, rcond=None)[0].T
            # Actually, T_unfold_1 = U @ V_kr^T
            # U = T_unfold_1 @ V_kr^T @ inv(V_kr @ V_kr^T)
            # Use lstsq: V_kr @ U^T = T_unfold_1^T
            U_new = np.linalg.lstsq(V_kr, T_unfold_1.T, rcond=None)[0].T
            
            # Update V
            T_unfold_2 = T.reshape(T.shape[1], -1)
            U_kr = np.einsum('ir,kr->ikr', U, W).reshape(-1, rank)
            V_new = np.linalg.lstsq(U_kr, T_unfold_2.T, rcond=None)[0].T
            
            # Update W
            T_unfold_3 = T.reshape(T.shape[2], -1)
            U_V_kr = np.einsum('ir,jr->ijr', U, V).reshape(-1, rank)
            W_new = np.linalg.lstsq(U_V_kr, T_unfold_3.T, rcond=None)[0].T
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            # Reconstruct
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                break
                
        return U, V, W, err

    # Try ALS a few times
    for attempt in range(10):
        np.random.seed(42 + attempt)
        U, V, W, err = als_decompose(T, target_rank, max_iters=100)
        if err < 1e-6:
            # Try to round to integers or halves
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_rec = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(T_rec, T, atol=1e-9):
                best_U, best_V, best_W = U_r, V_r, W_r
                break

    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
# EVOLVE-BLOCK-END
