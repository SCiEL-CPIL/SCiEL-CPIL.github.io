"""
Optimized rank decomposition for 2x3 by 3x4 matrix multiplication.
Uses a constructive algorithm based on structural symmetries and
iterative refinement to find a low-rank exact decomposition.
"""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if (i % m == j // p) and (j % p == k // n)
    # i = i1*m + i2, j = j1*p + j2, k = k1*n + k2
    # i2 == j1 and j2 == k2
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        i2 = i % m
        i1 = i // m
        for j in range(m * p):
            j1 = j // p
            j2 = j % p
            if i2 == j1:
                for k in range(p * n):
                    k2 = k % n
                    if j2 == k2:
                        T[i, j, k] = 1.0

    def try_decompose(rank, max_iters=50, restarts=5, seed=42):
        rng = np.random.default_rng(seed)
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        for _ in range(restarts):
            # Initialize U, V, W with small random values
            U = rng.normal(0, 0.1, (dim_u, rank))
            V = rng.normal(0, 0.1, (dim_v, rank))
            W = rng.normal(0, 0.1, (dim_w, rank))
            
            for it in range(max_iters):
                # Solve for U
                # T[i, :, :] = sum_c U[i, c] * (V[:, c] outer W[:, c])
                # Flatten V[:, c] outer W[:, c] to vector of size dim_v * dim_w
                M_u = np.zeros((dim_v * dim_w, rank))
                for c in range(rank):
                    M_u[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                
                # T is (dim_u, dim_v, dim_w). Reshape to (dim_u, dim_v * dim_w)
                T_u = T.reshape(dim_u, -1)
                # Solve U = T_u @ M_u^+ (pseudo-inverse)
                # U[i, :] = T_u[i, :] @ M_u^+
                # This is equivalent to solving M_u U^T = T_u^T
                U_new, _, _, _ = np.linalg.lstsq(M_u, T_u.T, rcond=None)
                U = U_new.T
                
                # Solve for V
                # T[:, j, :] = sum_c V[j, c] * (U[:, c] outer W[:, c])
                # We need to flatten U[:, c] outer W[:, c] ? No, indices are different.
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # For fixed j, T[:, j, :] is a matrix of size dim_u x dim_w.
                # T[i, j, k] = sum_c V[j, c] (U[i, c] W[k, c])
                # Let N_v be matrix of size (dim_u * dim_w, rank)
                # N_v[ (i, k), c ] = U[i, c] * W[k, c]
                N_v = np.zeros((dim_u * dim_w, rank))
                for c in range(rank):
                    N_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                
                T_v = T.transpose(1, 0, 2).reshape(dim_v, -1) # Shape (dim_v, dim_u * dim_w)
                # T_v[j, (i,k)] = sum_c V[j, c] N_v[(i,k), c]
                # V = T_v @ N_v^+
                V_new, _, _, _ = np.linalg.lstsq(N_v, T_v.T, rcond=None)
                V = V_new.T
                
                # Solve for W
                # T[:, :, k] = sum_c W[k, c] * (U[:, c] outer V[:, c])
                N_w = np.zeros((dim_u * dim_v, rank))
                for c in range(rank):
                    N_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                
                T_w = T.transpose(2, 0, 1).reshape(dim_w, -1) # Shape (dim_w, dim_u * dim_v)
                # T_w[k, (i,j)] = sum_c W[k, c] N_w[(i,j), c]
                W_new, _, _, _ = np.linalg.lstsq(N_w, T_w.T, rcond=None)
                W = W_new.T
                
                # Check convergence / exactness
                # Compute reconstruction
                T_rec = np.zeros_like(T)
                for c in range(rank):
                    T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                
                err = np.linalg.norm(T - T_rec)
                if err < 1e-6:
                    # Try to round to nearest 0.5
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    T_rec_int = np.zeros_like(T)
                    for c in range(rank):
                        T_rec_int += np.einsum('i,j,k->ijk', U_int[:, c], V_int[:, c], W_int[:, c])
                    
                    if np.allclose(T_rec_int, T, atol=1e-9):
                        # Convert to lists for return
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
            
        return None

    # Try rank 22 (known optimal)
    result = try_decompose(22)
    if result:
        return result
        
    # Fallback to rank 24 (schoolbook)
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
