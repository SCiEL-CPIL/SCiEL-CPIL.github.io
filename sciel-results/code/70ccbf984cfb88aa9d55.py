"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A: n x m, B: m x p, C: n x p
    
    # Dimensions of the factor matrices
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the tensor T[i, j, k] = 1 if C[i,k] depends on A[i,j] and B[j,k]
    # In the standard matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is a 3-way array of size (n*m, m*p, p*n)
    # Index mapping:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((dim_u, dim_v, dim_w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # We want to find a rank-k decomposition:
    # T = sum_{r=1}^k u_r v_r^T w_r^T (outer product)
    # This is equivalent to:
    # U (dim_u x k), V (dim_v x k), W (dim_w x k)
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    
    # Strategy: Try ranks from 18 upwards.
    # For each rank, we can try to find U, V, W.
    # This is a non-convex problem.
    # A simple heuristic: Use SVD on the matricizations to get initial guesses,
    # then refine. Or, since the tensor is very sparse and structured,
    # we can use a greedy algorithm to pick rank-1 components.
    
    # However, a more reliable method for small tensors is to use 
    # `scipy.optimize` or a custom solver. 
    # Given the 120s limit, we can afford a few iterations.
    
    # Let's try a different structural approach:
    # The tensor is the sum of n*m*p rank-1 tensors (schoolbook).
    # We know rank <= 30.
    # We suspect rank is 18 or 20.
    
    # Let's implement a "rank-revealing" approach.
    # We will use the fact that the tensor is defined over reals.
    # We can try to find a decomposition of rank k by solving for the
    # columns of U, V, W.
    
    # A known efficient method for small ranks is to use the 
    # "Tensor Train" or "CP decomposition" via ALS (Alternating Least Squares).
    # Let's implement a simple ALS with random restarts and verification.
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # This is a contraction.
        # We can check on a random subset of indices or all indices.
        # All indices: 6*15*10 = 900 checks. Very fast.
        for i in range(dim_u):
            for j in range(dim_v):
                # T[i,j,:] should be sum_r U[i,r] V[j,r] W[:,r]
                # Let's compute the vector for fixed i,j
                # v_i = U[i,:] (k)
                # v_j = V[j,:] (k)
                # T_recon = sum_r U[i,r] V[j,r] W[:,r]
                # This is (U[i,:] * V[j,:]) @ W.T
                # Let's do it for all i,j to be safe.
                pass
        
        # Vectorized check:
        # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # We can compute this as:
        # tmp[i,j] = sum_r U[i,r] V[j,r]  -> (dim_u x dim_v)
        # T_recon[i,j,k] = sum_j' tmp[i,j'] ... no, that's not right.
        
        # Correct vectorized check:
        # T_recon is a 3D array.
        # We can compute it by:
        # 1. Compute M_uv[i, j] = sum_r U[i, r] * V[j, r]
        # 2. Then T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # This is not simply M_uv @ W.
        
        # Let's use np.einsum
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def try_rank(k):
        """Try to find a rank-k decomposition using ALS."""
        # Random initialization
        # We want integer/half-integer entries.
        # ALS will find real entries. We can round at the end and verify.
        
        for restart in range(10):
            U = np.random.randn(dim_u, k)
            V = np.random.randn(dim_v, k)
            W = np.random.randn(dim_w, k)
            
            # Scale to match the magnitude of T
            # The entries of T are 0 or 1.
            # The product U*V*W should be 1.
            # Let's normalize the vectors.
            # This is a non-convex optimization.
            # We use ALS:
            # Fix V, W, update U:
            # T[i, :, :] = sum_r U[i, r] (V[:, r] outer W[:, r])
            # This is a linear system for U[:, r] given V, W.
            
            for it in range(100):
                # Update U
                # For each r, we need to solve for U[:, r]
                # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Let's reshape T to (dim_u, dim_v*dim_w)
                T_mat = T.reshape(dim_u, dim_v * dim_w)
                
                # Build the matrix B_r of size (dim_v*dim_w) x k
                # B_r[j*p+k, r] = V[j, r] * W[k, r]
                B_r = np.outer(V, W).reshape(dim_v * dim_w, k)
                
                # Solve T_mat = U @ B_r.T
                # U = T_mat @ B_r @ (B_r.T @ B_r)^-1
                try:
                    U_new, _, _, _ = np.linalg.lstsq(B_r.T, T_mat.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Update V
                # T[i, j, k] = sum_r V[j, r] (U[i, r] outer W[k, r])
                # Reshape T to (dim_v, dim_u*dim_w)
                # Note: T is (u, v, w). We need (v, u, w)
                T_mat_v = np.transpose(T, (1, 0, 2)).reshape(dim_v, dim_u * dim_w)
                
                B_r_v = np.outer(U, W).reshape(dim_u * dim_w, k)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(B_r_v.T, T_mat_v.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Update W
                # T[i, j, k] = sum_r W[k, r] (U[i, r] outer V[j, r])
                # Reshape T to (dim_w, dim_u*dim_v)
                T_mat_w = np.transpose(T, (2, 0, 1)).reshape(dim_w, dim_u * dim_v)
                
                B_r_w = np.outer(U, V).reshape(dim_u * dim_v, k)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(B_r_w.T, T_mat_w.T, rcond=None)
                    W = W_new.T
                except:
                    break
            
            # Check if the solution is good
            if verify(U, V, W):
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
                # If rounding fails, try integer rounding?
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
                # If exact rounding fails, return the real values?
                # The task says "integer/half-integer entries".
                # If we can't find one, this rank might not have a simple rational solution.
                # But matrix multiplication tensors always have integer decompositions.
                pass
        
        return None

    # Try ranks 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    for k in range(18, 31):
        result = try_rank(k)
        if result:
            return {"U": result[0], "V": result[1], "W": result[2]}
            
    # Fallback: Schoolbook rank 30
    rank = n * m * p
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
