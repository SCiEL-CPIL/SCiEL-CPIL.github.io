import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Tensor dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Create the target tensor T
    # T is a 3D array of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Try to find a low rank decomposition
    # Start with a reasonable upper bound, e.g., 40
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 40 down to 30
    for rank in range(40, 29, -1):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W randomly
        np.random.seed(42) # Fixed seed for reproducibility
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # ALS Optimization
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Update U
            # T reshaped to (dim_u, dim_v * dim_w)
            T_u = T.reshape(dim_u, -1)
            VW = (V @ W.T).reshape(dim_v * dim_w) # This is not quite right for ALS
            # Correct ALS step:
            # For each column c of U, we want to minimize || T - sum_c U[:,c] V[:,c]^T W[:,c] ||
            # This is equivalent to solving for U[:,c] given V[:,c] and W[:,c]
            
            # Efficient ALS for tensor:
            # Fix V, W. Solve for U.
            # T = U * V * W (elementwise product of outer products)
            # T.reshape(dim_u, dim_v, dim_w)
            # For a fixed c, the contribution is U[:, c] * V[:, c] * W[:, c]
            # We want to minimize || T - sum_c U[:, c] (V[:, c] * W[:, c]) ||
            # Let Z_c = V[:, c] * W[:, c]
            # Then T - sum_{c' != c} U[:, c'] Z_{c'} = U[:, c] Z_c
            # This is a linear system for U[:, c].
            
            # Precompute Z for all c
            Z = V * W # Shape (max_dim, rank) - wait, V and W have different dims
            # V is (dim_v, rank), W is (dim_w, rank)
            # We need to align them.
            # The product V[:, c] * W[:, c] is not defined directly if dims differ.
            # But in the tensor T, the indices are separate.
            # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
            
            # To solve for U[:, c], we fix V and W.
            # For each i, T[i, :, :] = U[i, :] (V * W) -- no.
            # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
            # Let M_c = V[:, c] * W[:, c] ? No.
            # We can view T as a matrix of size dim_u x (dim_v * dim_w).
            # T_mat[i, j*dim_w + k] = sum_c U[i, c] V[j, c] W[k, c]
            # Let B_c be a vector of size dim_v * dim_w where B_c[j*dim_w + k] = V[j, c] W[k, c]
            # Then T_mat = U * B^T (where B is rank x dim_v*dim_w)
            # So U = T_mat * B^T * (B * B^T)^-1
            
            # Construct B
            B = np.zeros((rank, dim_v * dim_w))
            for c in range(rank):
                B[c, :] = (V[:, c] * W[:, c]).reshape(-1) # This is wrong. V and W have different lengths.
                # V[:, c] is length dim_v. W[:, c] is length dim_w.
                # The product is not elementwise in the same space.
                # We need to form the Kronecker product or similar.
                # Actually, B_c is the vectorization of V[:, c] * W[:, c]^T?
                # No, T[i, j, k] = U[i, c] V[j, c] W[k, c]
                # This is not a standard matrix product.
                # It is a tensor product.
                
            # Alternative ALS approach:
            # Flatten T to (dim_u, dim_v * dim_w)
            T_mat = T.reshape(dim_u, dim_v * dim_w)
            
            # For each c, define B_c as the vector of size dim_v * dim_w
            # B_c[j * dim_w + k] = V[j, c] * W[k, c]
            # This is the Kronecker product V[:, c] \otimes W[:, c]
            
            B = np.zeros((rank, dim_v * dim_w))
            for c in range(rank):
                B[c, :] = np.kron(V[:, c], W[:, c])
                
            # Solve U = T_mat @ B.T @ inv(B @ B.T)
            # Use least squares
            try:
                U, _, _, _ = np.linalg.lstsq(B.T, T_mat.T, rcond=None)
                U = U.T
            except:
                pass
                
            # Update V
            # T_mat_v of shape (dim_v, dim_u * dim_w)
            # T[i, j, k] = U[i, c] V[j, c] W[k, c]
            # For fixed j, sum_c V[j, c] (U[i, c] W[k, c])
            # Let C_c[i * dim_w + k] = U[i, c] W[k, c] = np.kron(U[:, c], W[:, c])
            C = np.zeros((rank, dim_u * dim_w))
            for c in range(rank):
                C[c, :] = np.kron(U[:, c], W[:, c])
            T_mat_v = T.reshape(dim_v, dim_u * dim_w)
            try:
                V, _, _, _ = np.linalg.lstsq(C.T, T_mat_v.T, rcond=None)
                V = V.T
            except:
                pass
                
            # Update W
            # T_mat_w of shape (dim_w, dim_u * dim_v)
            # Let D_c[i * dim_v + j] = U[i, c] V[j, c] = np.kron(U[:, c], V[:, c])
            D = np.zeros((rank, dim_u * dim_v))
            for c in range(rank):
                D[c, :] = np.kron(U[:, c], V[:, c])
            T_mat_w = T.reshape(dim_w, dim_u * dim_v)
            try:
                W, _, _, _ = np.linalg.lstsq(D.T, T_mat_w.T, rcond=None)
                W = W.T
            except:
                pass
                
        # Check error
        error = 0.0
        for c in range(rank):
            error += np.sum((T - np.outer(np.outer(U[:, c], V[:, c]), W[:, c]))**2)
            
        if error < 1e-4:
            # Try to snap to integers
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness
            exact = True
            for c in range(rank):
                recon = np.outer(np.outer(U_int[:, c], V_int[:, c]), W_int[:, c])
                if not np.allclose(recon, T, atol=1e-9):
                    # Check if the sum is exact
                    pass
            
            # Sum of reconstructions
            recon_total = np.zeros_like(T)
            for c in range(rank):
                recon_total += np.outer(np.outer(U_int[:, c], V_int[:, c]), W_int[:, c])
                
            if np.allclose(recon_total, T, atol=1e-9):
                best_rank = rank
                best_U = U_int
                best_V = V_int
                best_W = W_int
                break # Found a good one
                
    # If no low rank found, return schoolbook
    if best_U is None:
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
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
