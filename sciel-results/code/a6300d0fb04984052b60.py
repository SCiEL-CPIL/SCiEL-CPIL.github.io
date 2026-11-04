import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T for matrix multiplication A(nxm) * B(mxp) -> C(nxp)
    # T has shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1 for all i, j, k
    # All other entries are 0.
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Try to find a low rank decomposition
    # We will try ranks from 50 down to 30
    # Using a randomized approach:
    # 1. Randomly initialize U, V with small integers
    # 2. Solve for W using least squares
    # 3. Verify if U, V, W exactly reproduce T
    
    # Construct T as a vector of length size_u * size_v * size_w? 
    # No, we can represent the equation:
    # sum_r U[:, r] outer V[:, r] = T_slice?
    # Actually, T is a 3-tensor.
    # We can flatten T to a matrix M of shape (size_u * size_v, size_w)
    # where M[(i*m+j)*(m*p) + (j*p+k), k*n+i] = 1
    
    # Let's build the linear system A * w = b
    # The equation is: sum_r U_{alpha, r} V_{beta, r} W_{gamma, r} = T_{alpha, beta, gamma}
    # For each gamma, we have: sum_r (U_{alpha, r} V_{beta, r}) W_{gamma, r} = T_{alpha, beta, gamma}
    # This is not linear in U, V, W simultaneously.
    
    # Better approach:
    # Fix U and V. Then the equation is linear in W.
    # For each r, let M_r = U[:, r] outer V[:, r] (shape size_u x size_v)
    # We want sum_r M_r * W[:, r] to "reproduce" T.
    # Wait, the tensor contraction is:
    # C_{alpha, gamma} = sum_beta A_{alpha, beta} B_{beta, gamma}
    # In tensor form: T_{alpha, beta, gamma}
    # Decomposition: T_{alpha, beta, gamma} = sum_r U_{alpha, r} V_{beta, r} W_{gamma, r}
    
    # So for each r, we have a rank-1 matrix M_r = U_r V_r^T (if we view U, V as columns)
    # No, U_{alpha, r} is the entry.
    # The matrix formed by fixing r is U_r V_r^T.
    # We need sum_r U_r V_r^T \otimes W_r? No.
    # The tensor T is sum_r U_r \otimes V_r \otimes W_r.
    
    # If we fix U and V, we can solve for W.
    # Let's flatten the "input" part (alpha, beta) and "output" part (gamma).
    # For each gamma, the vector T_{:, :, gamma} (shape size_u x size_v) must equal
    # sum_r W_{gamma, r} (U_r \otimes V_r)
    # where U_r \otimes V_r is the vectorized outer product of U[:, r] and V[:, r].
    
    # So we build a matrix S of shape (size_u * size_v, rank)
    # S[alpha * size_v + beta, r] = U_{alpha, r} * V_{beta, r}
    # And a matrix B of shape (size_u * size_v, size_w)
    # B[alpha * size_v + beta, gamma] = T_{alpha, beta, gamma}
    
    # Then we solve S * W^T = B ?
    # W has shape (size_w, rank).
    # B = S * W^T  =>  B^T = W * S^T
    # W = B^T (S^T)^+
    
    def try_rank(rank, seed):
        np.random.seed(seed)
        # Initialize U and V with small integers (-1, 0, 1)
        # Sparse initialization helps with exactness
        U = np.random.choice([-1, 0, 1], size=(size_u, rank))
        V = np.random.choice([-1, 0, 1], size=(size_v, rank))
        
        # Build S
        # S[i, r] = U[i//size_v, r] * V[i%size_v, r]
        # This is expensive to build in a loop, use broadcasting
        # U_expanded: (size_u, 1, rank)
        # V_expanded: (1, size_v, rank)
        # S_flat: (size_u * size_v, rank)
        
        U_exp = U[:, np.newaxis, :] # (size_u, 1, rank)
        V_exp = V[np.newaxis, :, :] # (1, size_v, rank)
        S_flat = (U_exp * V_exp).reshape(size_u * size_v, rank)
        
        # Build B (Target Tensor flattened)
        B = np.zeros((size_u * size_v, size_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    B[alpha * size_v + beta, gamma] = 1.0
        
        # Solve W = B^T (S^T)^+
        # B^T is (size_w, size_u * size_v)
        # S^T is (rank, size_u * size_v)
        # W = B^T @ pinv(S^T)
        
        try:
            # Use least squares
            # We want to minimize || S * W^T - B ||_F
            # This is equivalent to solving for each column of B
            # W^T = pinv(S) @ B
            # W = (pinv(S) @ B)^T
            
            # Check rank of S
            # If S is rank deficient, pinv still works but solution may not be unique
            W = np.linalg.lstsq(S_flat, B, rcond=None)[0].T
        except:
            return None
            
        # Check if the decomposition is exact
        # Reconstruct T from U, V, W
        # T_recon[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
        
        # We can check by multiplying back
        # Residual = B - S_flat @ W
        residual = B - S_flat @ W
        
        # Check if residual is close to zero
        # Since we want integer/half-integer, we check for small error
        if np.allclose(residual, 0, atol=1e-9):
            # Verify entries are integer or half-integer
            # W might have floating point errors
            # Round W to nearest 0.5
            W_rounded = np.round(W * 2) / 2.0
            residual_rounded = B - S_flat @ W_rounded
            if np.allclose(residual_rounded, 0, atol=1e-9):
                return U, V, W_rounded
        return None

    # Try ranks from 50 down to 40
    for rank in range(50, 39, -1):
        for seed in range(20):
            res = try_rank(rank, seed)
            if res is not None:
                U, V, W = res
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to trivial decomposition if no better found
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
