import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] = 1 if i,j,k are valid indices for a term."""
    # The tensor T is defined such that C[i, k] = sum_j A[i, j] B[j, k]
    # In tensor form, T is a 3-tensor of shape (n, m, p) where T[i, j, k] = 1.
    # But the decomposition is sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k].
    # Wait, the standard decomposition is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This can be written as:
    # C[i, k] = sum_r (sum_j A[i, j] U[j, r]) (sum_k W[k, r] B[j, k])? No.
    # The rank-1 term is U[i, r] * V[j, r] * W[k, r].
    # We need sum_r U[i, r] V[j, r] W[k, r] = delta_{i, j, k}? No.
    # The multiplication tensor M has entries M[i, j, k] = 1.
    # We want M = sum_r u_r v_r w_r (outer product).
    # So U[i, r], V[j, r], W[k, r] are the factors.
    pass

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try to find a lower rank decomposition using ALS
    # We know rank 19 is possible for 3x3x4.
    # We split 3x3x8 into two 3x3x4.
    # Let's try to find a rank 19 decomposition for 3x3x4 first.
    
    def find_decomp_334(rank_target=19, max_iter=100):
        # Tensor T of shape (3, 3, 4) with all 1s? No.
        # The tensor for matrix multiplication 3x3x4 is T[i, j, k] = 1.
        # We want T = sum_r U[:, r] V[:, r] W[:, r]
        # U is 3xR, V is 3xR, W is 4xR
        T = np.ones((3, 3, 4))
        
        # Initialize randomly
        U = np.random.randn(3, rank_target)
        V = np.random.randn(3, rank_target)
        W = np.random.randn(4, rank_target)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U
            # U[i, r] = sum_{j, k} T[i, j, k] V[j, r] W[k, r] / sum_{j, k} V[j, r]^2 W[k, r]^2
            # This is standard ALS for tensor decomposition.
            for r in range(rank_target):
                # Compute U[:, r]
                # U[i, r] = <T[i, :, :], V[:, r] W[:, r]> / ||V[:, r] W[:, r]||^2
                # T[i, :, :] is a 3x4 matrix.
                # V[:, r] is 3x1, W[:, r] is 4x1.
                # The product V[:, r] W[:, r] is a 3x4 matrix (outer product).
                # Inner product is sum(T[i, :, :] * (V[:, r] W[:, r]))
                # Denominator is sum((V[:, r] W[:, r])^2)
                
                # Vectorize for speed
                # T_flat[i, j, k]
                # We can do this in batch for all r?
                pass
            
            # Simplified ALS update for all r at once
            # U = T x2 (V * W) / (V * W)^2? No, it's per r.
            
            # Let's do it element-wise or with matrix operations.
            # U[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r] / sum_{j,k} V[j,r]^2 W[k,r]^2
            # Numerator: For each i, U[i, :] = sum_{j,k} T[i,j,k] (V[:, r] W[:, r])
            # This is equivalent to:
            # Temp = np.einsum('ijk, jr, kr -> ir', T, V, W)
            # Denom = np.einsum('jr, kr, jr, kr -> r', V, W, V, W)
            # U = Temp / Denom[None, :]
            
            Temp_U = np.einsum('ijk, jr, kr -> ir', T, V, W)
            Denom_U = np.einsum('jr, kr, jr, kr -> r', V, W, V, W)
            U = Temp_U / (Denom_U[None, :] + 1e-9)
            
            Temp_V = np.einsum('ijk, ir, kr -> jr', T, U, W)
            Denom_V = np.einsum('ir, kr, ir, kr -> r', U, W, U, W)
            V = Temp_V / (Denom_V[None, :] + 1e-9)
            
            Temp_W = np.einsum('ijk, ir, jr -> kr', T, U, V)
            Denom_W = np.einsum('ir, jr, ir, jr -> r', U, V, U, V)
            W = Temp_W / (Denom_W[None, :] + 1e-9)
            
            # Check convergence
            Reconstructed = np.einsum('ir, jr, kr -> ijk', U, V, W)
            Error = np.linalg.norm(Reconstructed - T)
            if Error < 1e-6:
                break
        
        # Round to half-integers
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Verify
        Recon = np.einsum('ir, jr, kr -> ijk', U_round, V_round, W_round)
        if np.allclose(Recon, T, atol=1e-9):
            return U_round, V_round, W_round
        return None

    # Try to find 3x3x4 decomp
    decomp_334 = find_decomp_334(19, 200)
    
    if decomp_334 is not None:
        U1, V1, W1 = decomp_334
        # Construct 3x3x8 by concatenating two 3x3x4 decomp
        # A is 3x3, B is 3x8 = [B1, B2] where B1, B2 are 3x4
        # C = A B = [A B1, A B2]
        # U is 3x38? No, U is for A. Shape (3, rank).
        # V is for B. Shape (3, rank).
        # W is for C. Shape (8, rank).
        
        # For the first half (k=0..3):
        # U[:, 0:19] = U1
        # V[:, 0:19] = V1
        # W[0:4, 0:19] = W1
        
        # For the second half (k=4..7):
        # U[:, 19:38] = U1
        # V[:, 19:38] = V1
        # W[4:8, 19:38] = W1
        
        rank = 38
        U = np.zeros((3, rank))
        V = np.zeros((3, rank))
        W = np.zeros((8, rank))
        
        U[:, 0:19] = U1
        V[:, 0:19] = V1
        W[0:4, 0:19] = W1
        
        U[:, 19:38] = U1
        V[:, 19:38] = V1
        W[4:8, 19:38] = W1
        
        # Verify full 3x3x8
        T_full = np.ones((3, 3, 8))
        Recon_full = np.einsum('ir, jr, kr -> ijk', U, V, W)
        if np.allclose(Recon_full, T_full, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
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
