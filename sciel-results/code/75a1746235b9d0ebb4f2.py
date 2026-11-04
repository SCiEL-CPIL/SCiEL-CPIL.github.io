import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    # Build the target tensor T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((9, 9, 9))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the 3x3x3 matrix multiplication tensor."""
        r = U.shape[1]
        R = np.einsum('at,bt,ct->abc', U, V, W)
        if not np.allclose(R, T, atol=tol, rtol=0):
            return False
        return True

    def is_half_integer(x):
        """Check if x is a half-integer."""
        return np.isclose(2 * x, np.round(2 * x), atol=1e-9)

    def round_to_half_int(x):
        """Round a matrix to the nearest half-integer."""
        return np.round(2 * x) / 2.0

    # Rank 27 Fallback (schoolbook)
    U27 = np.zeros((9, 27))
    V27 = np.zeros((9, 27))
    W27 = np.zeros((9, 27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i*m+j, col] = 1
                V27[j*p+k, col] = 1
                W27[k*n+i, col] = 1
                col += 1

    # Attempt to construct a rank-23 Winograd-style decomposition
    # The Winograd F(3,3) algorithm structure:
    # We use the fact that C = A @ B can be computed via:
    # 1. Transform rows of A: u_i = M @ A[i,:]  (M is 3x3)
    # 2. Transform cols of B: v_k = B[:,k] @ N  (N is 3x3)
    # 3. Multiply: w_{i,k,l} = u_i[l] * v_k[l]
    # 4. Inverse transform: C[i,k] = sum_l w_{i,k,l} * L[l]
    #
    # For rank reduction, we exploit that some products can be shared.
    # The standard Winograd 3x3 has rank 23.
    #
    # Let's try to construct it explicitly.
    # The key is to find matrices M, N, L such that:
    # sum_l M[i,l] * N[j,l] * L[k,l] = delta_{i,k} * delta_{j,j} ... 
    # Actually, the tensor is T[i,j,k] = delta_{i,k} for the (i,j,k) index.
    # Wait, T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i.
    # So T[i,j,k] (in matrix multiplication indices) = 1 for all i,j,k.
    # That's not right. Let me re-read.
    #
    # T[a,b,c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i.
    # So for each (i,j,k), we have T[i*m+j, j*p+k, k*n+i] = 1.
    # This is the standard matrix multiplication tensor.
    #
    # The Winograd construction for 3x3:
    # We want to find rank-1 terms such that:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 if (i,j,k) is valid, 0 otherwise.
    #
    # Let me try a direct construction based on the Winograd algorithm.
    # The algorithm computes C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # Using the F(3,3) Winograd scheme:
    # Define transforms such that the products can be shared.
    #
    # Actually, let me just try to find a rank-23 decomposition numerically
    # using a structured initialization based on the Winograd idea.
    
    def attempt_rank_als(r, max_iter=2000, seed=42):
        """Attempt to find a rank r decomposition using ALS with structured init."""
        np.random.seed(seed)
        
        # Initialize with a structured guess based on Winograd
        # U[a, t], V[b, t], W[c, t]
        # For the Winograd structure, we can initialize based on the transform matrices.
        
        # Try different initialization strategies
        for init_strategy in range(5):
            if time.time() > deadline:
                return None
            
            if init_strategy == 0:
                # Random initialization
                U = np.random.randn(9, r) * 0.5
                V = np.random.randn(9, r) * 0.5
                W = np.random.randn(9, r) * 0.5
            elif init_strategy == 1:
                # Initialize with sparse structure
                U = np.zeros((9, r))
                V = np.zeros((9, r))
                W = np.zeros((9, r))
                # Assign random non-zero entries
                for t in range(r):
                    # Pick 3-4 non-zero entries in each
                    u_idx = np.random.choice(9, size=min(4, r//3 + 2), replace=False)
                    v_idx = np.random.choice(9, size=min(4, r//3 + 2), replace=False)
                    w_idx = np.random.choice(9, size=min(4, r//3 + 2), replace=False)
                    for idx in u_idx:
                        U[idx, t] = np.random.choice([-1, 1])
                    for idx in v_idx:
                        V[idx, t] = np.random.choice([-1, 1])
                    for idx in w_idx:
                        W[idx, t] = np.random.choice([-1, 1])
            elif init_strategy == 2:
                # Initialize with the schoolbook structure and perturb
                U = np.zeros((9, r))
                V = np.zeros((9, r))
                W = np.zeros((9, r))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < r:
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = 1
                                W[k*n+i, col] = 1
                                col += 1
                # Perturb
                U += np.random.randn(9, r) * 0.1
                V += np.random.randn(9, r) * 0.1
                W += np.random.randn(9, r) * 0.1
            elif init_strategy == 3:
                # Winograd-inspired initialization
                # M = [[1,1,0],[1,0,1],[1,-1,0]], N = [[1,0,1],[1,1,0],[1,0,-1]]
                # Try to use these as structure
                M = np.array([[1,1,0],[1,0,1],[1,-1,0]], dtype=float)
                N = np.array([[1,0,1],[1,1,0],[1,0,-1]], dtype=float)
                U = np.zeros((9, r))
                V = np.zeros((9, r))
                W = np.zeros((9, r))
                # Fill with patterns from M and N
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            t = i * 3 + j * 3 + k
                            if t < r:
                                U[i*3+j, t] = M[i, k]
                                V[j*3+k, t] = N[k, j]
                                W[k*3+i, t] = 1.0
            else:
                # Sparse random with larger values
                U = np.random.randn(9, r) * 2
                V = np.random.randn(9, r) * 2
                W = np.random.randn(9, r) * 2
                # Sparsify
                for t in range(r):
                    U[np.random.choice(9, size=5, replace=False), t] = 0
                    V[np.random.choice(9, size=5, replace=False), t] = 0
                    W[np.random.choice(9, size=5, replace=False), t] = 0
            
            T_flat = T.reshape(9, 81)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Update U
                M_VW = np.einsum('bt,ct->bct', V, W)  # Shape (9, 9, r)
                M_VW_flat = M_VW.reshape(81, r)
                try:
                    G = M_VW_flat @ M_VW_flat.T
                    G_inv = np.linalg.inv(G + 1e-9 * np.eye(r))
                    U_new = T_flat @ M_VW_flat.T @ G_inv
                    # Check if this is an improvement
                    R_old = np.einsum('at,bt,ct->abc', U, V, W)
                    R_new = np.einsum('at,bt,ct->abc', U_new, V, W)
                    if np.linalg.norm(R_new - T) < np.linalg.norm(R_old - T):
                        U = U_new
                except:
                    pass
                
                # Update V
                M_UW = np.einsum('at,ct->act', U, W)  # Shape (9, 9, r)
                M_UW_flat = M_UW.reshape(81, r)
                try:
                    G = M_UW_flat @ M_UW_flat.T
                    G_inv = np.linalg.inv(G + 1e-9 * np.eye(r))
                    V_new = T_flat @ M_UW_flat.T @ G_inv
                    R_old = np.einsum('at,bt,ct->abc', U, V, W)
                    R_new = np.einsum('at,bt,ct->abc', U, V_new, W)
                    if np.linalg.norm(R_new - T) < np.linalg.norm(R_old - T):
                        V = V_new
                except:
                    pass
                
                # Update W
                M_UV = np.einsum('at,bt->abt', U, V)  # Shape (9, 9, r)
                M_UV_flat = M_UV.reshape(81, r)
                try:
                    G = M_UV_flat @ M_UV_flat.T
                    G_inv = np.linalg.inv(G + 1e-9 * np.eye(r))
                    W_new = T_flat @ M_UV_flat.T @ G_inv
                    R_old = np.einsum('at,bt,ct->abc', U, V, W)
                    R_new = np.einsum('at,bt,ct->abc', U, V, W_new)
                    if np.linalg.norm(R_new - T) < np.linalg.norm(R_old - T):
                        W = W_new
                except:
                    pass
                
                # Check convergence
                if it % 20 == 0:
                    R = np.einsum('at,bt,ct->abc', U, V, W)
                    if np.allclose(R, T, atol=1e-8, rtol=0):
                        # Try to round to integers or half-integers
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                        
                        U_half = round_to_half_int(U)
                        V_half = round_to_half_int(V)
                        W_half = round_to_half_int(W)
                        if verify(U_half, V_half, W_half):
                            return U_half, V_half, W_half
                        
                        # Try a mix: round some to int, some to half
                        for frac in [0.5, 0.25, 0.75]:
                            U_mix = np.where(np.abs(U - np.round(U)) < 1e-6, np.round(U), round_to_half_int(U))
                            V_mix = np.where(np.abs(V - np.round(V)) < 1e-6, np.round(V), round_to_half_int(V))
                            W_mix = np.where(np.abs(W - np.round(W)) < 1e-6, np.round(W), round_to_half_int(W))
                            if verify(U_mix, V_mix, W_mix):
                                return U_mix, V_mix, W_mix
        
        return None

    # Try to find rank 23
    for seed in range(15):
        if time.time() > deadline - 30:
            break
        res = attempt_rank_als(23, max_iter=1500, seed=seed)
        if res:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Try rank 24 as a middle ground
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        res = attempt_rank_als(24, max_iter=1000, seed=seed + 100)
        if res:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to Rank 27
    return {
        "U": U27.tolist(),
        "V": V27.tolist(),
        "W": W27.tolist()
    }
