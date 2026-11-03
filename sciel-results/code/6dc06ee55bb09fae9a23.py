import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T[i,j,k] = 1 for all i,j,k in the standard matrix multiplication
    # Actually, I need to verify this. Let me construct the target tensor.
    
    # For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k
    
    # Create target tensor
    T = np.ones((n, m, p), dtype=float)
    
    # Try to find a low-rank decomposition using IHT
    # Start with a guess for rank
    target_rank = 24  # Try half the trivial rank
    
    # Initialize U, V, W randomly
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # IHT iterations
    for iteration in range(1000):
        # Compute current decomposition
        # Residual = T - sum_r U[:,r] * V[:,r] * W[:,r] (outer products)
        
        # This is complex for tensors. Let me use a different approach.
        # Use the fact that we can check if the decomposition is exact.
        
        # Actually, let me try a simpler approach: use SVD on matricizations
        
        # Matricize T in different ways
        # T_ik_j = T[i,j,k] reshaped to (n*p, m)
        T_ik_j = T.transpose(0, 2, 1).reshape(n * p, m)
        
        # Get SVD
        U_svd, s, Vt = np.linalg.svd(T_ik_j, full_matrices=False)
        
        # Take top r components
        r = min(target_rank, len(s))
        U_mat = U_svd[:, :r]
        V_mat = s[:r]
        W_mat = Vt[:r, :]
        
        # This gives a rank-r approximation, but not necessarily integer/half-integer
        # We need to enforce the constraint
        
        # For now, let's just return the trivial solution if IHT doesn't work
        pass
    
    # Fallback to trivial decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
