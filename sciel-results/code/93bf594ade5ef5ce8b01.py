"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Strassen coefficients for M_{2,2,2}
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook coefficients for M_{2,2,1}
    U_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_school = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_school = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # Build fallback: 7 Strassen + 1 schoolbook = rank 53
    def build_fallback():
        rank = 7 * 7 + 4
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col_offset = 0
        for block in range(7):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = col_offset + r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = U_strassen[i * m + j, r]
                for j in range(m):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(n):
                        W[kk * n + i, c] = W_strassen[k * n + i, r]
            col_offset += 7
        
        k_last = 14
        for r in range(4):
            c = col_offset + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = U_school[i * m + j, r]
            for j in range(m):
                V[j * p + k_last, c] = V_school[j, r]
            for i in range(n):
                W[k_last * n + i, c] = W_school[i, r]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find M_{2,2,3} with rank 10
    def try_m223_rank10():
        """Try to find rank-10 decomposition for M_{2,2,3}."""
        if time.time() > deadline - 10:
            return None
        
        # M_{2,2,3}: i in {0,1}, j in {0,1}, k in {0,1,2}
        # Tensor: T[i,j,k] = 1 if i==k else 0
        # We need U (4x10), V (6x10), W (6x10) such that
        # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
        
        # Use a structured approach: try to extend Strassen for k=0,1 and handle k=2
        # Or use a different decomposition
        
        # Let's try: use 6 terms for k=0,1 (partial Strassen) and 4 for k=2
        # Actually, let me try a direct approach with random initialization and Newton
        
        np.random.seed(42)
        
        # Initialize with a plausible structure
        U = np.random.randn(4, 10) * 0.5
        V = np.random.randn(6, 10) * 0.5
        W = np.random.randn(6, 10) * 0.5
        
        # The constraint is nonlinear: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
        # This is a system of 12 equations (2*2*3) with 30 unknowns per matrix
        
        # Let me try a different approach: fix the structure
        # Use the fact that T[i,j,k] = delta_{i,k}
        
        # Try a "block diagonal" approach where each k is handled separately
        # but with some sharing
        
        # Actually, let me just return None and use the fallback
        # The search for rank-10 M_{2,2,3} is complex and may not succeed in time
        
        return None
    
    # Try alternative: 6 Strassen (rank 42) + M_{2,2,3} with rank 11 (schoolbook)
    # Total: 42 + 11 = 53, same as fallback
    
    # Try: 5 Strassen (rank 35) + M_{2,2,5} 
    # M_{2,2,5} schoolbook is rank 20, total 55 - worse
    
    # The best we can do with known blocks is 53
    
    # Let me try to verify if we can do 52 by finding M_{2,2,3} with rank 10
    result = try_m223_rank10()
    
    if result is not None:
        # Build full solution with rank 52
        rank = 6 * 7 + 10
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # 6 Strassen blocks for k=0..11
        for block in range(6):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = block * 7 + r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = U_strassen[i * m + j, r]
                for j in range(m):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(n):
                        W[kk * n + i, c] = W_strassen[k * n + i, r]
        
        # M_{2,2,3} for k=12,13,14
        U_223, V_223, W_223 = result
        U[:, 42:] = U_223
        V[:, 42:] = V_223
        W[:, 42:] = W_223
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return build_fallback()
# EVOLVE-BLOCK-END
