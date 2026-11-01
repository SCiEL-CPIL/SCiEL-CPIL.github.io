"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Uses block-diagonal Strassen decomposition.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients
    strassen_A = [
        [1, 0, 0, 1],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1]  # P7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: e+h
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: f-h
        [-1, 0, 1, 0], # P4: g-e
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: e+f
        [0, 0, 1, 1]   # P7: g+h
    ]
    
    w_contrib = [
        [1, 0, 0, 1],  # P1 -> C00, C11
        [0, 0, 1, -1], # P2 -> C10, C11
        [0, 1, 0, 1],  # P3 -> C01, C11
        [1, 0, 1, 0],  # P4 -> C00, C10
        [-1, 1, 0, 0], # P5 -> C00, C01
        [0, 0, 0, 1],  # P6 -> C11
        [1, 0, 0, 0]   # P7 -> C00
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            k0 = 2 * b
            k1 = 2 * b + 1
            
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    # Verify correctness with a simple test
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute via decomposition
    C_computed = np.zeros((n, p))
    for r in range(rank):
        a_part = np.dot(U_np[:, r], A_test.flatten())
        b_part = np.dot(V_np[:, r], B_test.flatten())
        w_vec = W_np[:, r]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += a_part * b_part * w_vec[k * n + i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback: return schoolbook if Strassen fails verification
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][idx] = 1
                    V_s[j * p + k][idx] = 1
                    W_s[k * n + i][idx] = 1
                    idx += 1
        
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
