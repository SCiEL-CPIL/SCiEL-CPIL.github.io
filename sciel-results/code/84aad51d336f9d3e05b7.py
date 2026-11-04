"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    def schoolbook_3x3x2():
        """Schoolbook decomposition for 3x3x2, rank 18"""
        n2, m2, p2 = 3, 3, 2
        rank = n2 * m2 * p2
        U = np.zeros((n2 * m2, rank), dtype=float)
        V = np.zeros((m2 * p2, rank), dtype=float)
        W = np.zeros((p2 * n2, rank), dtype=float)
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U[i * m2 + j, col] = 1
                    V[j * p2 + k, col] = 1
                    W[k * n2 + i, col] = 1
                    col += 1
        return U, V, W

    def bsc_3x3x3():
        """BSC rank-23 decomposition for 3x3x3"""
        n3, m3, p3 = 3, 3, 3
        rank = 23
        U = np.zeros((n3 * m3, rank), dtype=float)
        V = np.zeros((m3 * p3, rank), dtype=float)
        W = np.zeros((p3 * n3, rank), dtype=float)
        
        # BSC algorithm coefficients (Bini, Strassen, Caporaso 1980)
        # Using a known valid representation
        # Rows of U: (i,j) for i,j in 0..2
        # Rows of V: (j,k) for j,k in 0..2  
        # Rows of W: (k,i) for k,i in 0..2
        
        # The 23 terms of BSC:
        # Term 1: A00*B00 -> C00
        U[0*3+0, 0] = 1; V[0*3+0, 0] = 1; W[0*3+0, 0] = 1
        # Term 2: A01*B10 -> C00
        U[0*3+1, 1] = 1; V[1*3+0, 1] = 1; W[0*3+0, 1] = 1
        # Term 3: A02*B20 -> C00
        U[0*3+2, 2] = 1; V[2*3+0, 2] = 1; W[0*3+0, 2] = 1
        # Term 4: A10*B01 -> C01
        U[1*3+0, 3] = 1; V[0*3+1, 3] = 1; W[1*3+0, 3] = 1
        # Term 5: A11*B11 -> C01
        U[1*3+1, 4] = 1; V[1*3+1, 4] = 1; W[1*3+0, 4] = 1
        # Term 6: A12*B21 -> C01
        U[1*3+2, 5] = 1; V[2*3+1, 5] = 1; W[1*3+0, 5] = 1
        # Term 7: A20*B02 -> C02
        U[2*3+0, 6] = 1; V[0*3+2, 6] = 1; W[2*3+0, 6] = 1
        # Term 8: A21*B12 -> C02
        U[2*3+1, 7] = 1; V[1*3+2, 7] = 1; W[2*3+0, 7] = 1
        # Term 9: A22*B22 -> C02
        U[2*3+2, 8] = 1; V[2*3+2, 8] = 1; W[2*3+0, 8] = 1
        
        # Now the 14 additional terms that make BSC rank 23 instead of schoolbook 27
        # These involve combinations that exploit algebraic structure
        
        # Term 10: (A00+A11)*(B10+B21) - A00*B10 - A11*B21
        # Simplified: A00*B21 + A11*B10
        U[0*3+0, 9] = 1; V[2*3+1, 9] = 1; W[1*3+0, 9] = 1
        U[1*3+1, 9] = 1; V[1*3+0, 9] = 1; W[0*3+1, 9] = 1
        
        # This approach of manually deriving BSC is error-prone.
        # Let me use a verified set of coefficients instead.
        
        # Actually, let me just return schoolbook for 3x3x3 as well, 
        # and use a different splitting strategy.
        
        # For now, return schoolbook 3x3x3 (rank 27)
        rank = n3 * m3 * p3
        U = np.zeros((n3 * m3, rank), dtype=float)
        V = np.zeros((m3 * p3, rank), dtype=float)
        W = np.zeros((p3 * n3, rank), dtype=float)
        col = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U[i * m3 + j, col] = 1
                    V[j * p3 + k, col] = 1
                    W[k * n3 + i, col] = 1
                    col += 1
        return U, V, W, 27

    # Try rank 41: BSC(23) for 3x3x3 + schoolbook(18) for 3x3x2
    # But since I don't have reliable BSC coefficients, let me try a different approach.
    
    # Alternative: Use schoolbook but verify it works
    # Schoolbook rank 45
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                if W[k * n + i, r] != 0:
                    u_term = sum(U[i * m + j, r] * A[i, j] for j in range(m))
                    v_term = sum(V[j * p + k, r] * B[j, k] for j in range(m))
                    C_computed[i, k] += W[k * n + i, r] * u_term * v_term
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
