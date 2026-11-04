"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Verification function
    def verify(U, V, W, rank):
        # Test with a few random matrices
        for _ in range(50):
            A = np.random.randint(-10, 10, (n, m)).astype(float)
            B = np.random.randint(-10, 10, (m, p)).astype(float)
            C_expected = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(rank):
                # u_i = sum_j A_ij U[i*m+j, r]
                u = np.array([sum(A[i, j] * U[i * m + j][r] for j in range(m)) for i in range(n)])
                # v_k = sum_j V[j*p+k, r] B_jk
                v = np.array([sum(V[j * p + k][r] * B[j, k] for j in range(m)) for k in range(p)])
                # C_ik += u_i * v_k * W[k*n+i, r]
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += u[i] * v[k] * W[k * n + i][r]
                        
            if not np.allclose(C_expected, C_calc, atol=1e-9):
                return False
        return True

    # Try Rank 20
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Placeholder for Rank 20 coefficients
    # If not verified, fallback to Rank 23
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Try Rank 23 (Winograd)
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Winograd's Rank 23 Algorithm
    # Terms: (u_coeffs, v_coeffs, w_coeffs)
    # u: (i, j, coeff), v: (j, k, coeff), w: (k, i, coeff)
    terms = [
        # 1
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # 2
        ([(0,1,1)], [(1,0,1)], [(0,0,1)]),
        # 3
        ([(1,0,1)], [(0,1,1)], [(0,0,1)]),
        # 4
        ([(1,1,1)], [(1,1,1)], [(0,0,1)]),
        # 5
        ([(0,0,1)], [(1,1,1)], [(0,1,1)]),
        # 6
        ([(0,1,1)], [(1,0,1)], [(0,1,1)]),
        # 7
        ([(1,0,1)], [(0,1,1)], [(0,1,1)]),
        # 8
        ([(1,1,1)], [(1,0,1)], [(0,1,1)]),
        # 9
        ([(0,0,1)], [(0,1,1)], [(1,0,1)]),
        # 10
        ([(0,1,1)], [(0,0,1)], [(1,0,1)]),
        # 11
        ([(1,0,1)], [(1,1,1)], [(1,0,1)]),
        # 12
        ([(1,1,1)], [(1,0,1)], [(1,0,1)]),
        # 13
        ([(0,0,1)], [(1,0,1)], [(1,1,1)]),
        # 14
        ([(0,1,1)], [(0,1,1)], [(1,1,1)]),
        # 15
        ([(1,0,1)], [(0,0,1)], [(1,1,1)]),
        # 16
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # 17
        ([(0,0,1), (0,1,1)], [(0,0,1), (0,1,1)], [(0,0,1), (0,1,1)]),
        # 18
        ([(1,0,1), (1,1,1)], [(1,0,1), (1,1,1)], [(1,0,1), (1,1,1)]),
        # 19
        ([(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)]),
        # 20
        ([(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)]),
        # 21
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # 22
        ([(0,1,1), (1,0,1)], [(0,1,1), (1,0,1)], [(0,1,1), (1,0,1)]),
        # 23
        ([(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]),
    ]
    
    for r, (u_terms, v_terms, w_terms) in enumerate(terms):
        for i, j, coeff in u_terms:
            U[i*m+j][r] = coeff
        for j, k, coeff in v_terms:
            V[j*p+k][r] = coeff
        for k, i, coeff in w_terms:
            W[k*n+i][r] = coeff
            
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

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
# EVOLVE-BLOCK-END
