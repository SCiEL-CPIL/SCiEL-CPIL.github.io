import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Fallback: Schoolbook Rank 27
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Attempt Rank 22 Decomposition (Strassen 2x2 + Schoolbook Border)
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_dict, v_dict, w_dict, r):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] += val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] += val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] += val

    # Strassen Terms for 2x2 block (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, 0)
    # M2 = (A10+A11)B00
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}, 1)
    # M3 = A00(B10-B11)
    add_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}, 2)
    # M4 = A11(B01-B00)
    add_term({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}, 3)
    # M5 = (A00+A01)B11
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}, 4)
    # M6 = (A10-A00)(B00+B01)
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}, 5)
    # M7 = (A01-A11)(B10+B11)
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}, 6)

    # Border Terms (Schoolbook)
    # C02 = A00B02 + A01B12 + A02B22
    add_term({(0,0):1}, {(0,2):1}, {(2,0):1}, 7)
    add_term({(0,1):1}, {(1,2):1}, {(2,0):1}, 8)
    add_term({(0,2):1}, {(2,2):1}, {(2,0):1}, 9)
    
    # C12 = A10B02 + A11B12 + A12B22
    add_term({(1,0):1}, {(0,2):1}, {(2,1):1}, 10)
    add_term({(1,1):1}, {(1,2):1}, {(2,1):1}, 11)
    add_term({(1,2):1}, {(2,2):1}, {(2,1):1}, 12)
    
    # C20 = A20B00 + A21B10 + A22B20
    add_term({(2,0):1}, {(0,0):1}, {(0,2):1}, 13)
    add_term({(2,1):1}, {(1,0):1}, {(0,2):1}, 14)
    add_term({(2,2):1}, {(2,0):1}, {(0,2):1}, 15)
    
    # C21 = A20B01 + A21B11 + A22B21
    add_term({(2,0):1}, {(0,1):1}, {(1,2):1}, 16)
    add_term({(2,1):1}, {(1,1):1}, {(1,2):1}, 17)
    add_term({(2,2):1}, {(2,1):1}, {(1,2):1}, 18)
    
    # C22 = A20B02 + A21B12 + A22B22
    add_term({(2,0):1}, {(0,2):1}, {(2,2):1}, 19)
    add_term({(2,1):1}, {(1,2):1}, {(2,2):1}, 20)
    add_term({(2,2):1}, {(2,2):1}, {(2,2):1}, 21)

    # Verification
    try:
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            A_r = 0
            for i in range(n):
                for j in range(m):
                    A_r += U[i * m + j][r] * A[i, j]
            B_r = 0
            for j in range(m):
                for k in range(p):
                    B_r += V[j * p + k][r] * B[j, k]
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W[k * n + i][r] * A_r * B_r
        
        if np.allclose(C_computed, C_expected):
            return {"U": U, "V": V, "W": W}
        else:
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
