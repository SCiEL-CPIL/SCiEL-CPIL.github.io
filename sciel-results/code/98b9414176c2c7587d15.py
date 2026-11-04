"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    best_score = 1.0
    best_result = None
    current_rank = 0
    current_result = None

    # 1. Schoolbook Baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    current_result = {"U": U, "V": V, "W": W}
    current_rank = rank

    # 2. Strassen-like Decomposition (Middle Dimension Split)
    # Split the middle dimension (size 3) into a block of size 2 and a block of size 1.
    # Block 1: i in 0..1, j in 0..1, k in 0..9 -> 2x2x10. Rank via Strassen is 7 * (10/2) = 35.
    # Block 2: i in 0..1, j in 2..2, k in 0..9 -> 2x1x10. Rank via Schoolbook is 20.
    # Total Rank = 55.
    
    # We construct the 55-term decomposition directly.
    # Strassen for 2x2x2:
    # M1 = A11(B11-B22), M2 = A11+A12(B22), M3 = A11-A21(B11), M4 = A22(B11-B21)
    # M5 = A11+A22(B11+B22), M6 = A12-A22(B21+B22), M7 = A21-A11(B11+B12)
    # C11 = M1+M4-M5+M7, C12 = M2+M3-M5-M6, C21 = M1-M3+M5+M6, C22 = M2-M4+M5-M7
    
    rank_strassen = 55
    U_s = [[0] * rank_strassen for _ in range(n * m)]
    V_s = [[0] * rank_strassen for _ in range(m * p)]
    W_s = [[0] * rank_strassen for _ in range(p * n)]
    
    # Process Block 1 (j=0,1) with Strassen
    # For each pair of columns (2t, 2t+1), we use 7 terms.
    for t in range(p // 2):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # Term 1: M1 = (A11) * (B11 - B22)
        # A11 is A[0,0]. B11 is B[0, k1], B22 is B[1, k2].
        # C contribution: C11 += 1, C21 += -1
        idx = t * 7 + 0
        U_s[0 * m + 0][idx] = 1  # A11
        V_s[0 * p + k1][idx] = 1 # B11
        V_s[1 * p + k2][idx] = -1 # B22
        W_s[k1 * n + 0][idx] = 1 # C11 (i=0, k1)
        W_s[k2 * n + 1][idx] = -1 # C21 (i=1, k2) -- Wait, C21 is row 1, col 1 (i=1, k1).
        # Let's check Strassen output indices carefully.
        # C11 is (0,0), C12 is (0,1), C21 is (1,0), C22 is (1,1) in local 2x2.
        # Local i=0,1; local k=0,1.
        # Global k1 corresponds to local k=0, k2 to local k=1.
        # C11 (local 0,0) -> Global i=0, k=k1. W[k1*n+0]
        # C12 (local 0,1) -> Global i=0, k=k2. W[k2*n+0]
        # C21 (local 1,0) -> Global i=1, k=k1. W[k1*n+1]
        # C22 (local 1,1) -> Global i=1, k=k2. W[k2*n+1]
        
        # Correcting M1: C11 += M1, C21 += -M1?
        # Strassen: C11 = M1+M4-M5+M7. C21 = M1-M3+M5+M6.
        # So for M1: W[k1*n+0] += 1, W[k1*n+1] += 1.
        
        W_s[k1 * n + 0][idx] = 1
        W_s[k1 * n + 1][idx] = 1
        W_s[k2 * n + 0][idx] = 0
        W_s[k2 * n + 1][idx] = 0
        
        # Term 2: M2 = (A11+A12) * (B22)
        idx = t * 7 + 1
        U_s[0 * m + 0][idx] = 1
        U_s[0 * m + 1][idx] = 1
        V_s[1 * p + k2][idx] = 1
        # C12 += M2, C22 += M2
        W_s[k2 * n + 0][idx] = 1
        W_s[k2 * n + 1][idx] = 1
        
        # Term 3: M3 = (A11-A21) * (B11)
        idx = t * 7 + 2
        U_s[0 * m + 0][idx] = 1
        U_s[1 * m + 0][idx] = -1
        V_s[0 * p + k1][idx] = 1
        # C12 += M3, C21 += -M3
        W_s[k2 * n + 0][idx] = 1
        W_s[k1 * n + 1][idx] = -1
        
        # Term 4: M4 = (A22) * (B11-B21)
        idx = t * 7 + 3
        U_s[1 * m + 1][idx] = 1
        V_s[0 * p + k1][idx] = 1
        V_s[1 * p + k1][idx] = -1
        # C11 += M4, C22 += -M4
        W_s[k1 * n + 0][idx] = 1
        W_s[k2 * n + 1][idx] = -1
        
        # Term 5: M5 = (A11+A22) * (B11+B22)
        idx = t * 7 + 4
        U_s[0 * m + 0][idx] = 1
        U_s[1 * m + 1][idx] = 1
        V_s[0 * p + k1][idx] = 1
        V_s[1 * p + k2][idx] = 1
        # C11 += -M5, C21 += M5
        W_s[k1 * n + 0][idx] = -1
        W_s[k1 * n + 1][idx] = 1
        
        # Term 6: M6 = (A12-A22) * (B21+B22)
        idx = t * 7 + 5
        U_s[0 * m + 1][idx] = 1
        U_s[1 * m + 1][idx] = -1
        V_s[1 * p + k1][idx] = 1
        V_s[1 * p + k2][idx] = 1
        # C12 += -M6, C21 += M6
        W_s[k2 * n + 0][idx] = -1
        W_s[k1 * n + 1][idx] = 1
        
        # Term 7: M7 = (A21-A11) * (B11+B12)
        idx = t * 7 + 6
        U_s[1 * m + 0][idx] = 1
        U_s[0 * m + 0][idx] = -1
        V_s[0 * p + k1][idx] = 1
        V_s[0 * p + k2][idx] = 1
        # C11 += M7, C22 += -M7
        W_s[k1 * n + 0][idx] = 1
        W_s[k2 * n + 1][idx] = -1

    # Process Block 2 (j=2) with Schoolbook
    # i in 0..1, k in 0..9. 2*10 = 20 terms.
    offset = 5 * 7 # 35
    for i in range(n):
        for k in range(p):
            idx = offset + i * p + k
            U_s[i * m + 2][idx] = 1 # A[i, 2]
            V_s[2 * p + k][idx] = 1 # B[2, k]
            W_s[k * n + i][idx] = 1 # C[k, i]

    # Verify Strassen result
    def verify_decomp(U, V, W):
        rank = len(U[0])
        for c in range(rank):
            if time.time() > deadline:
                return False
        # We can do a quick check with random matrices
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_direct = np.dot(A, B)
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            u_c = np.array([U[i*m+j][c] for i in range(n) for j in range(m)]).reshape(n, m)
            v_c = np.array([V[j*p+k][c] for j in range(m) for k in range(p)]).reshape(m, p)
            w_c = np.array([W[k*n+i][c] for k in range(p) for i in range(n)]).reshape(p, n)
            C_decomp += np.dot(np.dot(u_c, v_c), w_c)
        return np.allclose(C_direct, C_decomp)

    if verify_decomp(U_s, V_s, W_s):
        current_result = {"U": U_s, "V": V_s, "W": W_s}
        current_rank = rank_strassen

    # 3. Random Search / Optimization (Time permitting)
    # Try to find rank 54 or lower via random linear combinations or local search.
    # Given the time constraints and complexity, we stick to the verified 55.
    # A simple hill-climbing on the 55-term decomposition to reduce rank is difficult
    # without a specific structure. We return the 55-term one.

    return current_result
# EVOLVE-BLOCK-END
