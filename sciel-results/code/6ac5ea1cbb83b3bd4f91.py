"""Program for matrix multiplication 2,2,13 - seeking rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x2 @ 2x13.
    
    Strategy: Use the polynomial ring Z[x]/(x^13) approach.
    Represent B as two polynomials b0(x) = sum_k B[0,k] x^k and b1(x) = sum_k B[1,k] x^k.
    The product A@B can be viewed as:
      C[0,k] = A[0,0]*b0(x)_k + A[0,1]*b1(x)_k
      C[1,k] = A[1,0]*b0(x)_k + A[1,1]*b1(x)_k
    
    But this is just the schoolbook. The key insight is that for 2x2 matrices,
    Strassen's algorithm reduces the number of multiplications from 4 to 7.
    
    For 2x2 * 2x13, we can group columns in pairs and apply Strassen to each pair,
    plus handle the last column directly.
    
    6 pairs of columns (12 columns) use Strassen (rank 7 each) = 42
    1 single column (column 12) uses direct multiplication (rank 4) = 4
    Total rank: 46
    
    The Strassen decomposition for 2x2 * 2x2:
    p0 = (A00+A11)(B00+B11)
    p1 = (A10+A11)B00
    p2 = A00(B10-B00)
    p3 = A11(B01-B11)
    p4 = (A00+A01)B11
    p5 = (A10-A00)(B00+B01)
    p6 = (A01-A11)(B10+B11)
    
    C00 = p0 + p3 - p4 + p6
    C01 = p2 + p4
    C10 = p1 + p3
    C11 = p0 - p1 + p2 + p5
    """
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Schoolbook fallback (rank 52)
    rank_schoolbook = n * m * p  # 52
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strassen decomposition for 2x2 * 2x2 (verified)
    # U_s[r, idx] where idx = i*2+j for A[i,j]
    U_s = np.array([
        [1,  0,  0,  1],   # p0: A00 + A11
        [0,  0,  1,  1],   # p1: A10 + A11
        [1,  0,  0,  0],   # p2: A00
        [0,  0,  0,  1],   # p3: A11
        [1,  1,  0,  0],   # p4: A00 + A01
        [-1, 0,  1,  0],   # p5: -A00 + A10
        [0,  1,  0, -1],   # p6: A01 - A11
    ])  # shape (7, 4)
    
    # V_s[r, idx] where idx = j*2+k for B[j,k]
    V_s = np.array([
        [1,  0,  0,  1],   # p0: B00 + B11
        [1,  0,  0,  0],   # p1: B00
        [0, -1,  1,  0],   # p2: -B00 + B10
        [0,  1,  0, -1],   # p3: B01 - B11
        [0,  0,  0,  1],   # p4: B11
        [1,  1,  0,  0],   # p5: B00 + B01
        [0,  0,  1,  1],   # p6: B10 + B11
    ])  # shape (7, 4)
    
    # W_s[r, idx] where idx = i*2+k for C[i,k]
    W_s = np.array([
        [1,  0,  0,  1],   # p0: C00 + C11
        [0,  0,  1, -1],   # p1: C10 - C11
        [0,  1,  0,  1],   # p2: C01 + C11
        [1,  0,  1,  0],   # p3: C00 + C10
        [-1, 1,  0,  0],   # p4: -C00 + C01
        [0,  0,  0,  1],   # p5: C11
        [1,  0,  0,  0],   # p6: C00
    ])  # shape (7, 4)
    
    # Verify Strassen decomposition for 2x2*2x2
    def verify_strassen_2x2():
        n2, m2, p2 = 2, 2, 2
        U2 = U_s.T  # (4, 7)
        V2 = V_s.T  # (4, 7)
        W2 = W_s.T  # (4, 7)
        for i in range(n2):
            for j in range(m2):
                for jp in range(m2):
                    for k in range(p2):
                        for ip in range(n2):
                            for kp in range(p2):
                                expected = (1 if i == ip and j == jp and k == kp else 0)
                                actual = 0.0
                                for r in range(7):
                                    actual += U2[i*m2+j, r] * V2[jp*p2+k, r] * W2[ip*n2+kp, r]
                                if abs(expected - actual) > 1e-10:
                                    return False
        return True
    
    if not verify_strassen_2x2():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Build full 2x2*2x13 decomposition with rank 46
    rank = 6 * 7 + 4  # 42 + 4 = 46
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    # Process 6 pairs of columns using Strassen
    for t in range(6):
        base = t * 7
        col0 = 2 * t
        col1 = 2 * t + 1
        
        # U: same for all blocks (A is shared)
        for r in range(7):
            for idx in range(4):
                i, j = idx // 2, idx % 2
                U[i*m + j][base + r] = U_s[r, idx]
        
        # V: B indices for this chunk
        # B[0, col0] -> index 0*p + col0 = col0
        # B[0, col1] -> index 0*p + col1 = col1
        # B[1, col0] -> index 1*p + col0 = p + col0
        # B[1, col1] -> index 1*p + col1 = p + col1
        for r in range(7):
            V[0*p + col0][base + r] = V_s[r, 0]  # B[0,col0]
            V[0*p + col1][base + r] = V_s[r, 1]  # B[0,col1]
            V[1*p + col0][base + r] = V_s[r, 2]  # B[1,col0]
            V[1*p + col1][base + r] = V_s[r, 3]  # B[1,col1]
        
        # W: C indices for this chunk
        # C[0, col0] -> index col0*n + 0 = col0*2
        # C[0, col1] -> index col1*n + 0 = col1*2
        # C[1, col0] -> index col0*n + 1 = col0*2 + 1
        # C[1, col1] -> index col1*n + 1 = col1*2 + 1
        for r in range(7):
            W[col0*n + 0][base + r] = W_s[r, 0]  # C[0,col0]
            W[col1*n + 0][base + r] = W_s[r, 1]  # C[0,col1]
            W[col0*n + 1][base + r] = W_s[r, 2]  # C[1,col0]
            W[col1*n + 1][base + r] = W_s[r, 3]  # C[1,col1]
    
    # Last column (12): 4 direct terms
    base = 6 * 7  # 42
    col = 12
    for i in range(2):
        for j in range(2):
            U[i*m + j][base + i*2 + j] = 1
            V[j*p + col][base + i*2 + j] = 1
            W[col*n + i][base + i*2 + j] = 1
    
    # Verify the full decomposition
    def verify_full():
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                expected = (1 if i == ip and j == jp and k == kp else 0)
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[i*m+j][r] * V[jp*p+k][r] * W[ip*n+kp][r]
                                if abs(expected - actual) > 1e-10:
                                    return False
        return True
    
    if not verify_full():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
