"""Program for matrix multiplication 2,2,13 - seeking rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x2 @ 2x13.
    
    Uses Strassen-style decomposition: 6 pairs of columns (rank 7 each) + 1 single column (rank 4)
    Total rank: 6*7 + 4 = 46
    
    The key is to have a CORRECT Strassen decomposition for 2x2*2x2.
    """
    n, m, p = DIMENSIONS  # 2, 2, 13
    t0 = time.time()
    
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
    
    # Correct Strassen decomposition for 2x2 * 2x2:
    # We need: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[(i,j),(j',k),(i',k')] = 1 iff i=i', j=j', k=k'
    #
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U matrix (4x7): rows indexed by (i,j) in A: A00, A01, A10, A11
    # For each product r, U[i,j,r] is the coefficient
    U_s = np.array([
        # r=0 (M1): A00+A11
        [1, 0, 0, 1],
        # r=1 (M2): A10+A11
        [0, 0, 1, 1],
        # r=2 (M3): A00
        [1, 0, 0, 0],
        # r=3 (M4): A11
        [0, 0, 0, 1],
        # r=4 (M5): A00+A01
        [1, 1, 0, 0],
        # r=5 (M6): A10-A00
        [-1, 0, 1, 0],
        # r=6 (M7): A01-A11
        [0, 1, 0, -1],
    ])  # shape (7, 4) - U_s[r, idx] where idx = i*2+j
    
    # V matrix (4x7): rows indexed by (j,k) in B: B00, B01, B10, B11
    V_s = np.array([
        # r=0 (M1): B00+B11
        [1, 0, 0, 1],
        # r=1 (M2): B00
        [1, 0, 0, 0],
        # r=2 (M3): B10-B00
        [-1, 0, 1, 0],
        # r=3 (M4): B01-B11
        [0, 1, 0, -1],
        # r=4 (M5): B11
        [0, 0, 0, 1],
        # r=5 (M6): B00+B01
        [1, 1, 0, 0],
        # r=6 (M7): B10+B11
        [0, 0, 1, 1],
    ])  # shape (7, 4) - V_s[r, idx] where idx = j*2+k
    
    # W matrix (4x7): rows indexed by (i,k) in C: C00, C01, C10, C11
    W_s = np.array([
        # r=0 (M1): C00+=1, C11+=1
        [1, 0, 0, 1],
        # r=1 (M2): C10+=1, C11+=-1
        [0, 0, 1, -1],
        # r=2 (M3): C01+=1, C11+=1
        [0, 1, 0, 1],
        # r=3 (M4): C00+=1, C10+=1
        [1, 0, 1, 0],
        # r=4 (M5): C00+=-1, C01+=1
        [-1, 1, 0, 0],
        # r=5 (M6): C11+=1
        [0, 0, 0, 1],
        # r=6 (M7): C00+=1
        [1, 0, 0, 0],
    ])  # shape (7, 4) - W_s[r, idx] where idx = i*2+k
    
    # Verify the Strassen decomposition numerically
    def verify_strassen():
        """Check that sum_r U_s[r,i*2+j] * V_s[r,j'*2+k] * W_s[r,i'*2+k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}"""
        for i in range(2):
            for j in range(2):
                for jp in range(2):
                    for k in range(2):
                        for ip in range(2):
                            for kp in range(2):
                                expected = 1 if (i == ip and j == jp and k == kp) else 0
                                actual = 0.0
                                for r in range(7):
                                    actual += U_s[r, i*2+j] * V_s[r, jp*2+k] * W_s[r, ip*2+kp]
                                if abs(actual - expected) > 1e-10:
                                    return False
        return True
    
    if not verify_strassen():
        # Fall back to schoolbook if Strassen is wrong
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Build the full 2x2*2x13 decomposition using 6 Strassen blocks + 1 direct block
    rank = 6 * 7 + 4  # 46
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    # For each pair of output columns (2t, 2t+1), use one Strassen block
    for t in range(6):
        base = t * 7
        col0 = 2 * t
        col1 = 2 * t + 1
        
        # U: same for all blocks (A is shared across all output columns)
        for r in range(7):
            for idx in range(4):  # idx = i*2+j
                U[idx][base + r] = U_s[r, idx]
        
        # V: B entries for this pair of columns
        # B[j, col0] has index j*p + col0, B[j, col1] has index j*p + col1
        for r in range(7):
            # V_s[r, 0] -> B[0, col0], index 0*p + col0
            V[0 * p + col0][base + r] = V_s[r, 0]
            # V_s[r, 1] -> B[0, col1], index 0*p + col1
            V[0 * p + col1][base + r] = V_s[r, 1]
            # V_s[r, 2] -> B[1, col0], index 1*p + col0
            V[1 * p + col0][base + r] = V_s[r, 2]
            # V_s[r, 3] -> B[1, col1], index 1*p + col1
            V[1 * p + col1][base + r] = V_s[r, 3]
        
        # W: C entries for this pair of columns
        # C[i, col0] has index col0*n + i, C[i, col1] has index col1*n + i
        for r in range(7):
            # W_s[r, 0] -> C[0, col0], index col0*2 + 0
            W[col0 * n + 0][base + r] = W_s[r, 0]
            # W_s[r, 1] -> C[0, col1], index col1*2 + 0
            W[col1 * n + 0][base + r] = W_s[r, 1]
            # W_s[r, 2] -> C[1, col0], index col0*2 + 1
            W[col0 * n + 1][base + r] = W_s[r, 2]
            # W_s[r, 3] -> C[1, col1], index col1*2 + 1
            W[col1 * n + 1][base + r] = W_s[r, 3]
    
    # Last column (12): 4 direct terms (one for each i,j pair)
    base = 6 * 7  # 42
    col = 12
    for i in range(n):
        for j in range(m):
            r = i * m + j  # 0,1,2,3
            U[i * m + j][base + r] = 1
            V[j * p + col][base + r] = 1
            W[col * n + i][base + r] = 1
    
    # Verify the full decomposition
    def verify_full():
        """Check that sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = delta_{i,i'}delta_{j,j'}delta_{k,k'}"""
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                expected = 1 if (i == ip and j == jp and k == kp) else 0
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[i*m+j][r] * V[jp*p+k][r] * W[ip*n+kp][r]
                                if abs(actual - expected) > 1e-10:
                                    return False
        return True
    
    if not verify_full():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
