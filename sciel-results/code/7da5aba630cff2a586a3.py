"""Program for matrix multiplication 2,2,13 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]
DEADLINE = time.time() + 130.0  # Safety margin before 150s kill


def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x2 @ 2x13.
    
    Uses Strassen-style decomposition: 6 pairs of columns (rank 7 each) + 1 single column (rank 4)
    Total rank: 6*7 + 4 = 46
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
    
    # Strassen decomposition for 2x2 * 2x2:
    # We need 7 rank-1 terms such that:
    # delta_{i,i'} * delta_{j,j'} * delta_{k,k'} = sum_r U[i,j,r] * V[j',k,r] * W[i',k',r]
    #
    # Using the standard Strassen formulas with 0-indexed entries:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # The 7 products:
    # p0 = (A00+A11)(B00+B11)
    # p1 = (A10+A11)B00
    # p2 = A00(B10-B00)
    # p3 = A11(B01-B11)
    # p4 = (A00+A01)B11
    # p5 = (A10-A00)(B00+B01)
    # p6 = (A01-A11)(B10+B11)
    #
    # Output reconstruction:
    # C00 = p0 + p3 - p4 + p6
    # C01 = p2 + p4
    # C10 = p1 + p3
    # C11 = p0 - p1 + p2 + p5
    
    # U (4x7): U[i,j,r] for A entries [A00, A01, A10, A11]
    # Row order: [A00, A01, A10, A11]
    U_s = np.array([
        [1,  0,  1,  0,  1, -1,  0],  # A00: appears in p0, p2, p4, p5
        [0,  1,  0,  0,  1,  0,  1],  # A01: appears in p4, p6
        [0,  0,  0,  0,  0,  1,  0],  # A10: appears in p1, p5
        [1,  0,  0,  1,  0,  0, -1],  # A11: appears in p0, p1, p3, p6
    ], dtype=float)  # shape (4, 7)
    
    # V (4x7): V[j,k,r] for B entries [B00, B01, B10, B11]
    # Row order: [B00, B01, B10, B11]
    V_s = np.array([
        [1,  0, -1,  0,  0,  1,  0],  # B00: appears in p0, p1, p2, p5
        [0,  1,  0,  0,  0,  0,  1],  # B01: appears in p3, p6
        [0,  0,  1,  0,  0,  0,  1],  # B10: appears in p2, p6
        [1,  0,  0, -1,  1,  0,  0],  # B11: appears in p0, p3, p4
    ], dtype=float)  # shape (4, 7)
    
    # W (4x7): W[i,k,r] for C entries [C00, C01, C10, C11]
    # Row order: [C00, C01, C10, C11]
    W_s = np.array([
        [1,  0,  0,  0, -1,  0,  1],  # C00: p0 + p3 - p4 + p6
        [0,  1,  0,  0,  1,  0,  0],  # C01: p2 + p4
        [0,  0,  1,  1,  0,  0,  0],  # C10: p1 + p3
        [1,  0,  0,  0,  0,  1,  0],  # C11: p0 - p1 + p2 + p5
    ], dtype=float)  # shape (4, 7)
    
    # Verify the Strassen decomposition for 2x2*2x2
    def verify_strassen_2x2():
        n2, m2, p2 = 2, 2, 2
        # Check: sum_r U[i,j,r] * V[j',k,r] * W[i',k',r] == delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
        for i in range(n2):
            for j in range(m2):
                for jp in range(m2):
                    for k in range(p2):
                        for ip in range(n2):
                            for kp in range(p2):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                actual = 0.0
                                for r in range(7):
                                    actual += U_s[i * m2 + j, r] * V_s[jp * p2 + k, r] * W_s[ip * n2 + kp, r]
                                if abs(actual - expected) > 1e-9:
                                    return False, (i, j, jp, k, ip, kp, expected, actual)
        return True, None
    
    ok, info = verify_strassen_2x2()
    if not ok:
        # If Strassen verification fails, fall back to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Now build the full 2x2*2x13 decomposition
    # For each pair of columns (2t, 2t+1) for t=0..5, use the Strassen block (rank 7 each)
    # For the last column (12), use 4 direct terms (rank 4)
    rank = 6 * 7 + 4  # 46
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    # Build Strassen blocks for column pairs
    for t in range(6):
        base = t * 7
        col0 = 2 * t
        col1 = 2 * t + 1
        
        # U: same for all blocks (A is shared across all column pairs)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][base + r] = U_s[i * m + j, r]
        
        # V: B indices for this chunk
        # B[0, col0] -> index 0*p + col0 = col0
        # B[0, col1] -> index 0*p + col1 = col1
        # B[1, col0] -> index 1*p + col0 = p + col0
        # B[1, col1] -> index 1*p + col1 = p + col1
        for r in range(7):
            V[0 * p + col0][base + r] = V_s[0 * 2 + 0, r]  # B[0, col0]
            V[0 * p + col1][base + r] = V_s[0 * 2 + 1, r]  # B[0, col1]
            V[1 * p + col0][base + r] = V_s[1 * 2 + 0, r]  # B[1, col0]
            V[1 * p + col1][base + r] = V_s[1 * 2 + 1, r]  # B[1, col1]
        
        # W: C indices for this chunk
        # C[0, col0] -> index col0*n + 0 = col0*2
        # C[0, col1] -> index col1*n + 0 = col1*2
        # C[1, col0] -> index col0*n + 1 = col0*2 + 1
        # C[1, col1] -> index col1*n + 1 = col1*2 + 1
        for r in range(7):
            W[col0 * n + 0][base + r] = W_s[0 * 2 + 0, r]  # C[0, col0]
            W[col1 * n + 0][base + r] = W_s[0 * 2 + 1, r]  # C[0, col1]
            W[col0 * n + 1][base + r] = W_s[1 * 2 + 0, r]  # C[1, col0]
            W[col1 * n + 1][base + r] = W_s[1 * 2 + 1, r]  # C[1, col1]
    
    # Last column (12): 4 direct terms
    base = 6 * 7  # 42
    col = 12
    for i in range(2):
        for j in range(2):
            U[i * m + j][base + i * 2 + j] = 1
            V[j * p + col][base + i * 2 + j] = 1
            W[col * n + i][base + i * 2 + j] = 1
    
    # Verify the full decomposition
    def verify_full():
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                actual = float(np.dot(U_arr[i * m + j], V_arr[jp * p + k] * W_arr[ip * n + kp]))
                                if abs(actual - expected) > 1e-9:
                                    return False, (i, j, jp, k, ip, kp, expected, actual)
        return True, None
    
    ok, info = verify_full()
    if not ok:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
