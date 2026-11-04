import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # A is n x m (4x4)
    # B is m x p (4x5)
    # C is n x p (4x5)
    
    # 1. Strassen decomposition for M_4,4,4 (k=0..3)
    # Rank 49
    # 2. Schoolbook for M_4,4,1 (k=4)
    # Rank 4
    # Total Rank 53
    
    rank = 53
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # --- Part 1: M_4,4,4 using Strassen ---
    # Strassen steps for 4x4 using 2x2 blocks
    # Blocks of A: A00, A01, A10, A11
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Linear forms for L_A and L_B for each Strassen step s=0..6
    # L_A[s] is a 2x2 matrix of linear forms on A
    # Each linear form is a vector of length 16
    
    # Helper to create a block linear form
    # block(p, q) returns a 16-vector with 1s at the positions of block A[p,q]
    def block_A(p, q):
        vec = np.zeros(16)
        for r in range(2):
            for c in range(2):
                idx = (2*p + r) * 4 + (2*q + c)
                vec[idx] = 1.0
        return vec
    
    def block_B(p, q):
        # B is 4x5. Blocks are 2x2.
        # B00: rows 0-1, cols 0-1
        # B01: rows 0-1, cols 2-3
        # B10: rows 2-3, cols 0-1
        # B11: rows 2-3, cols 2-3
        vec = np.zeros(20)
        for r in range(2):
            for c in range(2):
                idx = (2*p + r) * 5 + (2*q + c)
                vec[idx] = 1.0
        return vec

    def block_C(p, q):
        # C is 4x5. Blocks are 2x2.
        # C00: rows 0-1, cols 0-1
        # C01: rows 0-1, cols 2-3
        # C10: rows 2-3, cols 0-1
        # C11: rows 2-3, cols 2-3
        # W index is k*n + i.
        # C[i,k].
        # Block C[p,q] corresponds to rows 2p..2p+1, cols 2q..2q+1.
        vec = np.zeros(20)
        for r in range(2):
            for c in range(2):
                i = 2*p + r
                k = 2*q + c
                idx = k * 4 + i
                vec[idx] = 1.0
        return vec

    # Strassen linear forms for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need the 2x2 matrices L_A and L_B for each step.
    # L_A[s] is 2x2.
    
    LA = []
    LB = []
    
    # Step 1
    LA.append([
        [block_A(0,0) + block_A(1,1), np.zeros(16)],
        [np.zeros(16), block_A(0,0) + block_A(1,1)]
    ])
    # Wait, L_A is a 2x2 matrix.
    # M1 = (A00+A11)(B00+B11)
    # The "matrix" being multiplied is (A00+A11) and (B00+B11).
    # These are 2x2 matrices.
    # So L_A[0] = A00+A11, L_B[0] = B00+B11.
    
    # Let's re-define LA and LB as lists of 2x2 arrays of 16-vectors.
    
    LA = [
        [[block_A(0,0) + block_A(1,1), np.zeros(16)],
         [np.zeros(16), block_A(0,0) + block_A(1,1)]], # A00+A11
        [[np.zeros(16), np.zeros(16)],
         [block_A(1,0) + block_A(1,1), block_A(1,0) + block_A(1,1)]], # A10+A11
        [[block_A(0,0), np.zeros(16)],
         [np.zeros(16), np.zeros(16)]], # A00
        [[np.zeros(16), np.zeros(16)],
         [np.zeros(16), block_A(1,1)]], # A11
        [[block_A(0,0) + block_A(0,1), np.zeros(16)],
         [np.zeros(16), block_A(0,0) + block_A(0,1)]], # A00+A01
        [[block_A(1,0) - block_A(0,0), np.zeros(16)],
         [np.zeros(16), block_A(1,0) - block_A(0,0)]], # A10-A00
        [[np.zeros(16), block_A(0,1) - block_A(1,1)],
         [np.zeros(16), np.zeros(16)]] # A01-A11
    ]
    
    LB = [
        [[block_B(0,0) + block_B(1,1), np.zeros(20)],
         [np.zeros(20), block_B(0,0) + block_B(1,1)]], # B00+B11
        [[block_B(0,0), np.zeros(20)],
         [np.zeros(20), block_B(0,0)]], # B00
        [[np.zeros(20), block_B(0,1) - block_B(1,1)],
         [np.zeros(20), np.zeros(20)]], # B01-B11
        [[np.zeros(20), np.zeros(20)],
         [block_B(1,0) - block_B(0,0), block_B(1,0) - block_B(0,0)]], # B10-B00
        [[np.zeros(20), block_B(1,1)],
         [np.zeros(20), block_B(1,1)]], # B11
        [[block_B(0,0) + block_B(0,1), np.zeros(20)],
         [np.zeros(20), block_B(0,0) + block_B(0,1)]], # B00+B01
        [[block_B(1,0) + block_B(1,1), np.zeros(20)],
         [np.zeros(20), block_B(1,0) + block_B(1,1)]] # B10+B11
    ]
    
    # Strassen 2x2 terms
    # u2, v2, w2 are vectors of length 4 (flattened 2x2)
    # Order: (0,0), (0,1), (1,0), (1,1)
    
    terms_2x2 = [
        # M1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]), # (A00+A11)(B00+B11) -> C00, C01, C10, C11 all get this?
        # No, the output is a 2x2 matrix.
        # C = L_C * M1
        # C00 = M1_00, C01 = M1_01, C10 = M1_10, C11 = M1_11
        # So w2 is [1,1,1,1]
    ]
    
    # Let's list the 7 terms properly.
    # 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,1,1,1]
    # 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] -> M2 = (A10+A11)B00. C10 += M2_00, C11 += M2_10?
    #    M2 is 2x2.
    #    M2 = [[A10_00+A11_00, A10_01+A11_01], [A10_10+A11_10, A10_11+A11_11]] * B00
    #    C10 = M2_00, C11 = M2_10?
    #    Standard Strassen:
    #    C00 = M1 - M4 + M6 + M7
    #    C01 = M3 + M5 + M7
    #    C10 = M2 + M4 + M6
    #    C11 = M1 - M3 + M2 + M5
    
    # So the contributions are:
    # M1: C00 += M1_00, C01 += M1_01, C10 += M1_10, C11 += M1_11
    #      w2 = [1,1,1,1]
    # M2: C10 += M2_00, C11 += M2_10?
    #      M2 = (A10+A11)B00.
    #      M2_00 = (A10_00+A11_00)B00_00
    #      M2_10 = (A10_10+A11_10)B00_00
    #      C10 is 2x2. C10_00 = M2_00, C10_01 = M2_01?
    #      Actually, C10 is a block.
    #      C10 = M2 + M4 + M6.
    #      So C10_00 = M2_00 + M4_00 + M6_00
    #      C10_01 = M2_01 + M4_01 + M6_01
    #      C10_10 = M2_10 + M4_10 + M6_10
    #      C10_11 = M2_11 + M4_11 + M6_11
    #      So for M2, w2 = [0,0,1,1]? No.
    #      M2 contributes to C10 and C11?
    #      C10 = M2 + M4 + M6.
    #      C11 = M1 - M3 + M2 + M5.
    #      So M2 contributes to C10 and C11.
    #      Specifically, C10 += M2, C11 += M2.
    #      So w2 = [0,0,1,1] is wrong.
    #      w2 should be the 2x2 matrix of coefficients.
    #      C10 is block (1,0). C11 is block (1,1).
    #      So w2[2] = 1, w2[3] = 1?
    #      w2 is flattened C.
    #      C00, C01, C10, C11.
    #      M2 contributes to C10 and C11.
    #      So w2 = [0,0,1,1].
    
    # Let's verify M2.
    # M2 = (A10+A11)B00.
    # C10 = M2 + M4 + M6.
    # C11 = M1 - M3 + M2 + M5.
    # Yes, M2 is added to C10 and C11.
    
    # M3 = A00(B01-B11).
    # C01 = M3 + M5 + M7.
    # C11 = M1 - M3 + M2 + M5.
    # So M3 contributes to C01 and C11.
    # w2 = [0,1,0,1].
    
    # M4 = A11(B10-B00).
    # C00 = M1 - M4 + M6 + M7.
    # C10 = M2 + M4 + M6.
    # So M4 contributes to C00 and C10.
    # w2 = [1,0,1,0].
    
    # M5 = (A00+A01)B11.
    # C01 = M3 + M5 + M7.
    # C11 = M1 - M3 + M2 + M5.
    # So M5 contributes to C01 and C11.
    # w2 = [0,1,0,1].
    
    # M6 = (A10-A00)(B00+B01).
    # C00 = M1 - M4 + M6 + M7.
    # C10 = M2 + M4 + M6.
    # So M6 contributes to C00 and C10.
    # w2 = [1,0,1,0].
    
    # M7 = (A01-A11)(B10+B11).
    # C00 = M1 - M4 + M6 + M7.
    # C01 = M3 + M5 + M7.
    # So M7 contributes to C00 and C01.
    # w2 = [1,1,0,0].

    terms_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,1,1,1]), # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,1]), # M2
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]), # M3
        ([0,0,0,1], [0,-1,1,0], [1,0,1,0]), # M4
        ([1,1,0,0], [0,0,0,1], [0,1,0,1]), # M5
        ([1,0,-1,0], [1,1,0,0], [1,0,1,0]), # M6
        ([0,1,0,-1], [0,0,1,1], [1,1,0,0])  # M7
    ]
    
    col = 0
    for s in range(7):
        for u2, v2, w2 in terms_2x2:
            U_vec = np.zeros(16)
            V_vec = np.zeros(20)
            W_vec = np.zeros(20)
            
            for r in range(2):
                for c in range(2):
                    idx = r * 2 + c
                    if u2[idx] != 0:
                        U_vec += u2[idx] * LA[s][r][c]
                    if v2[idx] != 0:
                        V_vec += v2[idx] * LB[s][r][c]
                    if w2[idx] != 0:
                        W_vec += w2[idx] * block_C(r, c)
            
            U[:, col] = U_vec
            V[:, col] = V_vec
            W[:, col] = W_vec
            col += 1
            
    # --- Part 2: M_4,4,1 (k=4) ---
    # C_i4 = sum_j A_ij B_j4
    # 4 terms
    for i in range(4):
        for j in range(4):
            U[:, col] = 0
            V[:, col] = 0
            W[:, col] = 0
            
            U[i * 4 + j] = 1
            V[j * 5 + 4] = 1
            W[4 * 4 + i] = 1 # k=4, i=i -> 16+i
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
