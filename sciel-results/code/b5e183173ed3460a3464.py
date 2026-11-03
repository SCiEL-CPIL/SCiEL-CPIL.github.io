import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> u: [1, 0, 0, 1], v: [1, 0, 0, 1], w: [1, 0, 0, -1] (for C11, C22 parts? No, w is the full 2x2 output contribution)
    # The rank 1 tensor for M1 is u1 v1 w1 where w1 is the matrix [[1, 0], [0, 1]]? No.
    # The product is M1 * I? No.
    # C = sum M_k * P_k where P_k are fixed 2x2 matrices.
    # P1 = [[1, 0], [0, 1]]? No.
    # C11 gets M1, C22 gets -M1.
    # So P1 = [[1, 0], [0, -1]].
    # C12 gets 0, C21 gets 0.
    
    # Let's define the 7 (u, v, P) tuples.
    # u is vector of 4 coeffs for A (row-major: a, b, c, d)
    # v is vector of 4 coeffs for B (row-major: e, f, g, h)
    # P is 2x2 matrix.
    
    strassen_blocks = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([[1, 0], [0, -1]])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([[0, 0], [1, 0]])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([[0, 1], [0, 1]])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 0, 1, -1]), np.array([[0, 0], [1, 0]])), # Wait, C11 gets -M4, C21 gets M4.
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([[1, 1], [0, 0]])),
        # M6
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([[0, 0], [0, 1]])), # C22 gets M6.
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([[-1, 0], [0, 1]])), # C11 gets -M7, C22 gets M7.
    ]
    
    # Verify reconstruction:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = -M1 + M3 + M6 + M7
    
    # Let's check P matrices against these.
    # P1: [[1, 0], [0, -1]] -> C11 += 1, C22 += -1. Correct.
    # P2: [[0, 0], [1, 0]] -> C21 += 1. Correct.
    # P3: [[0, 1], [0, 1]] -> C12 += 1, C22 += 1. Correct.
    # P4: [[0, 0], [1, 0]] -> C21 += 1. But we need C11 += -1.
    # My P4 is wrong. C11 gets -M4. C21 gets M4.
    # So P4 should be [[-1, 0], [1, 0]].
    
    # P5: [[1, 1], [0, 0]] -> C11 += 1, C12 += 1. Correct.
    # P6: [[0, 0], [0, 1]] -> C22 += 1. Correct.
    # P7: [[-1, 0], [0, 1]] -> C11 += -1, C22 += 1. Correct.
    
    # Corrected P4:
    strassen_blocks[3] = (np.array([0, 0, 0, 1]), np.array([0, 0, 1, -1]), np.array([[-1, 0], [1, 0]]))

    col = 0
    # Process A1 (rows 0,1 of A) with B1 (rows 0,1 of B)
    # A1 is 2x2. Indices in U: 0,1,2,3.
    # B1 is 2x6. Split into 3 blocks of 2x2.
    # B1 block l (0,1,2) corresponds to cols 2l, 2l+1 of B1.
    # Rows of B1 are 0,1. So indices in V: 0,1,2,3 for block 0?
    # V is m*p = 4*6 = 24.
    # B is 4x6.
    # B1 is top 2 rows: rows 0,1.
    # B1,0: rows 0,1, cols 0,1. V indices: 0,1,2,3.
    # B1,1: rows 0,1, cols 2,3. V indices: 4,5,6,7.
    # B1,2: rows 0,1, cols 4,5. V indices: 8,9,10,11.
    
    # A1 indices in U: 0,1,2,3.
    # C output for A1*B1,l goes to C[:, 2l:2l+2].
    # C is 2x6. W is p*n = 6*2 = 12.
    # W index for C[i,k] is k*n + i.
    # For block l, cols are 2l, 2l+1.
    # C[0, 2l] -> W[2l*2 + 0] = W[4l]
    # C[0, 2l+1] -> W[(2l+1)*2 + 0] = W[4l+2]
    # C[1, 2l] -> W[2l*2 + 1] = W[4l+1]
    # C[1, 2l+1] -> W[(2l+1)*2 + 1] = W[4l+3]
    
    for l in range(3):
        for u_vec, v_vec, P in strassen_blocks:
            # U part
            U[0:4, col] = u_vec
            # V part
            v_start = 4 * l
            V[v_start:v_start+4, col] = v_vec
            # W part
            w_start = 4 * l
            # P is 2x2. P[0,0] goes to C[0,0] of block -> W[4l+0]
            # P[0,1] goes to C[0,1] of block -> W[4l+2]
            # P[1,0] goes to C[1,0] of block -> W[4l+1]
            # P[1,1] goes to C[1,1] of block -> W[4l+3]
            W[w_start + 0, col] = P[0, 0]
            W[w_start + 2, col] = P[0, 1]
            W[w_start + 1, col] = P[1, 0]
            W[w_start + 3, col] = P[1, 1]
            col += 1
            
    # Process A2 (rows 2,3 of A) with B2 (rows 2,3 of B)
    # A2 indices in U: 4,5,6,7.
    # B2 is bottom 2 rows: rows 2,3.
    # B2,0: rows 2,3, cols 0,1. V indices: 12,13,14,15.
    # B2,1: rows 2,3, cols 2,3. V indices: 16,17,18,19.
    # B2,2: rows 2,3, cols 4,5. V indices: 20,21,22,23.
    
    for l in range(3):
        for u_vec, v_vec, P in strassen_blocks:
            # U part
            U[4:8, col] = u_vec
            # V part
            v_start = 12 + 4 * l
            V[v_start:v_start+4, col] = v_vec
            # W part
            w_start = 4 * l
            W[w_start + 0, col] = P[0, 0]
            W[w_start + 2, col] = P[0, 1]
            W[w_start + 1, col] = P[1, 0]
            W[w_start + 3, col] = P[1, 1]
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
