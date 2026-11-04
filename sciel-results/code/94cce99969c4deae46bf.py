import numpy as np
import time

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are (4, 7).
    """
    # The 7 terms
    # 1. (a11+a22)(b11+b22)
    # 2. (a21+a22)b11
    # 3. a11(b12-b22)
    # 4. a22(b21-b11)
    # 5. (a11+a12)b22
    # 6. (a21-a11)(b11+b12)
    # 7. (a12-a22)(b21+b22)
    
    # Indices for A (4 elements): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (4 elements): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (4 elements): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Builds U, V, W for 4x4x4 matrix multiplication using Strassen blocks.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    u2, v2, w2 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The 7 outer terms
    # Each term is a product of two 2x2 matrices.
    # We need to define the linear combinations for the 2x2 blocks.
    
    # A blocks: A11 (0-3), A12 (4-7), A21 (8-11), A22 (12-15)
    # B blocks: B11 (0-3), B12 (4-7), B21 (8-11), B22 (12-15)
    # C blocks: C11 (0-3), C12 (4-7), C21 (8-11), C22 (12-15)
    
    # Term 1: (A11+A22)(B11+B22)
    # A part: A11 + A22
    # B part: B11 + B22
    # C part: C11, C12, C21, C22 all get contribution?
    # No, the result of (A11+A22)(B11+B22) is a 2x2 matrix.
    # In Strassen, M1 contributes to all 4 blocks of C?
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # So for Term 1 (M1):
    # U: A11 + A22
    # V: B11 + B22
    # W: C11, C12, C21, C22 (all 1s)
    
    # We have 7 such terms.
    # For each term t (0..6), we have a 2x2x2 decomposition.
    # The U for term t is a 4x7 matrix.
    # We need to map the 4 elements of the 2x2 block to the 16 elements of the 4x4 matrix.
    
    # Let's define the block mappings.
    # A blocks: 0: A11, 1: A12, 2: A21, 3: A22
    # B blocks: 0: B11, 1: B12, 2: B21, 3: B22
    # C blocks: 0: C11, 1: C12, 2: C21, 3: C22
    
    # For each of the 7 Strassen terms, we define which A blocks and B blocks are used, and their signs.
    # And which C blocks they contribute to, and with what signs.
    
    # Term 0 (M1): A11+A22, B11+B22. C: C11(1), C12(0), C21(0), C22(1)? No.
    # M1 contributes to C11(1), C22(1).
    # Actually, the formula is:
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # So M1 contributes to C11 with +1, C22 with +1.
    # M2 contributes to C21 with +1, C22 with -1.
    # M3 contributes to C12 with +1, C22 with +1.
    # M4 contributes to C11 with -1, C21 with +1.
    # M5 contributes to C11 with -1, C12 with +1.
    # M6 contributes to C22 with -1.
    # M7 contributes to C11 with +1.
    
    # Let's list the (A_blocks, B_blocks, C_blocks, signs) for each term.
    # A_blocks: list of (block_id, sign)
    # B_blocks: list of (block_id, sign)
    # C_blocks: list of (block_id, sign)
    
    terms = [
        # M1
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        # M2
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)]
        },
        # M3
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)]
        },
        # M4
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, -1), (2, 1)]
        },
        # M5
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)]
        },
        # M6
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, -1)]
        },
        # M7
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1)]
        }
    ]
    
    for t_idx, term in enumerate(terms):
        # Get the base Strassen U, V, W for this term
        # u2, v2, w2 are (4, 7)
        # We need to expand them to (16, 7) for this term
        
        U_term = np.zeros((16, 7))
        V_term = np.zeros((16, 7))
        W_term = np.zeros((16, 7))
        
        # Fill U_term
        for a_block, sign in term['A']:
            # a_block is 0..3, corresponds to rows 4*a_block to 4*a_block+3
            start = 4 * a_block
            U_term[start:start+4, :] = sign * u2
            
        # Fill V_term
        for b_block, sign in term['B']:
            start = 4 * b_block
            V_term[start:start+4, :] = sign * v2
            
        # Fill W_term
        for c_block, sign in term['C']:
            start = 4 * c_block
            W_term[start:start+4, :] = sign * w2
            
        # Append to global U, V, W
        col_start = t_idx * 7
        U[:, col_start:col_start+7] = U_term
        V[:, col_start:col_start+7] = V_term
        W[:, col_start:col_start+7] = W_term
        
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_4x5x6():
    """
    Constructs the 4x5x6 decomposition.
    """
    # Part 1: 4x4x4 (Strassen, rank 49)
    # A[:, 0:4], B[0:4, 0:4], C[0:4, 0:4]
    U1, V1, W1 = _build_4x4x4_strassen()
    
    # Part 2: 4x4x2 (Schoolbook, rank 32)
    # A[:, 0:4], B[0:4, 4:6], C[0:4, 4:6]
    # We need to construct this.
    # n=4, m=4, p=2
    U2_local, V2_local, W2_local = _build_schoolbook(4, 4, 2)
    # Map to global indices
    # A: i in 0..3, j in 0..3 -> global i*m+j (m=5)
    # B: j in 0..3, k in 4..5 -> global j*p+k (p=6)
    # C: k in 4..5, i in 0..3 -> global k*n+i (n=4)
    
    U2 = np.zeros((20, 32))
    V2 = np.zeros((30, 32))
    W2 = np.zeros((24, 32))
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                col = i * 4 * 2 + j * 2 + k # Local col
                # Global indices
                a_idx = i * 5 + j
                b_idx = j * 6 + (4 + k)
                c_idx = (4 + k) * 4 + i
                
                U2[a_idx, col] = U2_local[i * 4 + j, col]
                V2[b_idx, col] = V2_local[j * 2 + k, col]
                W2[c_idx, col] = W2_local[k * 4 + i, col]
                
    # Part 3: 4x1x6 (Schoolbook, rank 24)
    # A[:, 4], B[4, 0:6], C[0:4, 0:6]
    # n=4, m=1, p=6
    U3_local, V3_local, W3_local = _build_schoolbook(4, 1, 6)
    
    U3 = np.zeros((20, 24))
    V3 = np.zeros((30, 24))
    W3 = np.zeros((24, 24))
    
    for i in range(4):
        for k in range(6):
            col = i * 6 + k # Local col (m=1, so j=0)
            # Global indices
            a_idx = i * 5 + 4
            b_idx = 4 * 6 + k
            c_idx = k * 4 + i
            
            U3[a_idx, col] = U3_local[i, col]
            V3[b_idx, col] = V3_local[k, col]
            W3[c_idx, col] = W3_local[k * 4 + i, col]
            
    # Concatenate
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    U, V, W = _build_4x5x6()
    
    # Verify
    if _verify_decomposition(U, V, W, 4, 5, 6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(4, 5, 6)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

