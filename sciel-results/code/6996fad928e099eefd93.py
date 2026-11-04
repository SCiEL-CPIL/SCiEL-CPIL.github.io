import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    """
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    
    # Check support
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                if abs(T[idx_u, idx_v, idx_w] - 1.0) > 1e-9:
                    return False
    
    # Check zeros
    valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                valid_mask[idx_u, idx_v, idx_w] = True
                
    T_check = T.copy()
    T_check[valid_mask] = 0.0
    
    if np.max(np.abs(T_check)) > 1e-9:
        return False
        
    return True

def _get_2x2_strassen():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U, V, W have shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 (0), A11 (3)
    U[0, 0] = 1; U[3, 0] = 1
    # V: B00 (0), B11 (3)
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00 (0), C11 (3)
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # U: A10 (2), A11 (3)
    U[2, 1] = 1; U[3, 1] = 1
    # V: B00 (0)
    V[0, 1] = 1
    # W: C10 (2)
    W[2, 1] = 1
    
    # M3 = A00(B10 - B11)
    # U: A00 (0)
    U[0, 2] = 1
    # V: B10 (2), B11 (3)
    V[2, 2] = 1; V[3, 2] = -1
    # W: C01 (1)
    W[1, 2] = 1
    
    # M4 = A11(B01 - B00)
    # U: A11 (3)
    U[3, 3] = 1
    # V: B01 (1), B00 (0)
    V[1, 3] = 1; V[0, 3] = -1
    # W: C11 (3)
    W[3, 3] = 1
    
    # M5 = (A00 + A01)B11
    # U: A00 (0), A01 (1)
    U[0, 4] = 1; U[1, 4] = 1
    # V: B11 (3)
    V[3, 4] = 1
    # W: C01 (1)
    W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10 (2), A00 (0)
    U[2, 5] = 1; U[0, 5] = -1
    # V: B00 (0), B01 (1)
    V[0, 5] = 1; V[1, 5] = 1
    # W: C11 (3)
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01 (1), A11 (3)
    U[1, 6] = 1; U[3, 6] = -1
    # V: B10 (2), B11 (3)
    V[2, 6] = 1; V[3, 6] = 1
    # W: C00 (0)
    W[0, 6] = 1
    
    return U, V, W

def _get_4x4_strassen():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition (rank 49).
    U, V, W have shape (16, 49).
    """
    n, m, p = 4, 4, 4
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # We apply Strassen 2x2x2 to the 4x4x4 problem.
    # The 4x4x4 problem is composed of 4 blocks of 2x2x2.
    # Actually, we treat the 2x2 blocks of A and B as "scalars".
    # There are 7 Strassen multiplications of these "scalars".
    # Each "scalar" multiplication is a 2x2x2 multiplication.
    # So we iterate over the 7 Strassen meta-terms.
    
    # The 7 Strassen meta-terms define linear combinations of the 4 2x2 blocks of A and B.
    # A blocks: A00 (top-left), A01 (top-right), A10 (bot-left), A11 (bot-right).
    # Indices in 4x4 A:
    # A00: rows 0-1, cols 0-1. Flat idx: 0, 1, 4, 5.
    # A01: rows 0-1, cols 2-3. Flat idx: 2, 3, 6, 7.
    # A10: rows 2-3, cols 0-1. Flat idx: 8, 9, 12, 13.
    # A11: rows 2-3, cols 2-3. Flat idx: 10, 11, 14, 15.
    
    # B blocks: B00, B01, B10, B11.
    # B00: rows 0-1, cols 0-1. Flat idx: 0, 1, 4, 5.
    # B01: rows 0-1, cols 2-3. Flat idx: 2, 3, 6, 7.
    # B10: rows 2-3, cols 0-1. Flat idx: 8, 9, 12, 13.
    # B11: rows 2-3, cols 2-3. Flat idx: 10, 11, 14, 15.
    
    # C blocks: C00, C01, C10, C11.
    # C00: rows 0-1, cols 0-1. Flat idx: 0, 1, 4, 5.
    # C01: rows 0-1, cols 2-3. Flat idx: 2, 3, 6, 7.
    # C10: rows 2-3, cols 0-1. Flat idx: 8, 9, 12, 13.
    # C11: rows 2-3, cols 2-3. Flat idx: 10, 11, 14, 15.
    
    # We need to map the 2x2 Strassen U, V, W to the 4x4 indices.
    # For a given meta-term t (0..6), we have:
    # U_t is a linear combination of A blocks.
    # V_t is a linear combination of B blocks.
    # W_t is a linear combination of C blocks.
    
    # Let's define the coefficients for the 7 meta-terms.
    # A coefficients (for U):
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    A_coeffs = [
        {0: 1, 3: 1}, # M1: A00, A11
        {2: 1, 3: 1}, # M2: A10, A11
        {0: 1},       # M3: A00
        {3: 1},       # M4: A11
        {0: 1, 1: 1}, # M5: A00, A01
        {2: 1, 0: -1},# M6: A10, -A00
        {1: 1, 3: -1} # M7: A01, -A11
    ]
    
    # B coefficients (for V):
    # M1: B00 + B11
    # M2: B00
    # M3: B10 - B11
    # M4: B01 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    B_coeffs = [
        {0: 1, 3: 1}, # M1: B00, B11
        {0: 1},       # M2: B00
        {2: 1, 3: -1},# M3: B10, -B11
        {1: 1, 0: -1},# M4: B01, -B00
        {3: 1},       # M5: B11
        {0: 1, 1: 1}, # M6: B00, B01
        {2: 1, 3: 1}  # M7: B10, B11
    ]
    
    # C coefficients (for W):
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # We need to invert this to find W coefficients for each M.
    # W_M1: C00 += 1, C11 += 1
    # W_M2: C10 += 1
    # W_M3: C01 += 1, C11 -= 1
    # W_M4: C00 -= 1, C10 += 1
    # W_M5: C00 += 1, C01 += 1
    # W_M6: C11 += 1
    # W_M7: C00 -= 1, C11 -= 1
    
    C_coeffs = [
        {0: 1, 3: 1},   # M1
        {2: 1},         # M2
        {1: 1, 3: -1},  # M3
        {0: -1, 2: 1},  # M4
        {0: 1, 1: 1},   # M5
        {3: 1},         # M6
        {0: -1, 3: -1}  # M7
    ]
    
    # Map block index to 4x4 flat indices
    # Block 0 (00): 0, 1, 4, 5
    # Block 1 (01): 2, 3, 6, 7
    # Block 2 (10): 8, 9, 12, 13
    # Block 3 (11): 10, 11, 14, 15
    block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Get 2x2 Strassen U, V, W
    U2, V2, W2 = _get_2x2_strassen()
    
    # Construct 4x4x4
    for t in range(7):
        # For each meta-term t, we add 7 terms from the 2x2 Strassen decomposition
        for s in range(7):
            idx = t * 7 + s
            
            # U contribution
            # U2[s, :] is the 2x2 U vector for sub-term s.
            # It has 4 non-zero entries (or fewer).
            # We need to expand this to 4x4.
            # U2 indices 0,1,2,3 correspond to A blocks 0,1,2,3?
            # No, U2 is for 2x2 A. Indices 0..3 are flat indices of 2x2 A.
            # Flat idx 0 -> A00 (block 0, pos 0)
            # Flat idx 1 -> A01 (block 1, pos 0)
            # Flat idx 2 -> A10 (block 2, pos 0)
            # Flat idx 3 -> A11 (block 3, pos 0)
            # Wait, 2x2 flat indices:
            # 0: (0,0) -> Block 0, pos 0
            # 1: (0,1) -> Block 1, pos 0
            # 2: (1,0) -> Block 2, pos 0
            # 3: (1,1) -> Block 3, pos 0
            
            # But U2 can have non-zeros at other positions?
            # In 2x2 Strassen, U vectors are sparse.
            # Let's just iterate over the 4 entries of U2[:, s].
            for local_idx in range(4):
                val = U2[local_idx, s]
                if val != 0:
                    # Map local_idx to block
                    # local_idx 0 -> Block 0, pos 0
                    # local_idx 1 -> Block 1, pos 0
                    # local_idx 2 -> Block 2, pos 0
                    # local_idx 3 -> Block 3, pos 0
                    # This mapping assumes U2 only has non-zeros at "pos 0" of blocks?
                    # No, U2 is a vector of length 4.
                    # The 2x2 Strassen U vectors are:
                    # M1: [1, 0, 0, 1] (A00, A11)
                    # M2: [0, 0, 1, 1] (A10, A11)
                    # M3: [1, 0, 0, 0] (A00)
                    # M4: [0, 0, 0, 1] (A11)
                    # M5: [1, 1, 0, 0] (A00, A01)
                    # M6: [1, 0, 1, 0] with signs? (A10 - A00) -> [1, 0, 1, 0] is A00+A10.
                    # My _get_2x2_strassen defined:
                    # M6: U[2, 5]=1, U[0, 5]=-1. So [ -1, 0, 1, 0 ].
                    
                    # So local_idx corresponds to the 2x2 flat index.
                    # 2x2 flat index 0 -> 4x4 Block 0, local pos 0.
                    # 2x2 flat index 1 -> 4x4 Block 1, local pos 0.
                    # 2x2 flat index 2 -> 4x4 Block 2, local pos 0.
                    # 2x2 flat index 3 -> 4x4 Block 3, local pos 0.
                    
                    # But what if U2 has non-zero at local_idx 1?
                    # local_idx 1 is (0,1) in 2x2.
                    # This corresponds to Block 1, local pos 0 (top-left of block 1).
                    
                    block = local_idx
                    pos_in_block = 0 # Since 2x2 flat index maps to top-left of the block?
                    # No. 2x2 flat index 1 is (0,1).
                    # In 4x4, Block 1 is top-right.
                    # (0,1) in 2x2 A corresponds to (0, 2) in 4x4 A?
                    # 4x4 A:
                    # 0 1 2 3
                    # 4 5 6 7
                    # 8 9 10 11
                    # 12 13 14 15
                    
                    # 2x2 A:
                    # 0 1
                    # 2 3
                    
                    # Mapping:
                    # 2x2 (0,0) -> 4x4 (0,0) -> idx 0. Block 0, pos 0.
                    # 2x2 (0,1) -> 4x4 (0,2) -> idx 2. Block 1, pos 0.
                    # 2x2 (1,0) -> 4x4 (2,0) -> idx 8. Block 2, pos 0.
                    # 2x2 (1,1) -> 4x4 (2,2) -> idx 10. Block 3, pos 0.
                    
                    # So local_idx maps to:
                    # 0 -> idx 0
                    # 1 -> idx 2
                    # 2 -> idx 8
                    # 3 -> idx 10
                    
                    map_2x2_to_4x4 = [0, 2, 8, 10]
                    global_idx = map_2x2_to_4x4[local_idx]
                    
                    # Now we need to apply the coefficient from A_coeffs[t] for this block.
                    # block = local_idx
                    coeff = A_coeffs[t].get(block, 0)
                    
                    if coeff != 0:
                        U[global_idx, idx] += val * coeff
            
            # V contribution
            # Similar logic for V.
            # 2x2 V flat indices map to B blocks.
            # 0 -> B00 (idx 0)
            # 1 -> B01 (idx 2)
            # 2 -> B10 (idx 8)
            # 3 -> B11 (idx 10)
            
            for local_idx in range(4):
                val = V2[local_idx, s]
                if val != 0:
                    block = local_idx
                    coeff = B_coeffs[t].get(block, 0)
                    if coeff != 0:
                        map_2x2_to_4x4 = [0, 2, 8, 10]
                        global_idx = map_2x2_to_4x4[local_idx]
                        V[global_idx, idx] += val * coeff
            
            # W contribution
            # 2x2 W flat indices map to C blocks.
            # 0 -> C00 (idx 0)
            # 1 -> C01 (idx 2)
            # 2 -> C10 (idx 8)
            # 3 -> C11 (idx 10)
            
            for local_idx in range(4):
                val = W2[local_idx, s]
                if val != 0:
                    block = local_idx
                    coeff = C_coeffs[t].get(block, 0)
                    if coeff != 0:
                        map_2x2_to_4x4 = [0, 2, 8, 10]
                        global_idx = map_2x2_to_4x4[local_idx]
                        W[global_idx, idx] += val * coeff
                        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Get 4x4x4 Strassen decomposition
    U444, V444, W444 = _get_4x4_strassen()
    
    # Construct 4x4x9 decomposition
    # Rank = 49 (cols 0-3) + 49 (cols 4-7) + 16 (col 8) = 114
    rank = 114
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Block 1: Cols 0-3
    # U: Copy U444
    U[:, 0:49] = U444
    
    # V: Map 4x4x4 V (16x49) to 4x4x9 V (36x49)
    # 4x4x4 V indices: j*4 + k (k=0..3)
    # 4x4x9 V indices: j*9 + k (k=0..3)
    for j in range(4):
        for k in range(4):
            v_idx_444 = j * 4 + k
            v_idx_449 = j * 9 + k
            V[v_idx_449, 0:49] = V444[v_idx_444, 0:49]
            
    # W: Map 4x4x4 W (16x49) to 4x4x9 W (36x49)
    # 4x4x4 W indices: k*4 + i (k=0..3)
    # 4x4x9 W indices: k*4 + i (k=0..3) -- Wait, W index is k*n + i.
    # n=4. So k*4 + i is the same.
    W[0:16, 0:49] = W444
    
    # Block 2: Cols 4-7
    # U: Copy U444
    U[:, 49:98] = U444
    
    # V: Map 4x4x4 V to cols 4-7 of 4x4x9
    # 4x4x4 V indices: j*4 + k (k=0..3)
    # 4x4x9 V indices: j*9 + (k+4)
    for j in range(4):
        for k in range(4):
            v_idx_444 = j * 4 + k
            v_idx_449 = j * 9 + (k + 4)
            V[v_idx_449, 49:98] = V444[v_idx_444, 0:49]
            
    # W: Map 4x4x4 W to cols 4-7 of 4x4x9
    # 4x4x4 W indices: k*4 + i (k=0..3)
    # 4x4x9 W indices: (k+4)*4 + i
    for k in range(4):
        for i in range(4):
            w_idx_444 = k * 4 + i
            w_idx_449 = (k + 4) * 4 + i
            W[w_idx_449, 49:98] = W444[w_idx_444, 0:49]
            
    # Block 3: Col 8
    # Schoolbook for 4x4x1
    # U: e_{i*4+j}
    # V: e_{j*9+8}
    # W: e_{8*4+i}
    
    for i in range(4):
        for j in range(4):
            idx = 98 + i * 4 + j
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            
            U[u_idx, idx] = 1
            V[v_idx, idx] = 1
            W[w_idx, idx] = 1
            
    res = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    if _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][c] = 1
                    V_school[j * p + k][c] = 1
                    W_school[k * n + i][c] = 1
                    c += 1
        return {"U": U_school, "V": V_school, "W": W_school}
