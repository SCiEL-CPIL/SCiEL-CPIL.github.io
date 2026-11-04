import time
import sys

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """
    Returns U, V, W for schoolbook decomposition of n x m * m x p -> n x p.
    U has shape (n*m, n*m*p)
    V has shape (m*p, n*m*p)
    W has shape (p*n, n*m*p)
    """
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
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    # Split n=4 into 3+1
    # Split m=5 into 3+2
    # Split p=5 into 3+2
    
    n1, n2 = 3, 1
    m1, m2 = 3, 2
    p1, p2 = 3, 2
    
    # Blocks of A (4x5)
    # A11: 3x3, A12: 3x2
    # A21: 1x3, A22: 1x2
    
    # Blocks of B (5x5)
    # B11: 3x3, B12: 3x2
    # B21: 2x3, B22: 2x2
    
    # Products:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We need decompositions for:
    # 1. 3x3 * 3x3 (3x3x3x3)
    # 2. 3x2 * 2x3 (3x2x2x3)
    # 3. 3x3 * 3x2 (3x3x3x2)
    # 4. 3x2 * 2x2 (3x2x2x2)
    # 5. 1x3 * 3x3 (1x3x3x3)
    # 6. 1x2 * 2x3 (1x2x2x3)
    # 7. 1x3 * 3x2 (1x3x3x2)
    # 8. 1x2 * 2x2 (1x2x2x2)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(n_b, m_b, p_b, U_b, V_b, W_b, offset_U, offset_V, offset_W):
        # U_b is n_b*m_b x rank_b
        # V_b is m_b*p_b x rank_b
        # W_b is p_b*n_b x rank_b
        # We need to map indices to the full matrices
        # Full U is 20 x Rank
        # Full V is 25 x Rank
        # Full W is 20 x Rank
        
        rank_b = len(U_b[0]) if U_b else 0
        for i in range(n_b * m_b):
            full_i = offset_U + i
            for r in range(rank_b):
                U_list[full_i][r] += U_b[i][r]
        
        for j in range(m_b * p_b):
            full_j = offset_V + j
            for r in range(rank_b):
                V_list[full_j][r] += V_b[j][r]
        
        for k in range(p_b * n_b):
            full_k = offset_W + k
            for r in range(rank_b):
                W_list[full_k][r] += W_b[k][r]
        
        return rank_b
    
    # Initialize U, V, W with zeros
    # We don't know the total rank yet, so we'll collect them and combine
    # Or we can estimate the total rank and allocate
    
    # Estimate total rank
    # 1. 3x3x3x3: 27
    # 2. 3x2x2x3: 12
    # 3. 3x3x3x2: 18
    # 4. 3x2x2x2: 12
    # 5. 1x3x3x3: 9
    # 6. 1x2x2x3: 6
    # 7. 1x3x3x2: 6
    # 8. 1x2x2x2: 4
    total_rank = 27 + 12 + 18 + 12 + 9 + 6 + 6 + 4
    # total_rank = 94
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # 1. A11*B11: 3x3 * 3x3
    U1, V1, W1 = _schoolbook(3, 3, 3)
    r1 = len(U1[0])
    # Map indices
    # A11 is rows 0-2, cols 0-2 of A.
    # In U (4x5), A11 corresponds to rows 0-2, cols 0-2.
    # U index: i*m + j. i in 0-2, j in 0-2.
    # So U indices 0-8.
    # B11 is rows 0-2, cols 0-2 of B.
    # In V (5x5), B11 corresponds to rows 0-2, cols 0-2.
    # V index: j*p + k. j in 0-2, k in 0-2.
    # So V indices 0-8.
    # C11 is rows 0-2, cols 0-2 of C.
    # In W (5x4), C11 corresponds to cols 0-2, rows 0-2.
    # W index: k*n + i. k in 0-2, i in 0-2.
    # So W indices 0-8.
    
    for i in range(9):
        for r in range(r1):
            U[i][col_offset + r] = U1[i][r]
    for j in range(9):
        for r in range(r1):
            V[j][col_offset + r] = V1[j][r]
    for k in range(9):
        for r in range(r1):
            W[k][col_offset + r] = W1[k][r]
    col_offset += r1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 2. A12*B21: 3x2 * 2x3
    U2, V2, W2 = _schoolbook(3, 2, 3)
    r2 = len(U2[0])
    # A12 is rows 0-2, cols 3-4 of A.
    # U indices: i in 0-2, j in 3-4.
    # i=0, j=3 -> 3; i=0, j=4 -> 4; i=1, j=3 -> 6; i=1, j=4 -> 7; i=2, j=3 -> 8; i=2, j=4 -> 9.
    # So U indices: 3,4,6,7,8,9.
    # B21 is rows 3-4, cols 0-2 of B.
    # V indices: j in 3-4, k in 0-2.
    # j=3, k=0 -> 15; j=3, k=1 -> 16; j=3, k=2 -> 17; j=4, k=0 -> 20; j=4, k=1 -> 21; j=4, k=2 -> 22.
    # So V indices: 15,16,17,20,21,22.
    # C11 is rows 0-2, cols 0-2 of C.
    # W indices: k in 0-2, i in 0-2.
    # k=0, i=0 -> 0; k=0, i=1 -> 1; k=0, i=2 -> 2; k=1, i=0 -> 4; k=1, i=1 -> 5; k=1, i=2 -> 6; k=2, i=0 -> 8; k=2, i=1 -> 9; k=2, i=2 -> 10.
    # So W indices: 0,1,2,4,5,6,8,9,10.
    
    u_indices = [3,4,6,7,8,9]
    v_indices = [15,16,17,20,21,22]
    w_indices = [0,1,2,4,5,6,8,9,10]
    
    for i_idx, i in enumerate(u_indices):
        for r in range(r2):
            U[i][col_offset + r] = U2[i_idx][r]
    for j_idx, j in enumerate(v_indices):
        for r in range(r2):
            V[j][col_offset + r] = V2[j_idx][r]
    for k_idx, k in enumerate(w_indices):
        for r in range(r2):
            W[k][col_offset + r] = W2[k_idx][r]
    col_offset += r2
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 3. A11*B12: 3x3 * 3x2
    U3, V3, W3 = _schoolbook(3, 3, 2)
    r3 = len(U3[0])
    # A11: U indices 0-8.
    # B12: rows 0-2, cols 3-4 of B.
    # V indices: j in 0-2, k in 3-4.
    # j=0, k=3 -> 3; j=0, k=4 -> 4; j=1, k=3 -> 8; j=1, k=4 -> 9; j=2, k=3 -> 13; j=2, k=4 -> 14.
    # So V indices: 3,4,8,9,13,14.
    # C12: rows 0-2, cols 3-4 of C.
    # W indices: k in 3-4, i in 0-2.
    # k=3, i=0 -> 12; k=3, i=1 -> 13; k=3, i=2 -> 14; k=4, i=0 -> 16; k=4, i=1 -> 17; k=4, i=2 -> 18.
    # So W indices: 12,13,14,16,17,18.
    
    u_indices = [0,1,2,3,4,5,6,7,8]
    v_indices = [3,4,8,9,13,14]
    w_indices = [12,13,14,16,17,18]
    
    for i_idx, i in enumerate(u_indices):
        for r in range(r3):
            U[i][col_offset + r] = U3[i_idx][r]
    for j_idx, j in enumerate(v_indices):
        for r in range(r3):
            V[j][col_offset + r] = V3[j_idx][r]
    for k_idx, k in enumerate(w_indices):
        for r in range(r3):
            W[k][col_offset + r] = W3[k_idx][r]
    col_offset += r3
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 4. A12*B22: 3x2 * 2x2
    U4, V4, W4 = _schoolbook(3, 2, 2)
    r4 = len(U4[0])
    # A12: U indices 3,4,6,7,8,9.
    # B22: rows 3-4, cols 3-4 of B.
    # V indices: j in 3-4, k in 3-4.
    # j=3, k=3 -> 18; j=3, k=4 -> 19; j=4, k=3 -> 23; j=4, k=4 -> 24.
    # So V indices: 18,19,23,24.
    # C12: rows 0-2, cols 3-4 of C.
    # W indices: 12,13,14,16,17,18.
    
    u_indices = [3,4,6,7,8,9]
    v_indices = [18,19,23,24]
    w_indices = [12,13,14,16,17,18]
    
    for i_idx, i in enumerate(u_indices):
        for r in range(r4):
            U[i][col_offset + r] = U4[i_idx][r]
    for j_idx, j in enumerate(v_indices):
        for r in range(r4):
            V[j][col_offset + r] = V4[j_idx][r]
    for k_idx, k in enumerate(w_indices):
        for r in range(r4):
            W[k][col_offset + r] = W4[k_idx][r]
    col_offset += r4
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 5. A21*B11: 1x3 * 3x3
    U5, V5, W5 = _schoolbook(1, 3, 3)
    r5 = len(U5[0])
    # A21: row 3, cols 0-2 of A.
    # U indices: i=3, j in 0-2.
    # i=3, j=0 -> 15; i=3, j=1 -> 16; i=3, j=2 -> 17.
    # So U indices: 15,16,17.
    # B11: rows 0-2, cols 0-2 of B.
    # V indices: 0-8.
    # C21: row 3, cols 0-2 of C.
    # W indices: k in 0-2, i=3.
    # k=0, i=3 -> 3; k=1, i=3 -> 7; k=2, i=3 -> 11.
    # So W indices: 3,7,11.
    
    u_indices = [15,16,17]
    v_indices = [0,1,2,3,4,5,6,7,8]
    w_indices = [3,7,11]
    
    for i_idx, i in enumerate(u_indices):
        for r in range(r5):
            U[i][col_offset + r] = U5[i_idx][r]
    for j_idx, j in enumerate(v_indices):
        for r in range(r5):
            V[j][col_offset + r] = V5[j_idx][r]
    for k_idx, k in enumerate(w_indices):
        for r in range(r5):
            W[k][col_offset + r] = W5[k_idx][r]
    col_offset += r5
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 6. A22*B21: 1x2 * 2x3
    U6, V6, W6 = _schoolbook(1, 2, 3)
    r6 = len(U6[0])
    # A22: row 3, cols 3-4 of A.
    # U indices: i=3, j in 3-4.
    # i=3, j=3 -> 18; i=3, j=4 -> 19.
    # So U indices: 18,19.
    # B21: rows 3-4, cols 0-2 of B.
    # V indices: 15,16,17,20,21,22.
    # C21: row 3, cols 0-2 of C.
    # W indices: 3,7,11.
    
    u_indices = [18,19]
    v_indices = [15,16,17,20,21,22]
    w_indices = [3,7,11]
    
    for i_idx, i in enumerate(u_indices):
        for r in range(r6):
            U[i][col_offset + r] = U6[i_idx][r]
    for j_idx, j in enumerate(v_indices):
        for r in range(r6):
            V[j][col_offset + r] = V6[j_idx][r]
    for k_idx, k in enumerate(w_indices):
        for r in range(r6):
            W[k][col_offset + r] = W6[k_idx][r]
    col_offset += r6
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 7. A21*B12: 1x3 * 3x2
    U7, V7, W7 = _schoolbook(1, 3, 2)
    r7 = len(U7[0])
    # A21: U indices 15,16,17.
    # B12: V indices 3,4,8,9,13,14.
    # C22: row 3, cols 3-4 of C.
    # W indices: k in 3-4, i=3.
    # k=3, i=3 -> 15; k=4, i=3 -> 19.
    # So W indices: 15,19.
    
    u_indices = [15,16,17]
    v_indices = [3,4,8,9,13,14]
    w_indices = [15,19]
    
    for i_idx, i in enumerate(u_indices):
        for r in range(r7):
            U[i][col_offset + r] = U7[i_idx][r]
    for j_idx, j in enumerate(v_indices):
        for r in range(r7):
            V[j][col_offset + r] = V7[j_idx][r]
    for k_idx, k in enumerate(w_indices):
        for r in range(r7):
            W[k][col_offset + r] = W7[k_idx][r]
    col_offset += r7
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 8. A22*B22: 1x2 * 2x2
    U8, V8, W8 = _schoolbook(1, 2, 2)
    r8 = len(U8[0])
    # A22: U indices 18,19.
    # B22: V indices 18,19,23,24.
    # C22: W indices 15,19.
    
    u_indices = [18,19]
    v_indices = [18,19,23,24]
    w_indices = [15,19]
    
    for i_idx, i in enumerate(u_indices):
        for r in range(r8):
            U[i][col_offset + r] = U8[i_idx][r]
    for j_idx, j in enumerate(v_indices):
        for r in range(r8):
            V[j][col_offset + r] = V8[j_idx][r]
    for k_idx, k in enumerate(w_indices):
        for r in range(r8):
            W[k][col_offset + r] = W8[k_idx][r]
    col_offset += r8
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Trim unused columns if any (should be none)
    if col_offset < total_rank:
        U = [row[:col_offset] for row in U]
        V = [row[:col_offset] for row in V]
        W = [row[:col_offset] for row in W]
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
