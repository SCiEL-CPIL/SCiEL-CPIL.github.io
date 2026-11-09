"""Matrix multiplication tensor decomposition for 4x5 * 5x5 using Strassen blocks."""
import time

n, m, p = 4, 5, 5


def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of n x m * m x p."""
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Return (U, V, W) for Strassen decomposition of 2x2 * 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[i*2+j, c], W[i*2+j, c]
    where i,j are matrix indices.
    """
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M7
    # C22 = M1 - M3 + M4
    
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1: (A11+A22)(B11+B22)
    # A indices: A11=0, A22=3
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    # B indices: B11=0, B22=3
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    # C11: +1, C22: +1
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2: (A21+A22)B11
    U[1][1] = 1  # A21 (i=1,j=0 -> index 2? Wait, i*2+j: i=1,j=0 -> 2)
    # Let me be careful with indexing:
    # A[i,j] where i=row, j=col. Index = i*2+j
    # A11 = i=0,j=0 -> 0
    # A12 = i=0,j=1 -> 1
    # A21 = i=1,j=0 -> 2
    # A22 = i=1,j=1 -> 3
    # Similarly for B and C
    
    # M2: (A21+A22)B11
    # A21 = index 2, A22 = index 3
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    # C21: +1
    W[2][1] = 1  # C21 (i=1,j=0 -> index 2)
    
    # M3: A11(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1  # B12 (i=0,j=1 -> index 1)
    V[3][2] = -1  # B22 (i=1,j=1 -> index 3)
    # C12: +1
    W[1][2] = 1  # C12 (i=0,j=1 -> index 1)
    
    # M4: A22(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1  # B21 (i=1,j=0 -> index 2)
    V[0][3] = -1  # B11
    # C11: -1, C22: +1
    W[0][3] = -1  # C11
    W[3][3] = 1  # C22
    
    # M5: (A11+A12)B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    # C12: +1
    W[1][4] = 1  # C12
    
    # M6: (A21-A11)(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1  # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    # C11: +1
    W[0][5] = 1  # C11
    
    # M7: (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1  # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    # C21: +1
    W[2][6] = 1  # C21
    
    return U, V, W


def _combine(decompositions):
    """
    Combine multiple decompositions by concatenating columns.
    Each decomposition is (U, V, W) with the same row counts.
    """
    if not decompositions:
        return [], [], []
    
    first = decompositions[0]
    rows_u = len(first[0])
    rows_v = len(first[1])
    rows_w = len(first[2])
    
    U = [[0] * sum(len(d[0][0]) for d in decompositions) for _ in range(rows_u)]
    V = [[0] * sum(len(d[1][0]) for d in decompositions) for _ in range(rows_v)]
    W = [[0] * sum(len(d[2][0]) for d in decompositions) for _ in range(rows_w)]
    
    col_offset = 0
    for U_d, V_d, W_d in decompositions:
        rank_d = len(U_d[0])
        for i in range(rows_u):
            for c in range(rank_d):
                U[i][col_offset + c] += U_d[i][c]
        for i in range(rows_v):
            for c in range(rank_d):
                V[i][col_offset + c] += V_d[i][c]
        for i in range(rows_w):
            for c in range(rank_d):
                W[i][col_offset + c] += W_d[i][c]
        col_offset += rank_d
    
    return U, V, W


def _strassen_4x4x4():
    """
    Return (U, V, W) for Strassen-based decomposition of 4x4 * 4x4.
    U, V, W each have shape (16, 49).
    """
    # Split 4x4 into 2x2 blocks:
    # A = [[A_LL, A_LR], [A_RL, A_RR]] where each is 2x2
    # B = [[B_LL, B_LR], [B_RL, B_RR]]
    # C = [[C_LL, C_LR], [C_RL, C_RR]]
    #
    # C_LL = A_LL B_LL + A_LR B_RL
    # C_LR = A_LL B_LR + A_LR B_RR
    # C_RL = A_RL B_LL + A_RR B_RL
    # C_RR = A_RL B_LR + A_RR B_RR
    #
    # Use Strassen on the 2x2 block matrix:
    # M1 = (A_LL+A_RR)(B_LL+B_RR)
    # M2 = (A_RL+A_RR)B_LL
    # M3 = A_LL(B_LR-B_RR)
    # M4 = A_RR(B_RL-B_LL)
    # M5 = (A_LL+A_LR)B_RR
    # M6 = (A_RL-A_LL)(B_LL+B_LR)
    # M7 = (A_LR-A_RR)(B_RL+B_RR)
    #
    # C_LL = M1 - M4 + M6
    # C_LR = M3 + M5
    # C_RL = M2 + M7
    # C_RR = M1 - M3 + M4
    #
    # Each M_j is a 2x2 * 2x2 product, which itself uses Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # We need to construct the full 4x4x4 decomposition.
    # For each of the 7 block-level Strassen terms, we have a 2x2*2x2 Strassen.
    
    # Let me think about this differently. I'll construct the decomposition
    # by composing the two levels of Strassen.
    
    # Level 1: 7 block multiplications, each is 2x2 * 2x2
    # Level 2: each 2x2 * 2x2 uses 7 scalar multiplications
    
    # The total U, V, W will have 49 columns.
    # For block term j (0..6) and inner term k (0..6), column j*7+k.
    
    # Block-level Strassen coefficients:
    # For each block term j, we need coefficients for A blocks and B blocks.
    # A blocks: LL(0), LR(1), RL(2), RR(3)
    # B blocks: LL(0), LR(1), RL(2), RR(3)
    
    # M1: (A_LL+A_RR)(B_LL+B_RR)
    # M2: (A_RL+A_RR)B_LL
    # M3: A_LL(B_LR-B_RR)
    # M4: A_RR(B_RL-B_LL)
    # M5: (A_LL+A_LR)B_RR
    # M6: (A_RL-A_LL)(B_LL+B_LR)
    # M7: (A_LR-A_RR)(B_RL+B_RR)
    
    # A block coefficients for each M_j:
    # j=0: A_LL=1, A_LR=0, A_RL=0, A_RR=1
    # j=1: A_LL=0, A_LR=0, A_RL=1, A_RR=1
    # j=2: A_LL=1, A_LR=0, A_RL=0, A_RR=0
    # j=3: A_LL=0, A_LR=0, A_RL=0, A_RR=1
    # j=4: A_LL=1, A_LR=1, A_RL=0, A_RR=0
    # j=5: A_LL=-1, A_LR=0, A_RL=1, A_RR=0
    # j=6: A_LL=0, A_LR=1, A_RL=0, A_RR=-1
    
    A_block_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    
    # B block coefficients for each M_j:
    # j=0: B_LL=1, B_LR=0, B_RL=0, B_RR=1
    # j=1: B_LL=1, B_LR=0, B_RL=0, B_RR=0
    # j=2: B_LL=0, B_LR=1, B_RL=0, B_RR=-1
    # j=3: B_LL=-1, B_LR=0, B_RL=1, B_RR=0
    # j=4: B_LL=0, B_LR=0, B_RL=0, B_RR=1
    # j=5: B_LL=1, B_LR=1, B_RL=0, B_RR=0
    # j=6: B_LL=0, B_LR=0, B_RL=1, B_RR=1
    
    B_block_coeffs = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    
    # C block output coefficients:
    # C_LL = M1 - M4 + M6
    # C_LR = M3 + M5
    # C_RL = M2 + M7
    # C_RR = M1 - M3 + M4
    
    # For each C block (0=LL, 1=LR, 2=RL, 3=RR), coefficients for M_j:
    C_block_coeffs = [
        [1, 0, 0, -1, 0, 1, 0],  # C_LL
        [0, 0, 1, 0, 1, 0, 0],   # C_LR
        [0, 1, 0, 0, 0, 0, 1],   # C_RL
        [1, 0, -1, 1, 0, 0, 0],  # C_RR
    ]
    
    # Now, for each block term j and inner Strassen term k, we construct the
    # contribution to U, V, W.
    
    # The 4x4 matrix A has elements A[i,j] with i,j in 0..3.
    # Block LL: i in 0..1, j in 0..1
    # Block LR: i in 0..1, j in 2..3
    # Block RL: i in 2..3, j in 0..1
    # Block RR: i in 2..3, j in 2..3
    
    # For a given block b (0..3) and inner index (ii, jj) in 0..1:
    # Global index: depends on block
    # LL: i=ii, j=jj
    # LR: i=ii, j=jj+2
    # RL: i=ii+2, j=jj
    # RR: i=ii+2, j=jj+2
    
    def block_global_i(b, ii):
        if b == 0 or b == 1:
            return ii
        else:
            return ii + 2
    
    def block_global_j(b, jj):
        if b == 0 or b == 2:
            return jj
        else:
            return jj + 2
    
    # Get the 2x2 Strassen decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Now construct the full 4x4x4 decomposition
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    for j in range(7):
        for k in range(7):
            col = j * 7 + k
            
            # For U: A block coefficients
            for b in range(4):
                coeff_a = A_block_coeffs[j][b]
                if coeff_a != 0:
                    for ii in range(2):
                        for jj in range(2):
                            global_i = block_global_i(b, ii)
                            global_j = block_global_j(b, jj)
                            idx = global_i * 4 + global_j
                            U[idx][col] += coeff_a * U_s[ii * 2 + jj][k]
            
            # For V: B block coefficients
            for b in range(4):
                coeff_b = B_block_coeffs[j][b]
                if coeff_b != 0:
                    for ii in range(2):
                        for jj in range(2):
                            global_i = block_global_i(b, ii)
                            global_j = block_global_j(b, jj)
                            idx = global_i * 4 + global_j
                            V[idx][col] += coeff_b * V_s[ii * 2 + jj][k]
            
            # For W: C block coefficients
            for b in range(4):
                coeff_c = C_block_coeffs[b][j]
                if coeff_c != 0:
                    for ii in range(2):
                        for jj in range(2):
                            global_i = block_global_i(b, ii)
                            global_j = block_global_j(b, jj)
                            idx = global_i * 4 + global_j
                            W[idx][col] += coeff_c * W_s[ii * 2 + jj][k]
    
    return U, V, W


def _scaled_schoolbook(n, m, p, scale_u, scale_v, scale_w):
    """
    Return (U, V, W) for schoolbook with scaling.
    U entries are scaled by scale_u, V by scale_v, W by scale_w.
    """
    U, V, W = _schoolbook(n, m, p)
    for i in range(len(U)):
        for c in range(len(U[0])):
            U[i][c] *= scale_u
    for i in range(len(V)):
        for c in range(len(V[0])):
            V[i][c] *= scale_v
    for i in range(len(W)):
        for c in range(len(W[0])):
            W[i][c] *= scale_w
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Decompose 4x5 * 5x5 into blocks:
    # A is 4x5: split into A_L (4x4) and A_R (4x1)
    # B is 5x5: split into B_L (5x4) and B_R (5x1)
    # B_L: split into B_LL (4x4), B_LR (1x4), B_RL (4x4? No)
    
    # Let me think more carefully:
    # A (4x5) * B (5x5)
    # Split A into A1 (4x4) and A2 (4x1)
    # Split B into B1 (5x4) and B2 (5x1)
    # C = A1*B1 + A2*B2
    # A1*B1 is 4x4 * 4x5
    # A2*B2 is 4x1 * 1x5
    
    # A1*B1: 4x4 * 4x5
    # Split B1 into B11 (4x4) and B12 (4x1)
    # A1*B1 = A1*B11 + A1*B12
    # A1*B11: 4x4 * 4x4 -> Strassen, rank 49
    # A1*B12: 4x4 * 4x1 -> schoolbook, rank 16
    # A2*B2: 4x1 * 1x5 -> schoolbook, rank 20
    
    # Total rank: 49 + 16 + 20 = 85
    
    # Let me construct each part.
    
    # Part 1: 4x4 * 4x4 (Strassen)
    U1, V1, W1 = _strassen_4x4x4()
    
    # Part 2: 4x4 * 4x1 (schoolbook)
    # A1 (4x4) * B12 (4x1)
    # U: 4x4 = 16 rows, rank 16
    # V: 4x1 = 4 rows, rank 16
    # W: 1x4 = 4 rows, rank 16
    U2, V2, W2 = _schoolbook(4, 4, 1)
    
    # Part 3: 4x1 * 1x5 (schoolbook)
    # A2 (4x1) * B2 (1x5)
    # U: 4x1 = 4 rows, rank 20
    # V: 1x5 = 5 rows, rank 20
    # W: 5x4 = 20 rows, rank 20
    U3, V3, W3 = _schoolbook(4, 1, 5)
    
    # Now I need to map these to the full 4x5 * 5x5 problem.
    # Full problem: n=4, m=5, p=5
    # U: n*m = 20 rows, V: m*p = 25 rows, W: p*n = 20 rows
    
    # Let me define the mapping:
    # A (4x5): A[i,j] for i=0..3, j=0..4
    # A1 = A[:, 0:4] (4x4), A2 = A[:, 4:5] (4x1)
    # B (5x5): B[j,k] for j=0..4, k=0..4
    # B1 = B[0:4, :] (4x5), B2 = B[4:5, :] (1x5)
    # B11 = B[0:4, 0:4] (4x4), B12 = B[0:4, 4:5] (4x1)
    # B2 = B[4:5, :] (1x5)
    
    # C = A1*B11 + A1*B12 + A2*B2
    
    # Part 1: A1*B11
    # U1: rows for A1 (4x4) -> indices 0..15 in full U (i*5+j for i=0..3, j=0..3)
    # V1: rows for B11 (4x4) -> indices 0..15 in full V (j*5+k for j=0..3, k=0..3)
    # W1: rows for C (4x4) -> indices k*4+i for k=0..3, i=0..3
    
    # Part 2: A1*B12
    # U2: rows for A1 (4x4) -> same as part 1
    # V2: rows for B12 (4x1) -> indices j*5+4 for j=0..3 -> 4, 9, 14, 19
    # W2: rows for C (4x1) -> indices 4*4+i for i=0..3 -> 16, 17, 18, 19
    
    # Part 3: A2*B2
    # U3: rows for A2 (4x1) -> indices i*5+4 for i=0..3 -> 4, 9, 14, 19
    # V3: rows for B2 (1x5) -> indices 4*5+k for k=0..4 -> 20, 21, 22, 23, 24
    # W3: rows for C (5x4) -> indices k*4+i for k=0..4, i=0..3 -> 0..19
    
    # Wait, I need to be more careful with the W indexing.
    # W has shape p*n = 20, indexed by k*n+i where k=0..4, i=0..3.
    # So W index = k*4+i.
    
    # Let me construct the full U, V, W.
    total_rank = 49 + 16 + 20
    U = [[0] * total_rank for _ in range(20)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(20)]
    
    # Part 1: Strassen 4x4x4, columns 0..48
    for i in range(16):
        a_i = i // 4  # 0..3
        a_j = i % 4   # 0..3
        u_idx = a_i * 5 + a_j
        for c in range(49):
            U[u_idx][c] += U1[i][c]
    
    for i in range(16):
        b_i = i // 4  # 0..3
        b_j = i % 4   # 0..3
        v_idx = b_i * 5 + b_j
        for c in range(49):
            V[v_idx][c] += V1[i][c]
    
    for i in range(16):
        c_k = i // 4  # 0..3
        c_i = i % 4   # 0..3
        w_idx = c_k * 4 + c_i
        for c in range(49):
            W[w_idx][c] += W1[i][c]
    
    # Part 2: Schoolbook 4x4x1, columns 49..64
    for i in range(16):
        a_i = i // 4
        a_j = i % 4
        u_idx = a_i * 5 + a_j
        for c in range(16):
            U[u_idx][49 + c] += U2[i][c]
    
    for i in range(16):
        # V2 has 4 rows (j=0..3), each with 16 columns
        # But V2 is for 4x1, so V2 has shape (4, 16)
        # Wait, _schoolbook(4,4,1) returns V with shape (m*p, rank) = (4*1, 16) = (4, 16)
        b_i = i  # 0..3
        b_j = 0  # only one column
        # Actually, V2[i] corresponds to B[j, k] where j=0..3, k=0 (only k=0)
        # So v_idx = j*5 + k = i*5 + 4
        v_idx = i * 5 + 4
        for c in range(16):
            V[v_idx][49 + c] += V2[i][c]
    
    for i in range(16):
        # W2 has shape (p*n, rank) = (1*4, 16) = (4, 16)
        # W2[i] corresponds to C[k, i] where k=0 (only k=0), i=0..3
        # w_idx = k*4 + i = 0*4 + i = i
        # Wait, that's not right. C is 4x1, so C[i, 0] for i=0..3
        # w_idx = k*4 + i where k=0, i=0..3 -> 0, 1, 2, 3
        # But that conflicts with part 1's W indices!
        
        # I think I made an error. Let me reconsider.
        
        # C = A1*B11 + A1*B12 + A2*B2
        # A1*B11 is 4x4, so it contributes to C[0..3, 0..3]
        # A1*B12 is 4x1, so it contributes to C[0..3, 4] (column 4)
        # A2*B2 is 4x5, so it contributes to C[0..3, 0..4]
        
        # Wait, that's not right either. Let me re-examine.
        
        # A (4x5) * B (5x5) = C (4x5)
        # C[i, k] = sum_j A[i, j] B[j, k]
        
        # Split j into 0..3 and 4:
        # C[i, k] = sum_{j=0}^{3} A[i, j] B[j, k] + A[i, 4] B[4, k]
        
        # Let C1[i, k] = sum_{j=0}^{3} A[i, j] B[j, k]
        # Let C2[i, k] = A[i, 4] B[4, k]
        
        # C1 is 4x5, C2 is 4x5
        
        # C1 = A1 * B1 where A1 is 4x4, B1 is 4x5
        # C2 = A2 * B2 where A2 is 4x1, B2 is 1x5
        
        # Now split B1 into B11 (4x4) and B12 (4x1):
        # C1 = A1 * B11 + A1 * B12
        # A1*B11 is 4x4, A1*B12 is 4x1
        
        # So:
        # C[i, 0..3] = (A1*B11)[i, 0..3] + (A1*B12)[i, 0] * 0 + ... 
        # Wait, A1*B12 is 4x1, so it only contributes to column 4 of C1.
        
        # C1[i, k] for k=0..3: from A1*B11
        # C1[i, 4]: from A1*B12
        
        # So:
        # C[i, 0..3] = (A1*B11)[i, 0..3] + C2[i, 0..3]
        # C[i, 4] = (A1*B12)[i, 0] + C2[i, 4]
        
        # This means:
        # Part 1 (A1*B11): contributes to C[0..3, 0..3]
        # Part 2 (A1*B12): contributes to C[0..3, 4]
        # Part 3 (A2*B2): contributes to C[0..3, 0..4]
        
        # So W indexing:
        # Part 1: W[k*4+i] for k=0..3, i=0..3
        # Part 2: W[4*4+i] = W[16+i] for i=0..3
        # Part 3: W[k*4+i] for k=0..4, i=0..3
        
        # This works! No conflicts because:
        # Part 1 uses W indices 0..15
        # Part 2 uses W indices 16..19
        # Part 3 uses W indices 0..19
        
        c_k = i // 4  # 0 (only k=0 for 4x1 result)
        c_i = i % 4   # 0..3
        w_idx = c_k * 4 + c_i  # 0..3
        # But wait, this should be column 4 of C, so w_idx = 4*4 + i = 16+i
        # Let me fix this.
        
        # A1*B12 is 4x1, result has one column (column 4 of C)
        # So w_idx = 4*4 + i for i=0..3 -> 16, 17, 18, 19
        
        # But W2 has shape (4, 16), and W2[i] corresponds to the i-th row of the 4x1 result.
        # The i-th row of the 4x1 result is C[i, 4], so w_idx = 4*4 + i = 16+i.
        
        pass  # Will fix below
    
    # Let me redo parts 2 and 3 more carefully.
    
    # Part 2: A1*B12 (4x4 * 4x1 = 4x1)
    # U2: shape (16, 16)
    # V2: shape (4, 16)
    # W2: shape (4, 16)
    
    for i in range(16):
        a_i = i // 4
        a_j = i % 4
        u_idx = a_i * 5 + a_j
        for c in range(16):
            U[u_idx][49 + c] += U2[i][c]
    
    for i in range(4):
        # V2[i] corresponds to B[j=i, k=0] in the 4x1 matrix
        # In the full B, this is B[j=i, k=4]
        v_idx = i * 5 + 4
        for c in range(16):
            V[v_idx][49 + c] += V2[i][c]
    
    for i in range(4):
        # W2[i] corresponds to C[i, 0] in the 4x1 result
        # In the full C, this is C[i, 4]
        w_idx = 4 * 4 + i
        for c in range(16):
            W[w_idx][49 + c] += W2[i][c]
    
    # Part 3: A2*B2 (4x1 * 1x5 = 4x5)
    # U3: shape (4, 20)
    # V3: shape (5, 20)
    # W3: shape (20, 20)
    
    for i in range(4):
        # U3[i] corresponds to A[i, 0] in the 4x1 matrix
        # In the full A, this is A[i, 4]
        u_idx = i * 5 + 4
        for c in range(20):
            U[u_idx][65 + c] += U3[i][c]
    
    for i in range(5):
        # V3[i] corresponds to B[0, i] in the 1x5 matrix
        # In the full B, this is B[4, i]
        v_idx = 4 * 5 + i
        for c in range(20):
            V[v_idx][65 + c] += V3[i][c]
    
    for i in range(20):
        # W3[i] corresponds to C[k, i] in the 4x5 result
        # k = i // 5? No, W3 has shape (p*n, rank) = (5*4, 20) = (20, 20)
        # W3[i] where i = k*4 + i for k=0..4, i=0..3
        # In the full C, this is C[i, k]
        c_k = i // 4
        c_i = i % 4
        w_idx = c_k * 4 + c_i
        for c in range(20):
            W[w_idx][65 + c] += W3[i][c]
    
    # Verify the decomposition
    if time.time() < deadline - 10:
        valid = True
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    total = 0
                    for c in range(total_rank):
                        total += U[i * 5 + j][c] * V[j * 5 + k][c] * W[k * 4 + i][c]
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fall back to schoolbook
            U, V, W = _schoolbook(4, 5, 5)
    
    return {"U": U, "V": V, "W": W}
