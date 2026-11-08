"""Matrix multiplication 4x5x6 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U rows: (i,j) -> i*2+j for A
    V rows: (j,k) -> j*2+k for B
    W rows: (k,i) -> k*2+i for C
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m * m x p -> n x p.
    Rank = n*m*p.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _decomp_block(n_sub, m_sub, p_sub, rank_sub, U_sub, V_sub, W_sub,
                  i_s, j_s, k_s, n, m, p, U, V, W, offset):
    """
    Place a sub-decomposition for block A[i_s:i_s+n_sub, j_s:j_s+m_sub] *
    B[j_s:j_s+m_sub, k_s:k_s+p_sub] -> C[i_s:i_s+n_sub, k_s:k_s+p_sub]
    into the global matrices at column offset.
    """
    for r in range(n_sub * m_sub):
        i_loc = r // m_sub
        j_loc = r % m_sub
        i_glob = i_s + i_loc
        j_glob = j_s + j_loc
        r_glob = i_glob * m + j_glob
        for c in range(rank_sub):
            U[r_glob][offset + c] = U_sub[r][c]
    
    for r in range(m_sub * p_sub):
        j_loc = r // p_sub
        k_loc = r % p_sub
        j_glob = j_s + j_loc
        k_glob = k_s + k_loc
        r_glob = j_glob * p + k_glob
        for c in range(rank_sub):
            V[r_glob][offset + c] = V_sub[r][c]
    
    for r in range(p_sub * n_sub):
        k_loc = r // n_sub
        i_loc = r % n_sub
        k_glob = k_s + k_loc
        i_glob = i_s + i_loc
        r_glob = k_glob * n + i_glob
        for c in range(rank_sub):
            W[r_glob][offset + c] = W_sub[r][c]

def _build_strassen_4x4():
    """
    Build 4x4x4 Strassen decomposition (rank 49).
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    rank44 = 49
    U44 = [[0]*rank44 for _ in range(16)]
    V44 = [[0]*rank44 for _ in range(16)]
    W44 = [[0]*rank44 for _ in range(16)]
    
    # 7 Strassen blocks, each 2x2x2, placed at different (i,j,k) block positions
    # Block positions: (bi, bj, bk) where bi,bj,bk in {0,1}
    # For each Strassen "step" s (0..6), we apply it to a 2x2x2 sub-block
    # The 4x4x4 is decomposed as:
    # For each of 7 Strassen formulas, applied to each of 7 block combinations
    # Actually, the standard recursive Strassen for 4x4x4:
    # Split into 8 sub-blocks of size 2x2x2
    # Use 7 Strassen products instead of 8
    
    # The 7 Strassen combinations for 4x4x4:
    # Each "super-Strassen" step operates on 2x2 blocks of the 4x4 matrices
    # There are 7 such steps, each with rank 7 (from 2x2 Strassen)
    # Total rank = 7*7 = 49
    
    # For step s (0..6), the block combination is:
    # The 7 Strassen formulas applied at the "block level" where each block is 2x2
    # Block (bi, bj, bk) means A[bi*2:(bi+1)*2, bj*2:(bj+1)*2], etc.
    
    # Strassen block definitions: (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of ((bi, bj), sigma) for A sub-blocks
    # b_blocks: list of ((bj, bk), sigma) for B sub-blocks  
    # c_blocks: list of ((bi, bk), sigma) for C sub-blocks
    
    strassen_defs = [
        # P1 = (A00+A11)*(B00+B11)
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # P2 = (A10+A11)*B00
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # P3 = A00*(B01-B11)
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # P4 = A11*(B10-B00)
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # P5 = (A00+A01)*B11
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # P6 = (A10-A00)*(B00+B01)
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # P7 = (A01-A11)*(B10+B11)
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            # Apply 2x2 Strassen within each selected block
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U44[u_row][col] += coeff_a * sigma
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V44[v_row][col] += coeff_b * sigma
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W44[w_row][col] += coeff_c * sigma
    
    return U44, V44, W44

def _build_strassen_4x4x2():
    """
    Build 4x4x2 Strassen decomposition (rank 28).
    A is 4x4, B is 4x2, C is 4x2.
    U shape: (16, 28), V shape: (8, 28), W shape: (8, 28)
    """
    U2, V2, W2 = _strassen_2x2()
    rank442 = 28
    U442 = [[0]*rank442 for _ in range(16)]
    V442 = [[0]*rank442 for _ in range(8)]
    W442 = [[0]*rank442 for _ in range(8)]
    
    # 4 Strassen 2x2x2 blocks
    # Block (bi, bj, bk) where bi in {0,1}, bj in {0,1}, bk=0 (only 1 col of B)
    # Actually B is 4x2, so B[bi*2:(bi+1)*2, bk] for bk in {0,1}
    # We need 2x2x2 blocks: A[2x2] * B[2x2] -> C[2x2]
    # B[2x2] means B[2 rows, 2 cols]. But B is 4x2, so each 2x2 block of B uses 2 rows and both cols.
    # 4 blocks: (bi=0,bj=0), (bi=0,bj=1), (bi=1,bj=0), (bi=1,bj=1)
    # Each is 2x2x2 with rank 7. Total: 4*7=28.
    
    block_positions = [(0,0), (0,1), (1,0), (1,1)]
    
    for idx, (bi, bj) in enumerate(block_positions):
        base = idx * 7
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = bi * 2 + li
                        global_j = bj * 2 + lj
                        u_row = global_i * 4 + global_j
                        U442[u_row][col] += coeff_a
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = bj * 2 + lj
                        global_k = lk  # B has 2 cols, k in {0,1}
                        v_row = global_j * 2 + global_k
                        V442[v_row][col] += coeff_b
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = lk
                        global_i = bi * 2 + li
                        w_row = global_k * 4 + global_i
                        W442[w_row][col] += coeff_c
    
    return U442, V442, W442

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    try:
        # Strategy: 
        # A is 4x5, B is 5x6, C is 4x6
        # Split: m=5 as 4+1, p=6 as 4+2
        # 
        # C[0:4, 0:4] = A[0:4,0:4]*B[0:4,0:4] + A[0:4,4:5]*B[4:5,0:4]
        # C[0:4, 4:6] = A[0:4,0:4]*B[0:4,4:6] + A[0:4,4:5]*B[4:5,4:6]
        #
        # Decompositions:
        # 1. 4x4x4 Strassen: rank 49 (A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4])
        # 2. 4x4x2 Strassen: rank 28 (A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6])
        # 3. 4x1x4 Schoolbook: rank 16 (A[0:4,4:5] * B[4:5,0:4] -> C[0:4,0:4])
        # 4. 4x1x2 Schoolbook: rank 8 (A[0:4,4:5] * B[4:5,4:6] -> C[0:4,4:6])
        #
        # Total: 49 + 28 + 16 + 8 = 101
        
        U44, V44, W44 = _build_strassen_4x4()
        U442, V442, W442 = _build_strassen_4x4x2()
        U414, V414, W414 = _schoolbook(4, 1, 4)
        U412, V412, W412 = _schoolbook(4, 1, 2)
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        
        total_rank = 49 + 28 + 16 + 8
        U = [[0]*total_rank for _ in range(n*m)]  # 4*5=20 rows
        V = [[0]*total_rank for _ in range(m*p)]  # 5*6=30 rows
        W = [[0]*total_rank for _ in range(p*n)]  # 6*4=24 rows
        
        # Place decomp1: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
        offset = 0
        _decomp_block(4, 4, 4, 49, U44, V44, W44, 0, 0, 0, n, m, p, U, V, W, offset)
        
        # Place decomp2: A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6]
        offset = 49
        _decomp_block(4, 4, 2, 28, U442, V442, W442, 0, 0, 4, n, m, p, U, V, W, offset)
        
        # Place decomp3: A[0:4,4:5] * B[4:5,0:4] -> C[0:4,0:4]
        offset = 77
        _decomp_block(4, 1, 4, 16, U414, V414, W414, 0, 4, 0, n, m, p, U, V, W, offset)
        
        # Place decomp4: A[0:4,4:5] * B[4:5,4:6] -> C[0:4,4:6]
        offset = 93
        _decomp_block(4, 1, 2, 8, U412, V412, W412, 0, 4, 4, n, m, p, U, V, W, offset)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
