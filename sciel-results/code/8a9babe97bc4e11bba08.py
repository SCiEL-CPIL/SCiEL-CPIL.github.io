"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
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

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition. Rank 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(16)] # 4x4 = 16 rows
    V = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows
    W = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows (p*n = 2*4)
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j # m=4
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk # B's col is C's col
                        v_row = global_j * 2 + global_k # p=2
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i # n=4
                        W[w_row][col] += coeff_c

    # 1. A11 * B11 -> C11
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # 2. A12 * B21 -> C11
    add_strassen_block(0, 2, 2, 0, 0, 7)
    
    # 3. A21 * B11 -> C21
    add_strassen_block(2, 0, 0, 2, 0, 14)
    
    # 4. A22 * B21 -> C21
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
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

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_2x2x2(n, m, p):
    """
    Build 2x2x2 Strassen block for arbitrary small dimensions.
    This is a helper that constructs a rank-7 decomposition for a 2x2x2 multiplication.
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
    V = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
    W = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
    
    # Same as _build_strassen_2x2 but with generic indexing
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x5x5x3():
    """
    Build 4x5x5x3 decomposition: 4x5 * 5x3 -> 4x3.
    Split A into [4x2 | 4x3], B into [2x3; 3x3].
    Use Strassen for 4x2x2 and 4x3x3 parts.
    """
    # A = [A11 (4x2) | A12 (4x3)]
    # B = [B11 (2x3); B12 (3x3)]
    # C = A11*B11 + A12*B12
    
    # Part 1: 4x2 * 2x3 -> 4x3
    # This is 4x2x3. We can decompose as 2 blocks of 2x2x2 + 1 block of 2x2x1
    # Actually, let's just use schoolbook for simplicity here, or a mixed approach.
    # Let's use: 4x2x3 = (2x2x3) + (2x2x3)
    # 2x2x3 can be done with 2 Strassen 2x2x2 blocks? No, 2x2x2 is rank 7.
    # 2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 2*2*1=4 or less)
    # Schoolbook for 2x2x3 is rank 12. Strassen 2x2x2 is 7, so 7+4=11 < 12.
    
    # Let's build 2x2x3 decomposition:
    # 2x2x2 Strassen (rank 7) for first 2 cols, + 2x2x1 schoolbook (rank 4) for last col
    def _build_2x2x3():
        rank = 11
        U = [[0]*rank for _ in range(4)]  # 2x2
        V = [[0]*rank for _ in range(6)]  # 2x3
        W = [[0]*rank for _ in range(6)]  # 2x3 (p*n = 3*2)
        
        # 2x2x2 Strassen for cols 0,1
        U2, V2, W2 = _build_strassen_2x2()
        for r in range(4):
            for c in range(7):
                U[r][c] = U2[r][c]
                V[r][c] = V2[r][c]
                W[r][c] = W2[r][c]
        
        # 2x2x1 schoolbook for col 2
        # Rank 4
        for i in range(2):
            for j in range(2):
                col = 7 + i*2 + j
                U[i*2 + j][col] = 1
                V[j*3 + 2][col] = 1  # B[j,2]
                W[2*2 + i][col] = 1  # W[k*n+i] with k=2, n=2 -> 2*2+i
        return U, V, W
    
    U223, V223, W223 = _build_2x2x3()
    rank1 = 11
    U1 = [[0]*rank1 for _ in range(8)]  # 4x2
    V1 = [[0]*rank1 for _ in range(6)]  # 2x3
    W1 = [[0]*rank1 for _ in range(6)]  # 3x4 (p*n = 3*4)
    
    # Two blocks of 2x2x3
    for block in range(2):
        for r in range(4):
            for c in range(rank1):
                U1[block*4 + r][c] = U223[r][c]
        for r in range(6):
            for c in range(rank1):
                V1[r][c] = V223[r][c]
        for r in range(6):
            for c in range(rank1):
                W1[r][c] = W223[r][c]
    
    # Part 2: 4x3 * 3x3 -> 4x3
    # This is 4x3x3. Use schoolbook? Rank 36.
    # Or Strassen: 4x3x3 = 4x2x2 (rank 28) + 4x1x1 (rank 4) + 4x2x1 (rank ?)
    # Actually, 4x3x3 = (4x2x2) + (4x1x2) + (4x2x1)
    # 4x2x2: rank 28 (from _build_strassen_4x4x2 style)
    # Let's just use schoolbook for 4x3x3 for now, rank 36.
    U2, V2, W2 = _build_schoolbook(4, 3, 3)
    rank2 = 36
    
    total_rank = rank1 + rank2
    U = [[0]*total_rank for _ in range(20)]  # 4x5
    V = [[0]*total_rank for _ in range(15)]  # 5x3
    W = [[0]*total_rank for _ in range(12)]  # 3x4
    
    # Copy Part 1: 4x2 * 2x3
    for r in range(8):  # 4x2 rows
        i = r // 2
        j = r % 2
        U[i*5 + j][c] = U1[r][c]
    for r in range(6):  # 2x3 rows
        V[r][c] = V1[r][c]
    for r in range(12):  # 3x4 rows
        W[r][c] = W1[r][c]
    
    # Copy Part 2: 4x3 * 3x3
    for r in range(12):  # 4x3 rows
        i = r // 3
        j = r % 3
        U[i*5 + 2 + j][c] = U2[r][c]
    for r in range(9):  # 3x3 rows
        V[2 + r][c] = V2[r][c]
    for r in range(12):  # 3x4 rows
        W[r][c] = W2[r][c]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # 1. Build 4x4x4 Strassen (Rank 49)
        U1, V1, W1 = _build_strassen_4x4()
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        
        # 2. Build 4x1x6 Schoolbook (Rank 24)
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        
        # 3. Build 4x4x2 Strassen (Rank 28)
        U3, V3, W3 = _build_strassen_4x4x2()
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
        
        U, V, W = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook if something goes wrong
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
