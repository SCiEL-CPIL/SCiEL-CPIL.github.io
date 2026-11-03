"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
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
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]

    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t

            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a

            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b

            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)

    return U, V, W

def _build_strassen_4x2x6():
    """
    Build 4x2x6 decomposition.
    A is 4x2, B is 2x6, C is 4x6.
    Split into 2 blocks: 2x2x6 each.
    Each 2x2x6 can be done with 3 Strassen 2x2x2 blocks: rank 21.
    Total rank = 42.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 42
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(24)]
    
    # Block 1: A rows 0-1, all cols; B rows 0-1, cols 0-5; C rows 0-1, cols 0-5
    # This is 2x2 * 2x6 -> 2x6
    # Split into 3 2x2x2 Strassen blocks
    for s in range(3):
        k_start = s * 2
        base_rank = s * 7
        for t in range(7):
            col = base_rank + t
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = li
                        global_j = lj
                        u_row = global_i * 2 + global_j
                        U[u_row][col] += coeff_a
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = lj
                        global_k = k_start + lk
                        v_row = global_j * 6 + global_k
                        V[v_row][col] += coeff_b
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = k_start + lk
                        global_i = li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c
    
    # Block 2: A rows 2-3, all cols; B rows 0-1, cols 0-5; C rows 2-3, cols 0-5
    for s in range(3):
        k_start = s * 2
        base_rank = 21 + s * 7
        for t in range(7):
            col = base_rank + t
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = 2 + li
                        global_j = lj
                        u_row = global_i * 2 + global_j
                        U[u_row][col] += coeff_a
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = lj
                        global_k = k_start + lk
                        v_row = global_j * 6 + global_k
                        V[v_row][col] += coeff_b
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = k_start + lk
                        global_i = 2 + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c
    
    return U, V, W

def _build_strassen_4x5_5x6():
    """
    Build a 4x5 * 5x6 decomposition using block partition:
    Split the 5-column dimension into 3+2.
    - A_left (4x3) * B_top (3x6) + A_right (4x2) * B_bot (2x6)
    
    A_left * B_top: 4x3 * 3x6
      Split A_left into A_1 (2x3) + A_2 (2x3)
      B_top is 3x6
      A_1 * B_top: 2x3 * 3x6 = 2x6
      A_2 * B_top: 2x3 * 3x6 = 2x6
      For 2x3 * 3x6: split into 2x2*2x6 + 2x1*1x6
        2x2*2x6: 3 Strassen 2x2x2 blocks, rank 21
        2x1*1x6: schoolbook, rank 12
        Total per 2x3*3x6: 33
      Two of these: 66
      
    A_right * B_bot: 4x2 * 2x6
      Split A_right into 2x2 + 2x2
      Each 2x2 * 2x6: 3 Strassen 2x2x2 blocks, rank 21
      Total: 42
      
    Grand total: 66 + 42 = 108. That's worse than 101.
    
    Let me try a different split:
    A (4x5) * B (5x6)
    Split B into B_1 (4x6) + B_2 (1x6)
    Split A into A_1 (4x4) + A_2 (4x1)
    
    A_1 * B_1: 4x4 * 4x6
      Split B_1 into B_1a (4x4) + B_1b (4x2)
      A_1 * B_1a: 4x4 * 4x4, Strassen, rank 49
      A_1 * B_1b: 4x4 * 4x2, Strassen, rank 28
      Total: 77
    
    A_2 * B_1: 4x1 * 1x6, schoolbook, rank 24
    A_2 * B_2: 4x1 * 1x6, schoolbook, rank 24
    Wait, B_2 is 1x6, so A_2 * B_2 is 4x1 * 1x6 = 4x6, rank 24.
    
    Total: 77 + 24 + 24 = 125. Worse.
    
    Let me stick with the original 101-rank approach but try to optimize.
    """
    # Use the 101-rank decomposition
    n, m, p = 4, 5, 6
    U_total = [[0]*101 for _ in range(n*m)]
    V_total = [[0]*101 for _ in range(m*p)]
    W_total = [[0]*101 for _ in range(p*n)]
    
    # Part 1: 4x4 * 4x4 Strassen, rank 49
    U1, V1, W1 = _build_strassen_4x4()
    for r in range(16):
        i = r // 4
        j = r % 4
        U_total[i*5+j][0:49] = U1[r]
    for r in range(16):
        j = r // 4
        k = r % 4
        V_total[j*6+k][0:49] = V1[r]
    for r in range(16):
        k = r // 4
        i = r % 4
        W_total[k*4+i][0:49] = W1[r]
    
    # Part 2: 4x4 * 4x2 Strassen, rank 28
    U2, V2, W2 = _build_strassen_4x4x2()
    for r in range(16):
        i = r // 4
        j = r % 4
        U_total[i*5+j][49:77] = U2[r]
    for r in range(8):
        j = r // 2
        k = r % 2
        V_total[j*6 + k + 4][49:77] = V2[r]
    for r in range(8):
        k = r // 4
        i = r % 4
        W_total[(k+4)*4 + i][49:77] = W2[r]
    
    # Part 3: 4x1 * 1x4 schoolbook, rank 16
    col_off = 77
    for i in range(4):
        for k in range(4):
            U_total[i*5+4][col_off + i*4 + k] = 1
            V_total[4*6 + k][col_off + i*4 + k] = 1
            W_total[k*4 + i][col_off + i*4 + k] = 1
    col_off += 16
    
    # Part 4: 4x1 * 1x2 schoolbook, rank 8
    for i in range(4):
        for k in range(2):
            U_total[i*5+4][col_off + i*2 + k] = 1
            V_total[4*6 + k + 4][col_off + i*2 + k] = 1
            W_total[(k+4)*4 + i][col_off + i*2 + k] = 1
    col_off += 8
    
    return U_total, V_total, W_total

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    try:
        U, V, W = _build_strassen_4x5_5x6()
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {s}")
        return {"U": U, "V": V, "W": W}
        
    except Exception:
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
