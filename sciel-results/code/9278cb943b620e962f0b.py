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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS  # 4, 5, 6

    # We want to compute C = A * B where A is n x m (4x5) and B is m x p (5x6).
    # We can decompose this into blocks.
    # Let's split A into A_top (4x4) and A_bottom (4x1).
    # Let's split B into B_left (4x6) and B_right (1x6).
    # C = A_top * B_left + A_bottom * B_right
    
    # A_top is 4x4. B_left is 4x6.
    # We can compute A_top * B_left by splitting B_left into B_left1 (4x4) and B_left2 (4x2).
    # A_top * B_left = A_top * B_left1 + A_top * B_left2
    
    # So we have three subproblems:
    # 1. A_top (4x4) * B_left1 (4x4) -> C1 (4x4)
    # 2. A_top (4x4) * B_left2 (4x2) -> C2 (4x2)
    # 3. A_bottom (4x1) * B_right (1x6) -> C3 (4x6)
    
    # Subproblem 1: 4x4 * 4x4 -> 4x4. Use Strassen 4x4x4. Rank 49.
    # Subproblem 2: 4x4 * 4x2 -> 4x2. Use Strassen 4x4x2. Rank 28.
    # Subproblem 3: 4x1 * 1x6 -> 4x6. This is an outer product. Rank 1? No, rank 4*1*6 = 24? 
    # Wait, A_bottom is 4x1, B_right is 1x6. The product is a 4x6 matrix.
    # The tensor rank of the outer product of vectors is 1.
    # C3[i,k] = sum_j A_bottom[i,j] * B_right[j,k] = A_bottom[i,0] * B_right[0,k].
    # This is a rank 1 tensor! 
    # U: 4x1 (A_bottom indices), V: 1x6 (B_right indices), W: 6x4 (C3 indices).
    # U[i,0] = 1, V[0,k] = 1, W[k,i] = 1.
    # Wait, the tensor is T[i,j,k] = A[i,j] B[j,k].
    # For the outer product part: A_bottom[i, 0] * B_right[0, k].
    # The indices are i (0..3), j (0), k (0..5).
    # U[i, col] = 1. V[0, col] = 1. W[k, i] -> W[k*4+i, col] = 1.
    # So rank 1.
    
    # Total rank = 49 + 28 + 1 = 78.
    # Schoolbook rank = 4*5*6 = 120.
    # Current best score 1.1881 corresponds to rank 101? 
    # 120 / 1.1881 = 101.
    # My proposed rank is 78. 120 / 78 = 1.538. This is a huge improvement!
    
    # Let's verify the dimensions and indices carefully.
    # A is 4x5. Indices i in [0,3], j in [0,4].
    # B is 5x6. Indices j in [0,4], k in [0,5].
    # C is 4x6. Indices i in [0,3], k in [0,5].
    
    # Split j into j1 in [0,3] and j2 = 4.
    # A_top: i in [0,3], j1 in [0,3]. (4x4)
    # A_bottom: i in [0,3], j2 = 4. (4x1)
    # B_left: j1 in [0,3], k in [0,5]. (4x6)
    # B_right: j2 = 4, k in [0,5]. (1x6)
    
    # C = sum_{j=0}^4 A[i,j] B[j,k]
    #   = sum_{j1=0}^3 A[i,j1] B[j1,k] + A[i,4] B[4,k]
    #   = (A_top * B_left)[i,k] + A_bottom[i,0] * B_right[0,k]
    
    # Now split B_left (4x6) into B_left1 (4x4) and B_left2 (4x2).
    # B_left1: j1 in [0,3], k1 in [0,3].
    # B_left2: j1 in [0,3], k2 in [4,5].
    
    # C = A_top * (B_left1 + B_left2) + A_bottom * B_right
    #   = A_top * B_left1 + A_top * B_left2 + A_bottom * B_right
    
    # Subproblem 1: A_top (4x4) * B_left1 (4x4) -> C1 (4x4).
    #   Indices: i in [0,3], j1 in [0,3], k1 in [0,3].
    #   Strassen 4x4x4 gives U1 (16x49), V1 (16x49), W1 (16x49).
    #   U1 row index: i*4 + j1.
    #   V1 row index: j1*4 + k1.
    #   W1 row index: k1*4 + i.
    
    # Subproblem 2: A_top (4x4) * B_left2 (4x2) -> C2 (4x2).
    #   Indices: i in [0,3], j1 in [0,3], k2 in [4,5].
    #   Strassen 4x4x2 gives U2 (16x28), V2 (8x28), W2 (8x28).
    #   U2 row index: i*4 + j1.
    #   V2 row index: j1*2 + (k2-4).  # Since B_left2 is 4x2, local k is 0,1.
    #   W2 row index: (k2-4)*4 + i.   # Since C2 is 4x2, local k is 0,1.
    
    # Subproblem 3: A_bottom (4x1) * B_right (1x6) -> C3 (4x6).
    #   Indices: i in [0,3], j2=4, k in [0,5].
    #   Rank 1.
    #   U3 row index: i*1 + 0 = i. (4x1)
    #   V3 row index: 0*6 + 0 = 0. (1x6) -> Wait, V shape is (m*p, rank).
    #   In the full problem, V has shape (5*6, rank).
    #   For this subproblem, the relevant row in V is for j=4, k.
    #   V_full row index: 4*6 + k = 24 + k.
    #   U_full row index: i*5 + 4.
    #   W_full row index: k*4 + i.
    
    # Let's assemble the full U, V, W.
    # Full U shape: (4*5, rank) = (20, rank).
    # Full V shape: (5*6, rank) = (30, rank).
    # Full W shape: (6*4, rank) = (24, rank).
    
    # Rank 1: 49 cols.
    # Rank 2: 28 cols.
    # Rank 3: 1 col.
    # Total rank: 78.
    
    U1, V1, W1 = _build_strassen_4x4()
    U2, V2, W2 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 1
    U = [[0]*total_rank for _ in range(20)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Fill Subproblem 1 (Strassen 4x4x4)
    # U1[i*4+j1][c] -> U[i*5+j1][c]
    # V1[j1*4+k1][c] -> V[j1*6+k1][c]
    # W1[k1*4+i][c] -> W[k1*4+i][c]
    for c in range(49):
        for i in range(4):
            for j1 in range(4):
                u_row_local = i*4 + j1
                u_row_global = i*5 + j1
                val = U1[u_row_local][c]
                if val != 0:
                    U[u_row_global][c] = val
        
        for j1 in range(4):
            for k1 in range(4):
                v_row_local = j1*4 + k1
                v_row_global = j1*6 + k1
                val = V1[v_row_local][c]
                if val != 0:
                    V[v_row_global][c] = val
        
        for k1 in range(4):
            for i in range(4):
                w_row_local = k1*4 + i
                w_row_global = k1*4 + i
                val = W1[w_row_local][c]
                if val != 0:
                    W[w_row_global][c] = val
    
    # Fill Subproblem 2 (Strassen 4x4x2)
    # U2[i*4+j1][c] -> U[i*5+j1][49+c]
    # V2[j1*2+k2_local][c] -> V[j1*6+(4+k2_local)][49+c]
    # W2[k2_local*4+i][c] -> W[(4+k2_local)*4+i][49+c]
    for c in range(28):
        for i in range(4):
            for j1 in range(4):
                u_row_local = i*4 + j1
                u_row_global = i*5 + j1
                val = U2[u_row_local][c]
                if val != 0:
                    U[u_row_global][49+c] = val
        
        for j1 in range(4):
            for k2_local in range(2):
                v_row_local = j1*2 + k2_local
                k2_global = 4 + k2_local
                v_row_global = j1*6 + k2_global
                val = V2[v_row_local][c]
                if val != 0:
                    V[v_row_global][49+c] = val
        
        for k2_local in range(2):
            for i in range(4):
                w_row_local = k2_local*4 + i
                k2_global = 4 + k2_local
                w_row_global = k2_global*4 + i
                val = W2[w_row_local][c]
                if val != 0:
                    W[w_row_global][49+c] = val
    
    # Fill Subproblem 3 (Outer Product)
    # Rank 1, col = 77.
    # U[i*5+4][77] = 1
    # V[4*6+k][77] = 1
    # W[k*4+i][77] = 1
    for i in range(4):
        U[i*5 + 4][77] = 1
    for k in range(6):
        V[4*6 + k][77] = 1
    for i in range(4):
        for k in range(6):
            W[k*4 + i][77] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
