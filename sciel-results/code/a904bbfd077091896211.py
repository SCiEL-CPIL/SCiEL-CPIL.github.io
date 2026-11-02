"""Strassen-based rank-28 decomposition for 2x4 * 4x4 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 4]

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS
    # We decompose into 4 independent 2x2*2x2 multiplications using Strassen (rank 7 each)
    # Block structure:
    #   A is 2x4: A = [A_L, A_R] where A_L, A_R are 2x2
    #   B is 4x4: B = [[B_LL, B_LR], [B_BL, B_BR]] each 2x2
    #   C = A * B = [A_L*B_LL + A_R*B_BL, A_L*B_LR + A_R*B_BR]
    #   C_L = A_L*B_LL + A_R*B_BL  (2x2)
    #   C_R = A_L*B_LR + A_R*B_BR  (2x2)
    #
    # We use Strassen on 4 blocks:
    #   Block 0: A_L * B_LL  -> contributes to C_L
    #   Block 1: A_L * B_LR  -> contributes to C_R
    #   Block 2: A_R * B_BL  -> contributes to C_L
    #   Block 3: A_R * B_BR  -> contributes to C_R
    #
    # Each block uses rank-7 Strassen, total rank = 4 * 7 = 28

    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28

    # Strassen rank-7 algorithm for 2x2 * 2x2:
    # Given X (2x2) and Y (2x2), compute Z = X*Y (2x2) using 7 multiplications.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    #
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y01+y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    # For each block, we define:
    #   - The 4 local (row, col) positions in A that form the 2x2 submatrix
    #   - The 4 local (row, col) positions in B that form the 2x2 submatrix
    #   - The 4 local (row, col) positions in C that receive the result

    # Block 0: A_L * B_LL -> C[0:2, 0:2]
    # A_L: rows 0-1 of A, cols 0-1 -> A indices: (0,0),(0,1),(1,0),(1,1)
    #   In U, A is 2x4 with index i*m+j = i*4+j
    #   A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
    # B_LL: B[0:2, 0:2] -> B indices: (0,0),(0,1),(1,0),(1,1)
    #   In V, B is 4x4 with index i*4+j
    #   B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
    # C result: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    #   W index: k*n+i = k*2+i
    #   C[0,0]: k=0,i=0 -> 0
    #   C[0,1]: k=1,i=0 -> 2
    #   C[1,0]: k=0,i=1 -> 1
    #   C[1,1]: k=1,i=1 -> 3

    # Block 1: A_L * B_LR -> C[0:2, 2:4]
    # A_L: same as Block 0
    # B_LR: B[0:2, 2:4] -> B indices: (0,2),(0,3),(1,2),(1,3)
    #   B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[6], B[1,3]=V[7]
    # C result: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    #   C[0,2]: k=2,i=0 -> 4
    #   C[0,3]: k=3,i=0 -> 6
    #   C[1,2]: k=2,i=1 -> 5
    #   C[1,3]: k=3,i=1 -> 7

    # Block 2: A_R * B_BL -> C[0:2, 0:2]
    # A_R: A[0:2, 2:4] -> A indices: (0,2),(0,3),(1,2),(1,3)
    #   A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    # B_BL: B[2:4, 0:2] -> B indices: (2,0),(2,1),(3,0),(3,1)
    #   B[2,0]=V[8], B[2,1]=V[9], B[3,0]=V[12], B[3,1]=V[13]
    # C result: same as Block 0: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]

    # Block 3: A_R * B_BR -> C[0:2, 2:4]
    # A_R: same as Block 2
    # B_BR: B[2:4, 2:4] -> B indices: (2,2),(2,3),(3,2),(3,3)
    #   B[2,2]=V[10], B[2,3]=V[11], B[3,2]=V[14], B[3,3]=V[15]
    # C result: same as Block 1: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]

    # Define Strassen coefficients
    # For each of 7 multiplications, we have:
    #   coeff_A[r] for r in 0..3 (local row/col mapping to x00,x01,x10,x11)
    #   coeff_B[c] for c in 0..3 (local row/col mapping to y00,y01,y10,y11)
    #   coeff_Z[z] for z in 0..3 (local result mapping to Z00,Z01,Z10,Z11)
    #
    # Local A indices: 0=x00, 1=x01, 2=x10, 3=x11
    # Local B indices: 0=y00, 1=y01, 2=y10, 3=y11
    # Local Z indices: 0=Z00, 1=Z01, 2=Z10, 3=Z11

    strassen_terms = [
        # M1: (x00+x11)*(y00+y11)
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'Z': {0: 1, 3: 1}},
        # M2: (x10+x11)*y00
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'Z': {2: 1, 3: -1}},
        # M3: x00*(y01-y11)
        {'A': {0: 1}, 'B': {1: 1, 3: -1}, 'Z': {0: -1, 1: 1, 3: 1}},
        # M4: x11*(y10-y00)
        {'A': {3: 1}, 'B': {2: 1, 0: -1}, 'Z': {0: 1, 2: 1}},
        # M5: (x00+x01)*y11
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'Z': {0: -1, 1: 1}},
        # M6: (x10-x00)*(y00+y10)
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 2: 1}, 'Z': {3: 1}},
        # M7: (x01-x11)*(y01+y11)
        {'A': {1: 1, 3: -1}, 'B': {1: 1, 3: 1}, 'Z': {0: 1}},
    ]

    # Block definitions: (A_local_map, B_local_map, Z_local_map)
    # A_local_map: list of 4 U indices for x00,x01,x10,x11
    # B_local_map: list of 4 V indices for y00,y01,y10,y11
    # Z_local_map: list of 4 W indices for Z00,Z01,Z10,Z11

    blocks = [
        # Block 0: A_L * B_LL
        (
            [0, 1, 4, 5],    # A_L: A[0,0],A[0,1],A[1,0],A[1,1]
            [0, 1, 4, 5],    # B_LL
            [0, 2, 1, 3],    # C[0,0],C[0,1],C[1,0],C[1,1]
        ),
        # Block 1: A_L * B_LR
        (
            [0, 1, 4, 5],    # A_L
            [2, 3, 6, 7],    # B_LR: B[0,2],B[0,3],B[1,2],B[1,3]
            [4, 6, 5, 7],    # C[0,2],C[0,3],C[1,2],C[1,3]
        ),
        # Block 2: A_R * B_BL
        (
            [2, 3, 6, 7],    # A_R: A[0,2],A[0,3],A[1,2],A[1,3]
            [8, 9, 12, 13],  # B_BL: B[2,0],B[2,1],B[3,0],B[3,1]
            [0, 2, 1, 3],    # C[0,0],C[0,1],C[1,0],C[1,1]
        ),
        # Block 3: A_R * B_BR
        (
            [2, 3, 6, 7],    # A_R
            [10, 11, 14, 15],# B_BR: B[2,2],B[2,3],B[3,2],B[3,3]
            [4, 6, 5, 7],    # C[0,2],C[0,3],C[1,2],C[1,3]
        ),
    ]

    col = 0
    for block_idx in range(4):
        if time.time() > deadline:
            break
        a_map, b_map, z_map = blocks[block_idx]
        for term in strassen_terms:
            if time.time() > deadline:
                break
            # Fill U, V, W for this column
            for local_a, coeff in term['A'].items():
                u_idx = a_map[local_a]
                U[u_idx][col] = coeff
            for local_b, coeff in term['B'].items():
                v_idx = b_map[local_b]
                V[v_idx][col] = coeff
            for local_z, coeff in term['Z'].items():
                w_idx = z_map[local_z]
                W[w_idx][col] = coeff
            col += 1

    return {"U": U, "V": V, "W": W}
