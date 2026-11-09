import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try Strassen-based decomposition for the 4x4 x 4x4 part
    # The idea: split A (4x5) into A_left (4x4) and A_right (4x1)
    # Split B (5x6) into B_top (4x6) and B_bottom (1x6)
    # C = A_left * B_top + A_right * B_bottom
    # For A_left (4x4) * B_top (4x6), we can use Strassen on the 4x4 x 4x4 part
    # and handle the extra 2 columns of B_top separately

    # Actually, let's think more carefully.
    # A is 4x5, B is 5x6.
    # We can write A = [A1, A2] where A1 is 4x4 and A2 is 4x1
    # B = [B1; B2] where B1 is 4x6 and B2 is 1x6
    # C = A1*B1 + A2*B2

    # For A1 (4x4) * B1 (4x6):
    # Split B1 = [B1a, B1b] where B1a is 4x4 and B1b is 4x2
    # A1*B1 = A1*B1a + A1*B1b
    # Use Strassen for A1*B1a (4x4 * 4x4) -> rank 49
    # Use schoolbook for A1*B1b (4x4 * 4x2) -> rank 4*4*2 = 32
    # A2*B2 (4x1 * 1x6) -> rank 4*1*6 = 24
    # Total: 49 + 32 + 24 = 105

    # Let's implement this.
    # Strassen for 4x4 * 4x4:
    # We need to decompose the 4x4 * 4x4 tensor using Strassen.
    # Strassen decomposes 2x2 * 2x2 with 7 multiplications.
    # For 4x4 * 4x4, we can split into four 2x2 blocks and apply Strassen.

    # Let's define the Strassen decomposition for 2x2 * 2x2.
    # For matrices X (2x2), Y (2x2):
    # M1 = (X11+X22)*(Y11+Y22)
    # M2 = (X21+X22)*Y11
    # M3 = X11*(Y12-Y22)
    # M4 = X22*(Y21-Y11)
    # M5 = (X11+X12)*Y22
    # M6 = (X21-X11)*(Y11+Y12)
    # M7 = (X12-X22)*(Y21+Y22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4 * 4x4, split X into 2x2 blocks:
    # X = [[X11, X12], [X21, X22]] where each is 2x2
    # Y = [[Y11, Y12], [Y21, Y22]] where each is 2x2
    # Then apply Strassen at the 2x2 block level.
    # But we need to be careful about how the entries map.

    # Actually, the standard way to apply Strassen to 4x4 is:
    # Treat the 4x4 matrices as 2x2 matrices of 2x2 blocks.
    # Apply the 7 Strassen formulas to these blocks.
    # This gives a rank-49 decomposition for the 4x4 * 4x4 multiplication.

    # Let's code the Strassen decomposition for 4x4 * 4x4.
    # The U, V, W for the 4x4 * 4x4 part will have dimensions:
    # U: 16 x 49 (for A1, which is 4x4)
    # V: 16 x 49 (for B1a, which is 4x4)
    # W: 16 x 49 (for the C part, which is 4x4)

    # First, let's build the Strassen decomposition for 2x2 * 2x2.
    # U_2x2: 4 x 7
    # V_2x2: 4 x 7
    # W_2x2: 4 x 7

    # Indices for 2x2: u[i*2+j], v[j*2+k], w[k*2+i]
    # Strassen terms:
    # M1: (X11+X22)*(Y11+Y22) -> u: [1,0,0,1], v: [1,0,0,1], w: [1,0,0,1]
    # M2: (X21+X22)*Y11 -> u: [0,0,1,1], v: [1,0,0,0], w: [0,0,1,0]
    # M3: X11*(Y12-Y22) -> u: [1,0,0,0], v: [0,1,0,-1], w: [0,1,0,0]
    # M4: X22*(Y21-Y11) -> u: [0,0,0,1], v: [0,0,1,-1], w: [0,0,0,1]
    # M5: (X11+X12)*Y22 -> u: [1,1,0,0], v: [0,0,0,1], w: [0,0,0,1]
    # M6: (X21-X11)*(Y11+Y12) -> u: [0,-1,1,0], v: [1,1,0,0], w: [0,0,1,0]
    # M7: (X12-X22)*(Y21+Y22) -> u: [0,1,0,-1], v: [0,0,1,1], w: [0,0,1,1]

    # Wait, let me re-derive the W vectors for 2x2 Strassen.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # W indices: w[k*2+i]
    # C11: i=0,k=0 -> w[0]
    # C12: i=0,k=1 -> w[1]
    # C21: i=1,k=0 -> w[2]
    # C22: i=1,k=1 -> w[3]

    # So:
    # M1 contributes to C11 and C22: w = [1, 0, 0, 1]
    # M2 contributes to C21: w = [0, 0, 1, 0]
    # M3 contributes to C12 and C22: w = [0, 1, 0, 1]
    # M4 contributes to C11 and C21: w = [1, 0, 1, 0]
    # M5 contributes to C12 and C22: w = [0, 1, 0, 1] ... wait, M5 is -C11 + C22
    # Let me re-check.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # So:
    # M1: C11 += 1, C22 += 1 -> w = [1, 0, 0, 1]
    # M2: C21 += 1, C22 -= 1 -> w = [0, 0, 1, -1]
    # M3: C12 += 1, C22 += 1 -> w = [0, 1, 0, 1]
    # M4: C11 += 1, C21 += 1 -> w = [1, 0, 1, 0]
    # M5: C11 -= 1, C12 += 1 -> w = [-1, 1, 0, 0]
    # M6: C22 += 1 -> w = [0, 0, 0, 1]
    # M7: C11 += 1 -> w = [1, 0, 0, 0]

    # Let me verify with a simple example.
    # X = [[1,2],[3,4]], Y = [[5,6],[7,8]]
    # C = [[19, 22], [43, 50]]
    # M1 = (1+4)*(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 7*5 = 35
    # M3 = 1*(6-8) = -2
    # M4 = 4*(7-5) = 8
    # M5 = (1+2)*8 = 24
    # M6 = (3-1)*(5+6) = 2*11 = 22
    # M7 = (2-4)*(7+8) = -2*15 = -30
    # C11 = M1 + M4 - M5 + M7 = 65 + 8 - 24 + (-30) = 19 ✓
    # C12 = M3 + M5 = -2 + 24 = 22 ✓
    # C21 = M2 + M4 = 35 + 8 = 43 ✓
    # C22 = M1 - M2 + M3 + M6 = 65 - 35 + (-2) + 22 = 50 ✓

    # So the W vectors are:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]

    # Now for 4x4 * 4x4, we split into 2x2 blocks.
    # X (4x4) = [[X11, X12], [X21, X22]] where each Xi is 2x2
    # Y (4x4) = [[Y11, Y12], [Y21, Y22]] where each Yj is 2x2
    # We apply Strassen to these 2x2 blocks.
    # The U vector for X has 16 entries (4x4 flattened).
    # The V vector for Y has 16 entries.
    # The W vector for C has 16 entries.

    # For each Strassen term (e.g., M1 = (X11+X22)*(Y11+Y22)):
    # The U vector has non-zero entries for X11 and X22 blocks.
    # X11 is at positions (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # X22 is at positions (2,0),(2,1),(3,0),(3,1) -> indices 8,9,10,11
    # So U_M1 = [1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0]
    # Similarly for V_M1 = [1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0]
    # W_M1: C11 and C22 blocks.
    # C11 is at positions (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # C22 is at positions (2,0),(2,1),(3,0),(3,1) -> indices 8,9,10,11
    # W_M1 = [1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0]

    # This is getting complex. Let me just hardcode the 49 terms for 4x4*4x4.
    # Actually, let me generate them programmatically.

    # Define the 2x2 Strassen U, V, W vectors.
    # u_2x2: 4 x 7
    # v_2x2: 4 x 7
    # w_2x2: 4 x 7

    u_2x2 = np.zeros((4, 7))
    v_2x2 = np.zeros((4, 7))
    w_2x2 = np.zeros((4, 7))

    # M1: (X11+X22)*(Y11+Y22)
    u_2x2[0, 0] = 1; u_2x2[3, 0] = 1  # X11=idx0, X22=idx3
    v_2x2[0, 0] = 1; v_2x2[3, 0] = 1  # Y11=idx0, Y22=idx3
    w_2x2[0, 0] = 1; w_2x2[3, 0] = 1  # C11=idx0, C22=idx3

    # M2: (X21+X22)*Y11
    u_2x2[2, 1] = 1; u_2x2[3, 1] = 1  # X21=idx2, X22=idx3
    v_2x2[0, 1] = 1  # Y11=idx0
    w_2x2[2, 1] = 1; w_2x2[3, 1] = -1  # C21=idx2, C22=idx3

    # M3: X11*(Y12-Y22)
    u_2x2[0, 2] = 1  # X11=idx0
    v_2x2[1, 2] = 1; v_2x2[3, 2] = -1  # Y12=idx1, Y22=idx3
    w_2x2[1, 2] = 1; w_2x2[3, 2] = 1  # C12=idx1, C22=idx3

    # M4: X22*(Y21-Y11)
    u_2x2[3, 3] = 1  # X22=idx3
    v_2x2[2, 3] = 1; v_2x2[0, 3] = -1  # Y21=idx2, Y11=idx0
    w_2x2[0, 3] = 1; w_2x2[2, 3] = 1  # C11=idx0, C21=idx2

    # M5: (X11+X12)*Y22
    u_2x2[0, 4] = 1; u_2x2[1, 4] = 1  # X11=idx0, X12=idx1
    v_2x2[3, 4] = 1  # Y22=idx3
    w_2x2[0, 4] = -1; w_2x2[1, 4] = 1  # C11=idx0, C12=idx1

    # M6: (X21-X11)*(Y11+Y12)
    u_2x2[2, 5] = 1; u_2x2[0, 5] = -1  # X21=idx2, X11=idx0
    v_2x2[0, 5] = 1; v_2x2[1, 5] = 1  # Y11=idx0, Y12=idx1
    w_2x2[3, 5] = 1  # C22=idx3

    # M7: (X12-X22)*(Y21+Y22)
    u_2x2[1, 6] = 1; u_2x2[3, 6] = -1  # X12=idx1, X22=idx3
    v_2x2[2, 6] = 1; v_2x2[3, 6] = 1  # Y21=idx2, Y22=idx3
    w_2x2[0, 6] = 1  # C11=idx0

    # Now, for 4x4 * 4x4, we split into 2x2 blocks.
    # The 4x4 matrix is split into:
    # Block (0,0): rows 0-1, cols 0-1 -> indices 0-3 in flattened 4x4
    # Block (0,1): rows 0-1, cols 2-3 -> indices 4-7
    # Block (1,0): rows 2-3, cols 0-1 -> indices 8-11
    # Block (1,1): rows 2-3, cols 2-3 -> indices 12-15

    # For each Strassen term, we need to construct the 16-dimensional U, V, W vectors.
    # The U vector for the 4x4 X matrix:
    # For M1: (X11+X22) -> X11 is block (0,0), X22 is block (1,1)
    # So U_M1 has 1s at positions corresponding to block (0,0) and block (1,1).

    # Let's define the block positions in the flattened 4x4 matrix.
    # Block (r, c) where r,c in {0,1}:
    # Positions: (r*2+i)*4 + (c*2+j) for i,j in {0,1}
    # Block (0,0): 0,1,4,5
    # Block (0,1): 2,3,6,7
    # Block (1,0): 8,9,12,13
    # Block (1,1): 10,11,14,15

    # Wait, that's not right. Let me re-index.
    # Flattened 4x4: index = row*4 + col
    # Block (0,0): rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block (0,1): rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block (1,0): rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block (1,1): rows 2-3, cols 2-3 -> indices 10,11,14,15

    # For the 2x2 block matrices:
    # X11 (block 0,0): indices 0,1,4,5
    # X12 (block 0,1): indices 2,3,6,7
    # X21 (block 1,0): indices 8,9,12,13
    # X22 (block 1,1): indices 10,11,14,15

    # Within each 2x2 block, the local indices are:
    # (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3
    # So for block (r,c), the global index for local (i,j) is:
    # (r*2+i)*4 + (c*2+j)

    # Let's create a helper function to get the global index.
    def block_idx(r, c, i, j):
        return (r * 2 + i) * 4 + (c * 2 + j)

    # Now, for each Strassen term, construct the 16-dim U, V, W vectors.
    u_4x4 = np.zeros((16, 7))
    v_4x4 = np.zeros((16, 7))
    w_4x4 = np.zeros((16, 7))

    # M1: (X11+X22)*(Y11+Y22)
    # X11: block (0,0), X22: block (1,1)
    for i in range(2):
        for j in range(2):
            u_4x4[block_idx(0, 0, i, j), 0] = 1
            u_4x4[block_idx(1, 1, i, j), 0] = 1
            v_4x4[block_idx(0, 0, i, j), 0] = 1
            v_4x4[block_idx(1, 1, i, j), 0] = 1
            w_4x4[block_idx(0, 0, i, j), 0] = 1
            w_4x4[block_idx(1, 1, i, j), 0] = 1

    # M2: (X21+X22)*Y11
    # X21: block (1,0), X22: block (1,1), Y11: block (0,0)
    for i in range(2):
        for j in range(2):
            u_4x4[block_idx(1, 0, i, j), 1] = 1
            u_4x4[block_idx(1, 1, i, j), 1] = 1
            v_4x4[block_idx(0, 0, i, j), 1] = 1
    # W: C21 and C22
    for i in range(2):
        for j in range(2):
            w_4x4[block_idx(1, 0, i, j), 1] = 1
            w_4x4[block_idx(1, 1, i, j), 1] = -1

    # M3: X11*(Y12-Y22)
    # X11: block (0,0), Y12: block (0,1), Y22: block (1,1)
    for i in range(2):
        for j in range(2):
            u_4x4[block_idx(0, 0, i, j), 2] = 1
            v_4x4[block_idx(0, 1, i, j), 2] = 1
            v_4x4[block_idx(1, 1, i, j), 2] = -1
    # W: C12 and C22
    for i in range(2):
        for j in range(2):
            w_4x4[block_idx(0, 1, i, j), 2] = 1
            w_4x4[block_idx(1, 1, i, j), 2] = 1

    # M4: X22*(Y21-Y11)
    # X22: block (1,1), Y21: block (1,0), Y11: block (0,0)
    for i in range(2):
        for j in range(2):
            u_4x4[block_idx(1, 1, i, j), 3] = 1
            v_4x4[block_idx(1, 0, i, j), 3] = 1
            v_4x4[block_idx(0, 0, i, j), 3] = -1
    # W: C11 and C21
    for i in range(2):
        for j in range(2):
            w_4x4[block_idx(0, 0, i, j), 3] = 1
            w_4x4[block_idx(1, 0, i, j), 3] = 1

    # M5: (X11+X12)*Y22
    # X11: block (0,0), X12: block (0,1), Y22: block (1,1)
    for i in range(2):
        for j in range(2):
            u_4x4[block_idx(0, 0, i, j), 4] = 1
            u_4x4[block_idx(0, 1, i, j), 4] = 1
            v_4x4[block_idx(1, 1, i, j), 4] = 1
    # W: C11 and C12
    for i in range(2):
        for j in range(2):
            w_4x4[block_idx(0, 0, i, j), 4] = -1
            w_4x4[block_idx(0, 1, i, j), 4] = 1

    # M6: (X21-X11)*(Y11+Y12)
    # X21: block (1,0), X11: block (0,0), Y11: block (0,0), Y12: block (0,1)
    for i in range(2):
        for j in range(2):
            u_4x4[block_idx(1, 0, i, j), 5] = 1
            u_4x4[block_idx(0, 0, i, j), 5] = -1
            v_4x4[block_idx(0, 0, i, j), 5] = 1
            v_4x4[block_idx(0, 1, i, j), 5] = 1
    # W: C22
    for i in range(2):
        for j in range(2):
            w_4x4[block_idx(1, 1, i, j), 5] = 1

    # M7: (X12-X22)*(Y21+Y22)
    # X12: block (0,1), X22: block (1,1), Y21: block (1,0), Y22: block (1,1)
    for i in range(2):
        for j in range(2):
            u_4x4[block_idx(0, 1, i, j), 6] = 1
            u_4x4[block_idx(1, 1, i, j), 6] = -1
            v_4x4[block_idx(1, 0, i, j), 6] = 1
            v_4x4[block_idx(1, 1, i, j), 6] = 1
    # W: C11
    for i in range(2):
        for j in range(2):
            w_4x4[block_idx(0, 0, i, j), 6] = 1

    # Now, let's build the full decomposition.
    # A is 4x5, B is 5x6.
    # A = [A1, A2] where A1 is 4x4 (cols 0-3), A2 is 4x1 (col 4)
    # B = [B1; B2] where B1 is 4x6 (rows 0-3), B2 is 1x6 (row 4)
    # C = A1*B1 + A2*B2

    # B1 is 4x6, split into B1a (4x4, cols 0-3) and B1b (4x2, cols 4-5)
    # C = A1*B1a + A1*B1b + A2*B2

    # A1*B1a: 4x4 * 4x4, use Strassen (rank 49)
    # A1*B1b: 4x4 * 4x2, schoolbook (rank 32)
    # A2*B2: 4x1 * 1x6, schoolbook (rank 24)
    # Total rank: 49 + 32 + 24 = 105

    total_rank = 49 + 32 + 24
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Part 1: A1*B1a (Strassen, rank 49)
    # A1 is 4x4, indices in A: row i (0-3), col j (0-3) -> U index i*5+j
    # B1a is 4x4, indices in B: row j (0-3), col k (0-3) -> V index j*6+k
    # C part is 4x4, indices in C: row i (0-3), col k (0-3) -> W index k*4+i

    for term in range(7):
        for i in range(4):
            for j in range(4):
                u_val = u_4x4[i * 4 + j, term]
                if u_val != 0:
                    U[i * 5 + j][term] = u_val

        for j in range(4):
            for k in range(4):
                v_val = v_4x4[j * 4 + k, term]
                if v_val != 0:
                    V[j * 6 + k][term] = v_val

        for i in range(4):
            for k in range(4):
                w_val = w_4x4[k * 4 + i, term]
                if w_val != 0:
                    W[k * 4 + i][term] = w_val

    # Part 2: A1*B1b (schoolbook, rank 32)
    # A1 is 4x4, B1b is 4x2 (cols 4-5)
    # For each i (0-3), j (0-3), k (4-5):
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * 5 + j][col_offset] = 1
                V[j * 6 + k][col_offset] = 1
                W[k * 4 + i][col_offset] = 1
                col_offset += 1

    # Part 3: A2*B2 (schoolbook, rank 24)
    # A2 is 4x1 (col 4), B2 is 1x6 (row 4)
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][col_offset] = 1
            V[4 * 6 + k][col_offset] = 1
            W[k * 4 + i][col_offset] = 1
            col_offset += 1

    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Check tensor decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if i,j,k are valid indices for C[i,k] = sum_j A[i,j]*B[j,k]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.

    # Let's verify with random matrices
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_true = A_test @ B_test

        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(total_rank):
                    for j in range(m):
                        total += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                C_decomp[i, k] = total

        error = np.max(np.abs(C_true - C_decomp))
        if error > 1e-9:
            # Fall back to schoolbook
            return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
