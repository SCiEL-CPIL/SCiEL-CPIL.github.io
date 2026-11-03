"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = 140.0
    start_time = time.time()

    # Strategy: Treat 2x4 * 4x8 as (4x2)^T * (4x8), i.e., a 4x8 * 8x4 product
    # after transposition, or more directly:
    # A is 2x4, B is 4x8, C is 2x8.
    # 
    # The directive suggests: transpose A to 4x2, permute C to 8x2,
    # then view as 4x8 * 8x2 (which is the same as 2x8 * 8x4 transposed).
    # Actually, let's think about it as: C^T = B^T A^T, so C^T is 8x2.
    # B^T is 8x4, A^T is 4x2. So C^T = B^T * A^T is 8x2 = (8x4)*(4x2).
    #
    # Alternatively, we can view the original problem C = A*B where A is 2x4 and B is 4x8.
    # Let's stick with the current approach but try to see if we can do better.
    #
    # The current approach uses 8 independent Strassen 2x2 decompositions giving rank 56.
    #
    # Let's try a different block structure. Split A into 2 rows (each 1x4) and B into 4 rows (each 1x8).
    # That doesn't help directly.
    #
    # Actually, let's consider: A is 2x4. We can split A into two 1x4 blocks: A0 (row 0) and A1 (row 1).
    # B is 4x8. We can split B into four 1x8 blocks: B0, B1, B2, B3 (rows).
    # C = A0*B0^T + A0*B1^T + ... no, that's not right.
    #
    # C[i,j] = sum_k A[i,k] * B[k,j]
    #
    # Let's try: split A's 4 columns into two groups of 2: A_left (2x2), A_right (2x2).
    # Split B's 4 rows into two groups of 2: B_top (2x8), B_bottom (2x8).
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each term is a 2x2 * 2x8 product. Split B_top into four 2x2 blocks: B_top[0..3]
    # B_bottom into four 2x2 blocks: B_bottom[0..3]
    # C = A_left * [B_top_0, B_top_1, B_top_2, B_top_3] + A_right * [B_bot_0, B_bot_1, B_bot_2, B_bot_3]
    #
    # So we have 8 independent 2x2 * 2x2 products. Each can be done with Strassen rank 7.
    # Total rank = 8 * 7 = 56.
    #
    # This is what the current code does. Let me verify the indexing is correct.
    #
    # A is 2x4. Flattened (row-major): A[i*4 + k] for i in {0,1}, k in {0,1,2,3}.
    # U has shape (8, rank). Row index: i*4 + k.
    #
    # In the Strassen decomposition of a 2x2 * 2x2 product:
    # For block (a, b) where a in {0,1} (row group of A) and b in {0,1,2,3} (column block of B):
    # A_block is rows 0,1 of A, columns 2*(b//2) ... wait, let me re-read the code.
    #
    # Looking at the code:
    # for a in range(2):  # a is 0 or 1, corresponding to A's row group (but A only has 2 rows!)
    #   for b in range(4):  # b is 0..3, corresponding to B's column blocks
    #
    # Wait, A is 2x4. It has 2 rows. So 'a' in range(2) corresponds to the two rows of A?
    # No, that doesn't make sense for a 2x2 block.
    #
    # Let me re-examine. The code says:
    # U[0 * m + 2 * a, t]  where m=4, so row = 2*a
    # U[1 * m + 2 * a + 1, t]  where row = 4 + 2*a + 1 = 5 + 2*a
    #
    # For a=0: rows 0 and 5. For a=1: rows 2 and 7.
    # A flattened: row 0: indices 0,1,2,3. Row 1: indices 4,5,6,7.
    # So for a=0: U uses A[0,0] and A[1,1] (indices 0 and 5)
    # For a=1: U uses A[0,2] and A[1,3] (indices 2 and 7)
    #
    # V[2 * a * p + 2 * b, t]  where p=8, so row = 16*a + 2*b
    # V[(2 * a + 1) * p + 2 * b + 1, t]  where row = (2a+1)*8 + 2b + 1
    #
    # B is 4x8. Flattened: B[k*8 + j] for k in {0,1,2,3}, j in {0..7}.
    # For a=0, b=0: V uses B[0,0] (row 0) and B[1,1] (row 9)
    # For a=0, b=1: V uses B[0,2] (row 2) and B[1,3] (row 11)
    # etc.
    #
    # W[2 * b * n + 0, t]  where n=2, so row = 4*b
    # W[(2 * b + 1) * n + 1, t]  where row = (2b+1)*2 + 1 = 4b + 3
    #
    # C is 2x8. Flattened: C[i*8 + j] for i in {0,1}, j in {0..7}.
    # W row index corresponds to C flattened index.
    # For b=0: W uses C[0,0] (row 0) and C[1,1] (row 3)
    # For b=1: W uses C[0,2] (row 4) and C[1,3] (row 7)
    # etc.
    #
    # This looks like a valid decomposition. The rank is 56.
    #
    # Let me just return the same construction, cleaned up.

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    t = 0
    for a in range(2):
        for b in range(4):
            for s in range(7):
                if s == 0:
                    U[0 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                    V[2 * a * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                    W[2 * b * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 1:
                    U[1 * m + 2 * a, t] += 1
                    U[1 * m + 2 * a + 1, t] += 1
                    V[2 * a * p + 2 * b, t] += 1
                    W[2 * b * n + 1, t] += 1
                    W[(2 * b + 1) * n + 1, t] += -1
                elif s == 2:
                    U[0 * m + 2 * a, t] += 1
                    V[2 * a * p + 2 * b + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += -1
                    W[(2 * b + 1) * n + 0, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 3:
                    U[1 * m + 2 * a + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b, t] += -1
                    W[2 * b * n + 0, t] += 1
                    W[2 * b * n + 1, t] += 1
                elif s == 4:
                    U[0 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a + 1, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                    W[2 * b * n + 0, t] += -1
                    W[(2 * b + 1) * n + 0, t] += 1
                elif s == 5:
                    U[1 * m + 2 * a, t] += 1
                    U[0 * m + 2 * a, t] += -1
                    V[2 * a * p + 2 * b, t] += 1
                    V[2 * a * p + 2 * b + 1, t] += 1
                    W[(2 * b + 1) * n + 1, t] += 1
                elif s == 6:
                    U[0 * m + 2 * a + 1, t] += 1
                    U[1 * m + 2 * a + 1, t] += -1
                    V[(2 * a + 1) * p + 2 * b, t] += 1
                    V[(2 * a + 1) * p + 2 * b + 1, t] += 1
                    W[2 * b * n + 0, t] += 1
                t += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
