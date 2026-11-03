"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=4, m=4, p=8

    # Build the multiplication tensor T[a, b, c]
    # a = i*m + j (index into A: n x m = 4x4 = 16)
    # b = j*p + k (index into B: m x p = 4x8 = 32)
    # c = k*n + i (index into C: p x n = 8x4 = 32)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook rank = n*m*p = 128
    schoolbook_rank = n * m * p

    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Strategy: Build 4x4x4 Strassen rank-49 decomposition, then extend to 4x4x8
    # by sharing the U (A-side) factors between the two 4x4 B-blocks.
    #
    # The key insight: C = A * B where B is 4x8.
    # Split B = [B1, B2] where B1, B2 are 4x4.
    # C = [A*B1, A*B2] but with the trace convention W[k*n+i, r],
    # C is 8x4, so C = [C1; C2] where C1 = A*B1 (4x4) and C2 = A*B2 (4x4).
    #
    # For each 4x4x4 multiplication, we use rank-49 Strassen.
    # The U factors (A-side) are the SAME for both multiplications.
    # So we can share U: total rank = 49 + 49 = 98, but U has only 49 unique columns
    # repeated. This doesn't reduce rank below 98 unless we find cross-block dependencies.
    #
    # However, we can try a different approach: build the full 4x4x8 tensor directly
    # using a block-Strassen structure where the 8 columns of B are handled together.
    #
    # Alternative: Use the fact that 4x4x8 can be seen as 2 independent 4x4x4 problems.
    # Best known rank for 4x4x4 is 49 (Strassen). So 2*49 = 98 is our baseline.
    #
    # Let's build the rank-49 4x4x4 decomposition and duplicate it.

    def build_strassen_2x2x2():
        """Rank-7 Strassen decomposition for 2x2 x 2x2 multiplication.
        A indices: 0=a, 1=b, 2=c, 3=d (row-major: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1])
        B indices: 0=e, 1=f, 2=g, 3=h
        C indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
        """
        U7 = np.zeros((4, 7), dtype=np.float64)
        V7 = np.zeros((4, 7), dtype=np.float64)
        W7 = np.zeros((4, 7), dtype=np.float64)

        # P1 = (a+b)*(e+f) -> C00 += P1, C11 += P1
        U7[0, 0] = 1; U7[1, 0] = 1
        V7[0, 0] = 1; V7[1, 0] = 1
        W7[0, 0] = 1; W7[3, 0] = 1

        # P2 = (c+d)*e -> C10 += P2
        U7[2, 1] = 1; U7[3, 1] = 1
        V7[0, 1] = 1
        W7[2, 1] = 1

        # P3 = a*(f-h) -> C01 += P3, C11 += P3
        U7[0, 2] = 1
        V7[1, 2] = 1; V7[3, 2] = -1
        W7[1, 2] = 1; W7[3, 2] = 1

        # P4 = d*(g-e) -> C00 += P4
        U7[3, 3] = 1
        V7[2, 3] = 1; V7[0, 3] = -1
        W7[0, 3] = 1

        # P5 = (a+d)*(e+h) -> C00 -= P5, C01 += P5, C10 += P5
        U7[0, 4] = 1; U7[3, 4] = 1
        V7[0, 4] = 1; V7[3, 4] = 1
        W7[0, 4] = -1; W7[1, 4] = 1; W7[2, 4] = 1

        # P6 = (b-d)*(g+h) -> C11 += P6
        U7[1, 5] = 1; U7[3, 5] = -1
        V7[2, 5] = 1; V7[3, 5] = 1
        W7[3, 5] = 1

        # P7 = (c-a)*(f+g) -> C00 += P7, C10 += P7
        U7[2, 6] = 1; U7[0, 6] = -1
        V7[1, 6] = 1; V7[2, 6] = 1
        W7[0, 6] = 1; W7[2, 6] = 1

        return U7, V7, W7

    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4 x 4x4 matrix multiplication using Strassen recursion.
        A, B, C are all 4x4, linearized in row-major order.
        A indices: i*4+j for i,j in 0..3
        B indices: j*4+k for j,k in 0..3
        C indices: k*4+i for k,i in 0..3
        """
        U7, V7, W7 = build_strassen_2x2x2()

        # Block structure:
        # A = [[A11, A12], [A21, A22]] where each block is 2x2
        # A11: rows 0-1, cols 0-1 -> linear indices 0,1,2,3
        # A12: rows 0-1, cols 2-3 -> linear indices 4,5,6,7
        # A21: rows 2-3, cols 0-1 -> linear indices 8,9,10,11
        # A22: rows 2-3, cols 2-3 -> linear indices 12,13,14,15
        #
        # B = [[B11, B12], [B21, B22]]
        # B11: rows 0-1, cols 0-1 -> linear indices 0,1,2,3
        # B12: rows 0-1, cols 2-3 -> linear indices 4,5,6,7
        # B21: rows 2-3, cols 0-1 -> linear indices 8,9,10,11
        # B22: rows 2-3, cols 2-3 -> linear indices 12,13,14,15
        #
        # C = [[C11, C12], [C21, C22]]
        # C11: rows 0-1, cols 0-1 -> linear indices 0,1,2,3
        # C12: rows 0-1, cols 2-3 -> linear indices 4,5,6,7
        # C21: rows 2-3, cols 0-1 -> linear indices 8,9,10,11
        # C22: rows 2-3, cols 2-3 -> linear indices 12,13,14,15

        # Strassen 4x4x4:
        # P1 = (A11+A22)*(B11+B22)
        # P2 = (A21+A22)*B11
        # P3 = A11*(B12-B22)
        # P4 = A22*(B21-B11)
        # P5 = (A11+A12)*B22
        # P6 = (A21-A11)*(B11+B12)
        # P7 = (A12-A22)*(B21+B22)
        #
        # C11 = P1 - P4 + P5 - P7
        # C12 = P3 + P5
        # C21 = P2 + P7
        # C22 = P1 - P2 + P3 + P6

        # A-block combinations for each Pi: (block_idx, sign)
        A_combos = [
            [(0, 1), (3, 1)],   # P1: A11 + A22
            [(2, 1), (3, 1)],   # P2: A21 + A22
            [(0, 1)],           # P3: A11
            [(3, 1)],           # P4: A22
            [(0, 1), (1, 1)],   # P5: A11 + A12
            [(2, 1), (0, -1)],  # P6: A21 - A11
            [(1, 1), (3, -1)],  # P7: A12 - A22
        ]

        B_combos = [
            [(0, 1), (3, 1)],   # P1: B11 + B22
            [(0, 1)],           # P2: B11
            [(1, 1), (3, -1)],  # P3: B12 - B22
            [(2, 1), (0, -1)],  # P4: B21 - B11
            [(3, 1)],           # P5: B22
            [(0, 1), (1, 1)],   # P6: B11 + B12
            [(2, 1), (3, 1)],   # P7: B21 + B22
        ]

        # C-block contributions: Cj = sum_i coeff[i] * Pi
        C_coeffs = [
            [1, 0, 0, -1, 1, 0, -1],   # C11: P1 - P4 + P5 - P7
            [0, 0, 1, 0, 1, 0, 0],     # C12: P3 + P5
            [0, 1, 0, 0, 0, 0, 1],     # C21: P2 + P7
            [1, -1, 1, 0, 0, 1, 0],    # C22: P1 - P2 + P3 + P6
        ]

        block_offsets = [0, 4, 8, 12]

        U4 = np.zeros((16, 49), dtype=np.float64)
        V4 = np.zeros((16, 49), dtype=np.float64)
        W4 = np.zeros((16, 49), dtype=np.float64)

        for pi in range(7):
            for t in range(7):
                col = pi * 7 + t

                # U: combine A-blocks
                for blk, sign in A_combos[pi]:
                    offset = block_offsets[blk]
                    for r in range(4):
                        U4[offset + r, col] += sign * U7[r, t]

                # V: combine B-blocks
                for blk, sign in B_combos[pi]:
                    offset = block_offsets[blk]
                    for r in range(4):
                        V4[offset + r, col] += sign * V7[r, t]

                # W: each C-block gets its coefficient
                for cj in range(4):
                    coeff = C_coeffs[cj][pi]
                    if coeff != 0:
                        offset = block_offsets[cj]
                        for r in range(4):
                            W4[offset + r, col] += coeff * W7[r, t]

        return U4, V4, W4

    def verify_decomposition(U, V, W, T, tol=1e-6):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = build_4x4x4_strassen()

    # Verify 4x4x4 decomposition
    T4 = np.zeros((16, 16, 16), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                T4[a, b, c] = 1

    if not verify_decomposition(U4, V4, W4, T4, tol=1e-6):
        # Fallback: schoolbook for 4x4x4
        U4 = np.zeros((16, 64), dtype=np.float64)
        V4 = np.zeros((16, 64), dtype=np.float64)
        W4 = np.zeros((16, 64), dtype=np.float64)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i * 4 + j, col] = 1
                    V4[j * 4 + k, col] = 1
                    W4[k * 4 + i, col] = 1
                    col += 1

    rank4 = U4.shape[1]  # should be 49

    # Extend to 4x4x8:
    # B is 4x8, split into B1 (cols 0-3) and B2 (cols 4-7)
    # C is 8x4, split into C1 (rows 0-3) and C2 (rows 4-7)
    # C1 = A * B1, C2 = A * B2
    #
    # U: (16, 2*rank4) - same A for both
    # V: (32, 2*rank4) - B indices: j*8+k
    # W: (32, 2*rank4) - C indices: k*4+i

    total_rank = 2 * rank4
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((32, total_rank), dtype=np.float64)
    W = np.zeros((32, total_rank), dtype=np.float64)

    # Left block (k=0..3): B1
    for col in range(rank4):
        U[:, col] = U4[:, col]
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, col] = V4[j * 4 + k, col]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, col] = W4[k * 4 + i, col]

    # Right block (k=4..7): B2
    for col in range(rank4):
        col2 = rank4 + col
        U[:, col2] = U4[:, col]
        for j in range(4):
            for k in range(4):
                V[j * 8 + (k + 4), col2] = V4[j * 4 + k, col]
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i, col2] = W4[k * 4 + i, col]

    # Verify full decomposition
    if not verify_decomposition(U, V, W, T, tol=1e-6):
        # Fallback to schoolbook
        U = U_sb
        V = V_sb
        W = W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank by finding linear dependencies
    # The tensor is T = sum_r U_r * V_r * W_r
    # If we can find a linear transformation that reduces the number of non-zero terms,
    # we can reduce the rank.
    #
    # Approach: Stack the factor matrices and find the rank of the combined tensor.
    # Flatten T into a matrix and find its rank, but that's not directly helpful.
    #
    # Better approach: Try to find a subset of columns that still spans the tensor.
    # This is NP-hard in general, but for small ranks we can try greedy elimination.
    #
    # Alternative: Use the fact that the two blocks share U. If we form a new
    # decomposition where U is shared, we might be able to reduce the effective rank.
    #
    # Let's try a different structural approach: build the 4x4x8 tensor directly
    # using a modified Strassen where the B-dimension is 8.
    #
    # For 4x4 x 4x8:
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # Split B into 2x4 blocks: B_top (rows 0-1, cols 0-7), B_bot (rows 2-3, cols 0-7)
    # But this doesn't align well with Strassen.
    #
    # Let's try: split B into four 2x2 blocks: B11, B12, B13, B14 (top row of blocks)
    # and B21, B22, B23, B24 (bottom row of blocks)
    # C is 8x4, split into four 2x2 blocks: C11, C12, C13, C14 (top) and C21, C22, C23, C24 (bottom)
    #
    # This is getting complex. Let's stick with the 98-rank decomposition and try
    # to reduce it via null-space analysis.
    #
    # Greedy rank reduction:
    # For each column r, check if removing it still allows exact reconstruction.
    # This is expensive, so we'll do a limited search.

    # Check time
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find linear dependencies by checking if any column is a linear combination
    # of others in a way that preserves the tensor.
    # This is equivalent to finding the rank of the set of rank-1 tensors.
    #
    # A simpler approach: flatten the tensor and compute its "tensor rank" approximately.
    # But exact tensor rank is hard.
    #
    # Let's try a different strategy: use the known result that the rank of 4x4x4 is 49,
    # and for 4x4x8, the rank should be at most 2*49 = 98.
    # The best known lower bound for 4x4x8 is not well-established, but it's likely
    # between 49 and 98.
    #
    # Let's try to see if we can do better by using a different block structure.
    # What if we treat the 4x4x8 as a single problem and use a 2x2x4 Strassen-like
    # decomposition?
    #
    # For 2x2 x 2x4:
    # A is 2x2, B is 2x4, C is 4x2
    # C = A * B
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[0,2] = a*i + b*k
    # C[0,3] = a*j + b*l
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    # C[1,2] = c*i + d*k
    # C[1,3] = c*j + d*l
    #
    # This is 8 terms schoolbook. Can we do better?
    # Strassen-like:
    # P1 = (a+b)*(e+g) -> contributes to C[0,0] and C[1,0]
    # P2 = (c+d)*e -> contributes to C[1,0]
    # P3 = a*(f-h) -> contributes to C[0,1] and C[1,1]... wait, C[1,1] = c*f + d*h
    # This doesn't work cleanly because B has 4 columns.
    #
    # Let's just go with the 98-rank decomposition. It's valid and better than schoolbook.

    # Check if entries are integers or half-integers
    # All entries should be integers since we only used +/-1 coefficients
    max_val = max(np.max(np.abs(U)), np.max(np.abs(V)), np.max(np.abs(W)))
    if max_val > 10:
        # Something went wrong
        pass

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
