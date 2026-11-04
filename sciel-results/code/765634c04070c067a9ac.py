"""Rank-61 decomposition for 3x3x7 matrix multiplication (Strassen for the 2x2 block)."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7

    # We split B into B1 (3x3) and B2 (3x4).
    # Part 1: 3x3x3 using Strassen (rank 7).
    # Part 2: 3x3x4 using schoolbook (rank 36).
    # Total rank = 7 + 36 = 43? No, 3x3x3 schoolbook is 27, Strassen is 7.
    # But we need to be careful with indices.
    # Actually, the task is 3x3 * 3x7 -> 3x7.
    # Let's just use the known rank 23 for 3x3x3? No, let's stick to a safe, verifiable rank.
    # The seed program returned 63.
    # I will implement Strassen for the first 3x3 block of B, and schoolbook for the last 4 columns.
    # Wait, the indices for U, V, W are global.
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (21, rank)

    # Let's try to implement a rank 23 decomposition for 3x3x3?
    # It is complex. Let's use Strassen (rank 7) for the 2x2x2 sub-blocks?
    # Actually, a simple, correct, and better-than-63 rank is achievable by using
    # Strassen for the 2x2 part of the 3x3 matrix multiplication.
    # However, ensuring the exact tensor decomposition for arbitrary splits is tricky.
    #
    # Let's use the "block" approach:
    # Split A into [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # Split B into [B1; B2] where B1 is 2x7, B2 is 1x7.
    # C = A11*B1 + A12*B2 + A21*B1 + A22*B2.
    #
    # Term 1: A11 (2x2) * B1 (2x7) -> C1 (2x7).
    #   Rank = 7 * (2x2 rank) = 7 * 4 = 28 (schoolbook) or 7 * 3 (Strassen-like? No, 2x2x2 is 7, but here it's 2x2x7).
    #   Actually, multiplying 2x2 by 2x7 can be done with rank 14 (two Strassen 2x2x2 blocks).
    # Term 2: A12 (2x1) * B2 (1x7) -> C2 (2x7).
    #   Rank = 2 * 7 = 14 (schoolbook, since one dimension is 1, it's just outer products? No, it's 2x1 * 1x7 = 2x7. Rank = 2*7 = 14? No, rank of 2x1x7 tensor is 14? Yes, if no structure).
    #   Actually, A12 is 2x1, B2 is 1x7. C2[i, k] = A12[i, 0] * B2[0, k].
    #   This is an outer product of a 2-vector and a 7-vector. Rank = 14?
    #   Wait, the tensor T[i, j, k] for 2x1x7 has i in 0..1, j=0, k in 0..6.
    #   T[i, 0, k] = 1. This is a rank 14 tensor? No, it's a set of 14 disjoint 1s. Rank 14.
    # Term 3: A21 (1x2) * B1 (2x7) -> C3 (1x7).
    #   Rank = 2 * 7 = 14.
    # Term 4: A22 (1x1) * B2 (1x7) -> C4 (1x7).
    #   Rank = 1 * 7 = 7.
    #
    # Total Rank = 14 + 14 + 14 + 7 = 49.
    # This is better than 63.
    #
    # Let's refine Term 1: 2x2 * 2x7.
    # Split B1 into B1a (2x3) and B1b (2x4)? No, just do 2x2 * 2x7 directly.
    # We can view 2x2 * 2x7 as two independent 2x2 * 2x1 multiplications? No, 7 columns.
    # Actually, we can use Strassen for the 2x2 * 2x2 blocks.
    # Split the 7 columns of B1 into 3+4? Or 2+2+2+1?
    # If we split into 2+2+2+1, we have three 2x2x2 blocks (rank 7 each) and one 2x2x1 block (rank 4).
    # Rank for Term 1 = 7*3 + 4 = 25.
    #
    # Let's try a simpler split for the whole 3x3x7.
    #
    # Actually, let's just use the rank 23 algorithm for 3x3x3 if I can code it.
    # Or better, let's use a known rank 30 algorithm for 3x3x7?
    #
    # Let's stick to the block decomposition that is easy to verify:
    # Rank = 49 as derived above.
    #
    # Let's implement the rank 49 decomposition.
    #
    # Indices:
    # n=3, m=3, p=7
    # U: 9 x R
    # V: 21 x R
    # W: 21 x R
    #
    # A indices: i (0..2), j (0..2) -> U[i*3+j]
    # B indices: j (0..2), k (0..6) -> V[j*7+k]
    # C indices: k (0..6), i (0..2) -> W[k*3+i]
    #
    # Split A:
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2}
    # A21: i in {2}, j in {0,1}
    # A22: i in {2}, j in {2}
    #
    # Split B:
    # B1: j in {0,1}, k in {0..6} (2x7)
    # B2: j in {2}, k in {0..6} (1x7)
    #
    # C contributions:
    # C1 (from A11*B1): i in {0,1}, k in {0..6}
    # C2 (from A12*B2): i in {0,1}, k in {0..6}
    # C3 (from A21*B1): i in {2}, k in {0..6}
    # C4 (from A22*B2): i in {2}, k in {0..6}
    #
    # Term 1: A11 (2x2) * B1 (2x7).
    #   We will use schoolbook for 2x2 * 2x7. Rank = 2*2*7 = 28.
    #   Wait, I calculated 25 using Strassen blocks. Let's use schoolbook for simplicity and correctness first.
    #   Rank 1 = 28.
    # Term 2: A12 (2x1) * B2 (1x7). Rank = 2*7 = 14.
    # Term 3: A21 (1x2) * B1 (2x7). Rank = 2*7 = 14.
    # Term 4: A22 (1x1) * B2 (1x7). Rank = 1*7 = 7.
    #
    # Total Rank = 28 + 14 + 14 + 7 = 63. Same as schoolbook.
    #
    # To improve, I MUST optimize Term 1 (2x2 * 2x7).
    # Let's use Strassen for 2x2 * 2x2 blocks.
    # Split B1 (2x7) into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1).
    # B1a: k in {0,1}
    # B1b: k in {2,3}
    # B1c: k in {4,5}
    # B1d: k in {6}
    #
    # A11 * B1a: Rank 7 (Strassen)
    # A11 * B1b: Rank 7 (Strassen)
    # A11 * B1c: Rank 7 (Strassen)
    # A11 * B1d: Rank 4 (Schoolbook 2x2x1)
    #
    # Rank 1 = 7 + 7 + 7 + 4 = 25.
    # Total Rank = 25 + 14 + 14 + 7 = 60.
    #
    # Let's implement this Rank 60 decomposition.

    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # --- Term 1: A11 (2x2) * B1 (2x7) ---
    # Sub-term 1a: A11 * B1a (2x2x2, Strassen)
    # B1a: j in {0,1}, k in {0,1}
    # Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices for A11: i in {0,1}, j in {0,1}. Local i, j. Global i, j.
    # B indices for B1a: j in {0,1}, k in {0,1}. Local j, k. Global j, k.
    # C indices for C1a: i in {0,1}, k in {0,1}. Local i, k. Global i, k.
    
    def add_strassen_2x2x2(U, V, W, col, i_off, j_off, k_off):
        # i_off, j_off, k_off are the global offsets for the 2x2 block
        # M1: (A00+A11)(B00+B11)
        # A00: i=0, j=0. A11: i=1, j=1.
        # B00: j=0, k=0. B11: j=1, k=1.
        # C00: i=0, k=0. C01: i=0, k=1. C10: i=1, k=0. C11: i=1, k=1.
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M6
        # C11 = M1 - M3 + M4 + M6
        
        # M1
        U[i_off*3 + j_off][col] += 1
        U[(i_off+1)*3 + (j_off+1)][col] += 1
        V[(j_off)*7 + (k_off)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        W[(k_off)*3 + i_off][col] += 1
        W[(k_off+1)*3 + i_off][col] += 1
        W[(k_off)*3 + i_off+1][col] -= 1
        W[(k_off+1)*3 + i_off+1][col] += 1
        col += 1

        # M2
        U[(i_off+1)*3 + j_off][col] += 1
        U[(i_off+1)*3 + (j_off+1)][col] += 1
        V[(j_off)*7 + (k_off)][col] += 1
        W[(k_off)*3 + i_off+1][col] += 1
        W[(k_off+1)*3 + i_off+1][col] += 1
        col += 1

        # M3
        U[i_off*3 + j_off][col] += 1
        V[(j_off+1)*7 + (k_off)][col] += 1
        V[(j_off)*7 + (k_off)][col] -= 1
        W[(k_off+1)*3 + i_off][col] += 1
        W[(k_off+1)*3 + i_off+1][col] -= 1
        col += 1

        # M4
        U[(i_off+1)*3 + (j_off+1)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        V[(j_off)*7 + (k_off)][col] -= 1
        W[(k_off)*3 + i_off][col] -= 1
        W[(k_off+1)*3 + i_off+1][col] += 1
        col += 1

        # M5
        U[i_off*3 + j_off][col] += 1
        U[i_off*3 + (j_off+1)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        W[(k_off)*3 + i_off][col] += 1
        W[(k_off+1)*3 + i_off+1][col] += 1
        col += 1

        # M6
        U[(i_off+1)*3 + j_off][col] += 1
        U[i_off*3 + j_off][col] -= 1
        V[(j_off)*7 + (k_off)][col] += 1
        V[(j_off)*7 + (k_off+1)][col] += 1
        W[(k_off)*3 + i_off+1][col] += 1
        W[(k_off+1)*3 + i_off+1][col] += 1
        col += 1

        # M7
        U[i_off*3 + (j_off+1)][col] += 1
        U[(i_off+1)*3 + (j_off+1)][col] -= 1
        V[(j_off+1)*7 + (k_off)][col] += 1
        V[(j_off+1)*7 + (k_off+1)][col] += 1
        W[(k_off)*3 + i_off][col] -= 1
        W[(k_off+1)*3 + i_off][col] += 1
        col += 1

        return col

    # Apply Strassen to B1a (k=0,1), B1b (k=2,3), B1c (k=4,5)
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 0)
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 2)
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 4)

    # Sub-term 1d: A11 * B1d (2x2x1, Schoolbook)
    # B1d: j in {0,1}, k in {6}
    for i in range(2):
        for j in range(2):
            k = 6
            U[(i)*3 + j][col] = 1
            V[(j)*7 + k][col] = 1
            W[(k)*3 + i][col] = 1
            col += 1

    # --- Term 2: A12 (2x1) * B2 (1x7) ---
    # A12: i in {0,1}, j in {2}
    # B2: j in {2}, k in {0..6}
    # C2: i in {0,1}, k in {0..6}
    for i in range(2):
        for k in range(7):
            U[(i)*3 + 2][col] = 1
            V[(2)*7 + k][col] = 1
            W[(k)*3 + i][col] = 1
            col += 1

    # --- Term 3: A21 (1x2) * B1 (2x7) ---
    # A21: i in {2}, j in {0,1}
    # B1: j in {0,1}, k in {0..6}
    # C3: i in {2}, k in {0..6}
    for j in range(2):
        for k in range(7):
            U[(2)*3 + j][col] = 1
            V[(j)*7 + k][col] = 1
            W[(k)*3 + 2][col] = 1
            col += 1

    # --- Term 4: A22 (1x1) * B2 (1x7) ---
    # A22: i in {2}, j in {2}
    # B2: j in {2}, k in {0..6}
    # C4: i in {2}, k in {0..6}
    for k in range(7):
        U[(2)*3 + 2][col] = 1
        V[(2)*7 + k][col] = 1
        W[(k)*3 + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
