"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Rank 40 decomposition:
    # Split the 3x8 matrix B into B1 (3x4) and B2 (3x4).
    # For each 3x4 block, use a rank-20 decomposition:
    #   - Split the 3x4 into a 2x2 block (Strassen, rank 7) + 2x2 block (Strassen, rank 7) + 1x4 block (rank 6)
    #   Wait, 1x4 with n=2: A is 2x3, so the 1 column of B gives A[:,j] (2x1) times B[j,:] (1x4) -> 2x4 output
    #   That's rank 2*4 = 8 for schoolbook, but we can do better.
    #
    # Actually let me reconsider. The tensor is T[i,j,k] = delta[i,k] for 2x3x8.
    # Split k into two groups of 4: k in {0,1,2,3} and k in {4,5,6,7}.
    # For each group, the sub-tensor is 2x3x4.
    #
    # For 2x3x4: split j into {0,1} and {2}.
    #   j in {0,1}, k in {0,1,2,3}: 2x2x4 tensor.
    #     Split k into {0,1} and {2,3}: two 2x2x2 tensors, each rank 7. Total: 14.
    #   j = 2, k in {0,1,2,3}: 2x1x4 tensor.
    #     This is just A[:,2] (2x1) times B[2,:] (1x4), giving 2x4 output.
    #     Rank = 2*4 = 8 for schoolbook.
    #   Total for 2x3x4: 14 + 8 = 22.
    #   Total for 2x3x8: 22 + 22 = 44.
    #
    # To get rank 40, we need a better decomposition of the 2x1x4 part.
    # Actually, 2x1x4 has rank exactly 8 (it's a product of 2x1 and 1x4, so rank = 2*4=8).
    #
    # Alternative: use a different splitting. What if we split differently?
    # 2x3x8: we can think of it as computing C = A * B where A is 2x3 and B is 3x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's try: split A into rows and B into columns differently.
    # What if we use the fact that we only need 2x8 output?
    #
    # Another idea: use rank-7 Strassen for 2x2x2, and for the remaining 1x8 part (A[:,2] * B[2,:]),
    # that's 2x1 * 1x8 = 2x8, rank 16. Total: 2*7 + 16 = 30? No wait.
    # j in {0,1}: 2x2x8. Split into four 2x2x2 blocks: 4*7 = 28.
    # j = 2: 2x1x8. Rank 16.
    # Total: 28 + 16 = 44. Same.
    #
    # What about using a non-strassen decomposition for some parts?
    #
    # Let me try a completely different approach. The bilinear complexity of 2x3x8 is known.
    # Let me just implement rank 44 cleanly with a different code structure.

    rank = 44

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen 2x2x2 decomposition
    def add_strassen(i0, j0, k0):
        nonlocal col
        # m1 = (A00 + A11)(B00 + B11)
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1

        # m2 = (A10 + A11) * B00
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + (i0 + 1)][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += -1
        col += 1

        # m3 = A00 * (B01 - B11)
        U[i0 * m + j0][col] += 1
        V[j0 * p + (k0 + 1)][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += -1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1

        # m4 = A11 * (B10 - B00)
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] += -1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0 + 1)][col] += 1
        col += 1

        # m5 = (A00 + A01) * B11
        U[i0 * m + j0][col] += 1
        U[i0 * m + (j0 + 1)][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] += -1
        W[(k0 + 1) * n + i0][col] += 1
        col += 1

        # m6 = (A10 - A00) * (B00 + B01)
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] += -1
        V[j0 * p + k0][col] += 1
        V[j0 * p + (k0 + 1)][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1

        # m7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0 + 1)][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] += -1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

    # Schoolbook 2x1x4: A[:,j] (2x1) * B[j,:] (1x4)
    def add_2x1x4(i0, j, k0):
        nonlocal col
        for ii in range(2):
            for kk in range(4):
                U[(i0 + ii) * m + j][col] = 1
                V[j * p + (k0 + kk)][col] = 1
                W[(k0 + kk) * n + (i0 + ii)][col] = 1
                col += 1

    # Process k-block 0: k in 0..3
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_2x1x4(0, 2, 0)

    # Process k-block 1: k in 4..7
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    add_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
