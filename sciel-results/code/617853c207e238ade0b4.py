"""Block-Strassen decomposition for 4x4 by 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _build_decomposition():
    """Build the rank-65 block-Strassen + schoolbook decomposition."""
    n, m, p = DIMENSIONS  # 4, 4, 5

    # Strassen 2x2 formulas (in terms of 7 subproducts)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # 2x2 Strassen linear forms for A-block (indices: (row, col) of 2x2)
    # For each Mj, the A-side is a 2x2 matrix of linear forms in A entries
    S2_A = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1},
    }
    S2_B = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(1,0): 1, (1,1): -1},
        4: {(0,1): 1, (1,1): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1},
    }
    # C-side: for each C block position (u,v) in {11,12,21,22}, the coefficient of Mj
    S2_C = {
        (0,0): {1: 1, 4: 1, 5: -1, 7: 1},
        (0,1): {3: 1, 5: 1},
        (1,0): {2: 1, 4: 1},
        (1,1): {1: 1, 2: -1, 3: 1, 6: 1},
    }

    # 4x4 Strassen at block level (blocks are 2x2)
    # X1 = A11 + A22, Y1 = B11 + B22
    # X2 = (A21+A22), Y2 = B11
    # X3 = A11, Y3 = (B21-B22)
    # X4 = A22, Y4 = (B12-B22)
    # X5 = (A11+A12), Y5 = B22
    # X6 = (A21-A11), Y6 = (B11+B12)
    # X7 = (A12-A22), Y7 = (B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each 4x4-Strassen index k (1..7), the A-block (2x2 blocks of 4x4)
    # and B-block are linear combinations of 2x2 blocks
    S4_A_block = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1},
    }
    S4_B_block = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(1,0): 1, (1,1): -1},
        4: {(0,1): 1, (1,1): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1},
    }
    # 4x4 C-block coefficients
    S4_C_block = {
        (0,0): {1: 1, 4: 1, 5: -1, 7: 1},
        (0,1): {3: 1, 5: 1},
        (1,0): {2: 1, 4: 1},
        (1,1): {1: 1, 2: -1, 3: 1, 6: 1},
    }

    # Build U, V, W as dictionaries: key -> value
    # U index: i*4 + j  (i in 0..3, j in 0..3)  -> row in A
    # V index: j*4 + k  (j in 0..3, k in 0..3)  -> row in B for 4x4 part
    # W index: k*4 + i  (k in 0..3, i in 0..3)  -> col in C for 4x4 part
    U = {}
    V = {}
    W = {}

    def add(U, V, W, urow, vrow, wrow, val):
        if val == 0:
            return
        U[(urow,)] = U.get((urow,), 0) + val
        V[(vrow,)] = V.get((vrow,), 0) + val
        W[(wrow,)] = W.get((wrow,), 0) + val

    # For each 4x4-Strassen term k (1..7) and each 2x2-Strassen term j (1..7)
    for k in range(1, 8):
        for j in range(1, 8):
            # A-side: X_k is a 2x2 matrix of 2x2 blocks, each block is a linear
            # form in A-blocks. Apply 2x2 Strassen j to get a linear form in X_k
            # entries, which is a linear form in A entries.
            #
            # X_k[u,v] = sum over (a_block, b_block) of S4_A_block[k][(u,v)] * A_block
            # where A_block is the 2x2 block at (u*2, v*2) in the 4x4 A matrix.
            #
            # Then the 2x2 Strassen j picks: linear form LA_j = sum_{(u',v')} S2_A[j][(u',v')] * X_k[u',v']
            # = sum_{(u',v')} S2_A[j][(u',v')] * sum_{(a,b) in S4_A_block[k][(u',v')]} coeff * A_block_{a,b}
            #
            # So the overall linear form in A is:
            # LA_j = sum_{(u',v')} S2_A[j][(u',v')] * (linear form of A-blocks for X_k at (u',v'))

            # Compute the A linear form: for each 4x4 entry (i, jA) (i in 0..3, jA in 0..3),
            # the coefficient.
            A_lin = {}
            for (u_prime, v_prime), coeff_sp in S2_A[j].items():
                # X_k[u_prime, v_prime] = sum over S4_A_block[k][(u_prime, v_prime)]
                for (a_block, b_block), coeff_sk in S4_A_block[k].get((u_prime, v_prime), {}).items():
                    # A_block at (a_block, b_block) corresponds to 4x4 entries
                    # rows a_block*2 .. a_block*2+1, cols b_block*2 .. b_block*2+1
                    for di in range(2):
                        for dj in range(2):
                            i_global = a_block * 2 + di
                            jA_global = b_block * 2 + dj
                            key = (i_global, jA_global)
                            A_lin[key] = A_lin.get(key, 0) + coeff_sp * coeff_sk

            # B linear form similarly
            B_lin = {}
            for (u_prime, v_prime), coeff_sp in S2_B[j].items():
                for (a_block, b_block), coeff_sk in S4_B_block[k].get((u_prime, v_prime), {}).items():
                    for di in range(2):
                        for dj in range(2):
                            jB_global = a_block * 2 + di  # row in B
                            k_global = b_block * 2 + dj    # col in B
                            key = (jB_global, k_global)
                            B_lin[key] = B_lin.get(key, 0) + coeff_sp * coeff_sk

            # C contribution: for each 4x4 entry (iC, kC) in the 4x4 output,
            # the coefficient from this (k, j) term.
            # C_block at (uC, vC) gets contribution S4_C_block[(uC, vC)][k] * S2_C[(u_prime, v_prime)][j]
            # where the 2x2 Strassen j is applied to the blocks, and the block
            # position in the 4x4 C is (uC, vC), and within that block the position
            # is (u_prime, v_prime).

            # For each 4x4 C entry (iC, kC):
            # iC = uC*2 + u_prime, kC = vC*2 + v_prime
            # coefficient = S4_C_block[(uC, vC)][k] * S2_C[(u_prime, v_prime)][j]

            for uC in range(2):
                for vC in range(2):
                    if k not in S4_C_block.get((uC, vC), {}):
                        continue
                    coeff_4 = S4_C_block[(uC, vC)][k]
                    for (u_prime, v_prime), coeff_2 in S2_C.get((u_prime, v_prime), {}).items():
                        if j != (u_prime, v_prime):
                            pass
                        # We need the coefficient of M_j at position (u_prime, v_prime)
                        # from the 2x2 Strassen C-side.
                    # Redo: iterate over positions in 2x2
                    for (u_prime, v_prime) in S2_C.keys():
                        if j not in S2_C[(u_prime, v_prime)]:
                            continue
                        coeff_2 = S2_C[(u_prime, v_prime)][j]
                        iC = uC * 2 + u_prime
                        kC = vC * 2 + v_prime
                        w_key = (kC, iC)  # W index: k*n + i
                        coeff_total = coeff_4 * coeff_2
                        if coeff_total == 0:
                            continue
                        # Now this term contributes coeff_total to W[w_key] for this (k,j) column
                        # We need to add this to the W entry, and correspondingly to U and V.
                        # But U and V are linear forms: the (k,j) column of U is A_lin,
                        # column of V is B_lin, column of W is the accumulated C coefficients.
                        pass

    # The above nested approach is getting complicated. Let me restructure:
    # For each (k, j) pair, I compute:
    #   A_lin: dict from (i, jA) to coefficient  (linear form in A)
    #   B_lin: dict from (jB, kB) to coefficient  (linear form in B)
    #   C_lin: dict from (kC, iC) to coefficient  (linear form in C, i.e., W)
    # Then I add a new column with these linear forms.

    columns = []  # list of (A_lin, B_lin, C_lin)

    for k in range(1, 8):
        for j in range(1, 8):
            # A linear form
            A_lin = {}
            for (u_prime, v_prime), coeff_sp in S2_A[j].items():
                for (a_block, b_block), coeff_sk in S4_A_block[k].get((u_prime, v_prime), {}).items():
                    for di in range(2):
                        for dj in range(2):
                            i_global = a_block * 2 + di
                            jA_global = b_block * 2 + dj
                            A_lin[(i_global, jA_global)] = A_lin.get((i_global, jA_global), 0) + coeff_sp * coeff_sk

            # B linear form
            B_lin = {}
            for (u_prime, v_prime), coeff_sp in S2_B[j].items():
                for (a_block, b_block), coeff_sk in S4_B_block[k].get((u_prime, v_prime), {}).items():
                    for di in range(2):
                        for dj in range(2):
                            jB_global = a_block * 2 + di
                            kB_global = b_block * 2 + dj
                            B_lin[(jB_global, kB_global)] = B_lin.get((jB_global, kB_global), 0) + coeff_sp * coeff_sk

            # C linear form (W-side)
            C_lin = {}
            for uC in range(2):
                for vC in range(2):
                    if k not in S4_C_block.get((uC, vC), {}):
                        continue
                    coeff_4 = S4_C_block[(uC, vC)][k]
                    for (u_prime, v_prime) in S2_C.keys():
                        if j not in S2_C[(u_prime, v_prime)]:
                            continue
                        coeff_2 = S2_C[(u_prime, v_prime)][j]
                        iC = uC * 2 + u_prime
                        kC = vC * 2 + v_prime
                        C_lin[(kC, iC)] = C_lin.get((kC, iC), 0) + coeff_4 * coeff_2

            if any(v != 0 for v in A_lin.values()) and any(v != 0 for v in B_lin.values()):
                columns.append((A_lin, B_lin, C_lin))

    # Add schoolbook columns for the 5th column of B (k=4 in 0-indexed, i.e., jB in 0..3, kB=4)
    # Term: A[i, jB] * B[jB, 4] -> C[i, 4]
    # U: A[i, jB] -> index i*4 + jB
    # V: B[jB, 4] -> index jB*4 + 4
    # W: C[i, 4] -> index 4*4 + i = 16 + i
    for i in range(4):
        for jB in range(4):
            A_lin = {(i, jB): 1}
            B_lin = {(jB, 4): 1}
            C_lin = {(4, i): 1}
            columns.append((A_lin, B_lin, C_lin))

    # Now build U, V, W matrices
    rank = len(columns)
    U = [[0] * rank for _ in range(16)]  # 4x4
    V = [[0] * rank for _ in range(20)]  # 4x5
    W = [[0] * rank for _ in range(20)]  # 5x4

    for c, (A_lin, B_lin, C_lin) in enumerate(columns):
        for (i, jA), val in A_lin.items():
            if val != 0:
                U[i * 4 + jA][c] = val
        for (jB, kB), val in B_lin.items():
            if val != 0:
                V[jB * 4 + kB][c] = val
        for (kC, iC), val in C_lin.items():
            if val != 0:
                W[kC * 4 + iC][c] = val

    return U, V, W, rank

def _verify(U, V, W, rank):
    """Verify that sum_c U[:,c] * V[:,c]^T == W[:,c] * ... for all (i, jB, kC)."""
    # Check: for each (i, jB, kC), sum_c U[i*4+jB, c] * V[jB*4+kB, c] == W[kC*4+iC, c]
    # Actually: the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] should equal
    # 1 if (i, k) corresponds to output, else 0. But W gives the structure.
    # Correct check: for all (i, j, k), sum_c U[i*4+j, c] * V[j*5+k, c] == W[k*4+i, c] summed? No.
    #
    # The decomposition means: (AB)[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*4+j,c] A[i,j]) * (sum_j V[j*5+k,c] B[j,k])
    # So the tensor T[i, j, k] = delta_{i,k}... no.
    #
    # Actually for matrix multiplication: T[i, j, k] = 1 (the multiplication tensor).
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[?]. No.
    #
    # Standard: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] is wrong dimensionally.
    #
    # Correct: For matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{i,j,k} = delta (it's 1 for the multiplication).
    # A rank-r decomposition: T_{i,j,k} = sum_{c=1}^r u_{i,j,c} * v_{j,k,c} * w_{k,i,c}?
    # No, that's not right either. The standard form is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)?
    # That would be a trilinear form, but matrix multiplication is bilinear in A and B.
    #
    # Actually the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k (it's the
    # multiplication tensor). A decomposition of rank r means:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  -- NO, this is not standard.
    #
    # Let me re-read the task. The task says:
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]"
    #
    # So U has shape (n*m, rank) = (16, rank), indexed by (i,j) flattened as i*m+j.
    # V has shape (m*p, rank) = (20, rank), indexed by (j,k) flattened as j*p+k.
    # W has shape (p*n, rank) = (20, rank), indexed by (k,i) flattened as k*n+i.
    #
    # The decomposition: for the matrix multiplication tensor T[i,j,k] = 1,
    # we need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But T[i,j,k] = 1 for all i,j,k (matrix multiplication is just summing products).
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T[i, j, k] = 1 (it's a constant tensor). No, that's not a tensor decomposition problem.
    #
    # I think the correct interpretation is:
    # The matrix multiplication map: (A, B) -> A*B.
    # This is a bilinear map, which can be seen as a trilinear form:
    # trilinear(A, B, C) = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    # The tensor for this is T[i, j, k] = 1 (for the pairing with C).
    #
    # Actually, I think the standard formulation is:
    # T_{i,j,k} = delta (it's the multiplication tensor where T[i,j,k] = 1 for all valid indices).
    # A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where the sum is over c, and this should equal 1 for all (i,j,k).
    #
    # Let me just verify numerically: for random A, B, check that
    # sum_c (sum_j U[i*4+j, c] * A[i,j]) * (sum_j V[j*5+k, c] * B[j,k]) == (A*B)[i,k]
    # This is the correct check.

    import random
    random.seed(42)
    for _ in range(50):
        A = [[random.randint(-5, 5) for _ in range(4)] for _ in range(4)]
        B = [[random.randint(-5, 5) for _ in range(5)] for _ in range(4)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(4)) for k in range(5)] for i in range(4)]
        for i in range(4):
            for k in range(5):
                total = 0
                for c in range(rank):
                    u_sum = sum(U[i * 4 + j][c] * A[i][j] for j in range(4))
                    v_sum = sum(V[j * 5 + k][c] * B[j][k] for j in range(4))
                    total += u_sum * v_sum
                if total != AB[i][k]:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin

    U, V, W, rank = _build_decomposition()

    # Verify
    if not _verify(U, V, W, rank):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
