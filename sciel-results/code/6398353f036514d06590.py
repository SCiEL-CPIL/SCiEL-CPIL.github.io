"""Program for 4x4x4 matrix multiplication tensor decomposition.

Uses recursive Strassen (rank 49) to achieve a tighter upper bound
than the schoolbook (rank 64) decomposition.
"""

import time
from fractions import Fraction

DIMENSIONS = [4, 4, 4]


# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 Strassen applied to 2x2 blocks, then Strassen
    again on each 2x2 sub-matrix product).

    Returns dict with 'U', 'V', 'W' matrices of shapes (16, 49), (16, 49), (16, 49).
    """
    deadline = time.time() + 130  # safety margin before 150s kill

    # Strassen 2x2 formulas:
    # For 2x2 matrices P and Q:
    # M1 = (P11+P22)(Q11+Q22)
    # M2 = (P21+P22)Q11
    # M3 = P11(Q12-Q22)
    # M4 = P22(Q21-Q11)
    # M5 = (P11+P12)Q22
    # M6 = (P21-P11)(Q11+Q12)
    # M7 = (P12-P22)(Q21+Q22)
    # R11 = M1+M4-M5+M7
    # R12 = M3+M5
    # R21 = M2+M4
    # R22 = M1-M2+M3+M6

    # Top-level Strassen on 4x4 matrices A, B with 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai* is 2x2
    # B = [[B11, B12], [B21, B22]] where each B** is 2x2

    # S1 = A11 + A22
    # S2 = A21 + A22
    # S3 = A11
    # S4 = A22
    # S5 = A11 + A12
    # S6 = A21 - A11
    # S7 = A12 - A22
    # T1 = B11 + B22
    # T2 = B11
    # T3 = B12 - B22
    # T4 = B21 - B11
    # T5 = B22
    # T6 = B11 + B12
    # T7 = B21 + B22
    # X1 = S1 * T1  (2x2 product)
    # X2 = S2 * T2  (2x2 product)
    # X3 = S3 * T3  (2x2 product)
    # X4 = S4 * T4  (2x2 product)
    # X5 = S5 * T5  (2x2 product)
    # X6 = S6 * T6  (2x2 product)
    # X7 = S7 * T7  (2x2 product)

    # C11 = X1 + X4 - X5 + X7
    # C12 = X3 + X5
    # C21 = X2 + X4
    # C22 = X1 - X2 + X3 + X6

    # Each Xk is a 2x2 matrix product, computed via Strassen:
    # For Xk = Sk * Tk:
    #   Yk1 = (Sk11+Sk22)(Tk11+Tk22)
    #   Yk2 = (Sk21+Sk22)Tk11
    #   Yk3 = Sk11(Tk12-Tk22)
    #   Yk4 = Sk22(Tk21-Tk11)
    #   Yk5 = (Sk11+Sk12)Tk22
    #   Yk6 = (Sk21-Sk11)(Tk11+Tk12)
    #   Yk7 = (Sk12-Sk22)(Tk21+Tk22)
    #   Xk11 = Yk1+Yk4-Yk5+Yk7
    #   Xk12 = Yk3+Yk5
    #   Xk21 = Yk2+Yk4
    #   Xk22 = Yk1-Yk2+Yk3+Yk6

    # Indexing: A[i,j] for i,j in 0..3
    # A11 = [[A00,A01],[A10,A11]], A12 = [[A02,A03],[A12,A13]]
    # A21 = [[A20,A21],[A30,A31]], A22 = [[A22,A23],[A32,A33]]

    # We'll build U, V, W as lists of Fractions.
    # U has shape (16, 49): row index = i*4+j for A[i,j]
    # V has shape (16, 49): row index = j*4+k for B[j,k]
    # W has shape (16, 49): row index = k*4+i for C[k,i] (trace convention W[k*n+i])

    n = m = p = 4
    rank = 49

    U = [[Fraction(0)] * rank for _ in range(16)]
    V = [[Fraction(0)] * rank for _ in range(16)]
    W = [[Fraction(0)] * rank for _ in range(16)]

    # For each of the 7 top-level Strassen multipliers Xk, we apply Strassen
    # to compute Sk * Tk, giving 7 scalar products per Xk.
    # Total: 7 * 7 = 49 scalar products.

    # Define the 7 top-level S combinations as linear maps on A's 16 elements.
    # S1 = A11 + A22: elements A00,A01,A10,A11 + A22,A23,A32,A33
    # S2 = A21 + A22: A20,A21,A30,A31 + A22,A23,A32,A33
    # S3 = A11: A00,A01,A10,A11
    # S4 = A22: A22,A23,A32,A33
    # S5 = A11 + A12: A00,A01,A10,A11 + A02,A03,A12,A13
    # S6 = A21 - A11: A20,A21,A30,A31 - A00,A01,A10,A11
    # S7 = A12 - A22: A02,A03,A12,A13 - A22,A23,A32,A33

    # Similarly for T combinations on B.

    # For each S_k (2x2 matrix), we need to extract its 4 elements as linear combos of A.
    # S_k = [[s00, s01], [s10, s11]]

    # S1 = [[A00+A22, A01+A23], [A10+A32, A11+A33]]
    # S2 = [[A20+A22, A21+A23], [A30+A32, A31+A33]]
    # S3 = [[A00, A01], [A10, A11]]
    # S4 = [[A22, A23], [A32, A33]]
    # S5 = [[A00+A02, A01+A03], [A10+A12, A11+A13]]
    # S6 = [[A20-A00, A21-A01], [A30-A10, A31-A11]]
    # S7 = [[A02-A22, A03-A23], [A12-A32, A13-A33]]

    # T1 = [[B00+B22, B01+B23], [B10+B32, B11+B33]]
    # T2 = [[B00, B01], [B10, B11]]
    # T3 = [[B02-B22, B03-B23], [B12-B32, B13-B33]]
    # T4 = [[B20-B00, B21-B01], [B30-B10, B31-B11]]
    # T5 = [[B22, B23], [B32, B33]]
    # T6 = [[B00+B02, B01+B03], [B10+B12, B11+B13]]
    # T7 = [[B20+B22, B21+B23], [B30+B32, B31+B33]]

    # Define S_k elements as dict mapping A index -> coefficient
    # A index: i*4+j
    # B index: j*4+k (but B is indexed [j,k] where j is row, k is col)
    # Wait, let me re-check: B is 4x4, B[j,k]. In V, row index = j*p+k = j*4+k.

    # Let me define the linear combinations explicitly.
    # For S_k, element (a,b) where a,b in {0,1}:
    # This is a linear combo of A[i,j] with i,j in 0..3.

    # S_k[a,b] = sum over (i,j) of coeff[i,j] * A[i,j]

    # S1:
    # S1[0,0] = A[0,0] + A[2,2]
    # S1[0,1] = A[0,1] + A[2,3]
    # S1[1,0] = A[1,0] + A[3,2]
    # S1[1,1] = A[1,1] + A[3,3]

    S_combos = [
        # S1: A11 + A22
        [[{(0,0): 1, (2,2): 1}, {(0,1): 1, (2,3): 1}],
         [{(1,0): 1, (3,2): 1}, {(1,1): 1, (3,3): 1}]],
        # S2: A21 + A22
        [[{(2,0): 1, (2,2): 1}, {(2,1): 1, (2,3): 1}],
         [{(3,0): 1, (3,2): 1}, {(3,1): 1, (3,3): 1}]],
        # S3: A11
        [[{(0,0): 1}, {(0,1): 1}],
         [{(1,0): 1}, {(1,1): 1}]],
        # S4: A22
        [[{(2,2): 1}, {(2,3): 1}],
         [{(3,2): 1}, {(3,3): 1}]],
        # S5: A11 + A12
        [[{(0,0): 1, (0,2): 1}, {(0,1): 1, (0,3): 1}],
         [{(1,0): 1, (1,2): 1}, {(1,1): 1, (1,3): 1}]],
        # S6: A21 - A11
        [[{(2,0): 1, (0,0): -1}, {(2,1): 1, (0,1): -1}],
         [{(3,0): 1, (1,0): -1}, {(3,1): 1, (1,1): -1}]],
        # S7: A12 - A22
        [[{(0,2): 1, (2,2): -1}, {(0,3): 1, (2,3): -1}],
         [{(1,2): 1, (3,2): -1}, {(1,3): 1, (3,3): -1}]],
    ]

    T_combos = [
        # T1: B11 + B22
        [[{(0,0): 1, (2,2): 1}, {(0,1): 1, (2,3): 1}],
         [{(1,0): 1, (3,2): 1}, {(1,1): 1, (3,3): 1}]],
        # T2: B11
        [[{(0,0): 1}, {(0,1): 1}],
         [{(1,0): 1}, {(1,1): 1}]],
        # T3: B12 - B22
        [[{(0,2): 1, (2,2): -1}, {(0,3): 1, (2,3): -1}],
         [{(1,2): 1, (3,2): -1}, {(1,3): 1, (3,3): -1}]],
        # T4: B21 - B11
        [[{(2,0): 1, (0,0): -1}, {(2,1): 1, (0,1): -1}],
         [{(3,0): 1, (1,0): -1}, {(3,1): 1, (1,1): -1}]],
        # T5: B22
        [[{(2,2): 1}, {(2,3): 1}],
         [{(3,2): 1}, {(3,3): 1}]],
        # T6: B11 + B12
        [[{(0,0): 1, (0,2): 1}, {(0,1): 1, (0,3): 1}],
         [{(1,0): 1, (1,2): 1}, {(1,1): 1, (1,3): 1}]],
        # T7: B21 + B22
        [[{(2,0): 1, (2,2): 1}, {(2,1): 1, (2,3): 1}],
         [{(3,0): 1, (3,2): 1}, {(3,1): 1, (3,3): 1}]],
    ]

    # For each top-level k (0..6) and each bottom-level l (0..6):
    # The bottom-level Strassen on Sk * Tk uses 7 multipliers Yk1..Yk7.
    # Each Ykl is a product of two linear combos of Sk elements and Tk elements.

    # Strassen formulas for 2x2 matrices P, Q:
    # M1 = (P00+P11)(Q00+Q11)
    # M2 = (P10+P11)Q00
    # M3 = P00(Q01-Q11)
    # M4 = P11(Q10-Q00)
    # M5 = (P00+P01)Q11
    # M6 = (P10-P00)(Q00+Q01)
    # M7 = (P01-P11)(Q10+Q11)

    # For each multiplier type l (0..6), we need:
    # - A linear combo of Sk elements (which are themselves linear combos of A)
    # - A linear combo of Tk elements (which are themselves linear combos of B)

    # Let's define the Strassen multiplier patterns.
    # For P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]:
    # M1: P_part = p00+p11, Q_part = q00+q11
    # M2: P_part = p10+p11, Q_part = q00
    # M3: P_part = p00, Q_part = q01-q11
    # M4: P_part = p11, Q_part = q10-q00
    # M5: P_part = p00+p01, Q_part = q11
    # M6: P_part = p10-p00, Q_part = q00+q01
    # M7: P_part = p01-p11, Q_part = q10+q11

    # Each P_part is a linear combo of {p00, p01, p10, p11}
    # Each Q_part is a linear combo of {q00, q01, q10, q11}

    # P_part coefficients for each multiplier:
    # Index: [p00, p01, p10, p11]
    P_parts = [
        [1, 0, 0, 1],  # M1: p00+p11
        [0, 0, 1, 1],  # M2: p10+p11
        [1, 0, 0, 0],  # M3: p00
        [0, 0, 0, 1],  # M4: p11
        [1, 1, 0, 0],  # M5: p00+p01
        [-1, 0, 1, 0], # M6: p10-p00
        [0, 1, 0, -1], # M7: p01-p11
    ]

    Q_parts = [
        [1, 0, 0, 1],  # M1: q00+q11
        [1, 0, 0, 0],  # M2: q00
        [0, 1, 0, -1], # M3: q01-q11
        [0, 0, 1, -1], # M4: q10-q00
        [0, 0, 0, 1],  # M5: q11
        [1, 1, 0, 0],  # M6: q00+q01
        [0, 0, 1, 1],  # M7: q10+q11
    ]

    # Now, for each (k, l) where k is top-level (0..6) and l is bottom-level (0..6):
    # Column index = k * 7 + l

    # The P matrix (Sk) has elements S_k[0,0], S_k[0,1], S_k[1,0], S_k[1,1]
    # Each S_k[a,b] is a linear combo of A elements.
    # So the full linear combo of A for P_part is:
    # sum over (a,b) of P_parts[l][idx(a,b)] * S_k[a,b]
    # where idx(0,0)=0, idx(0,1)=1, idx(1,0)=2, idx(1,1)=3

    # Similarly for Q parts and T_k.

    # Then W: the result C is assembled from Xk values.
    # Xk = [[xk00, xk01], [xk10, xk11]]
    # Xk00 = M1+M4-M5+M7
    # Xk01 = M3+M5
    # Xk10 = M2+M4
    # Xk11 = M1-M2+M3+M6

    # So for each (k,l), the contribution to Xk element (a,b) is:
    # W_xk[a,b][l] where:
    # Xk00: [1, 0, 0, 1, -1, 0, 1]  (M1, M2, M3, M4, M5, M6, M7)
    # Xk01: [0, 0, 1, 0, 1, 0, 0]
    # Xk10: [0, 1, 0, 1, 0, 0, 0]
    # Xk11: [1, -1, 1, 0, 0, 1, 0]

    Xk_W = [
        [1, 0, 0, 1, -1, 0, 1],   # Xk00
        [0, 0, 1, 0, 1, 0, 0],    # Xk01
        [0, 1, 0, 1, 0, 0, 0],    # Xk10
        [1, -1, 1, 0, 0, 1, 0],   # Xk11
    ]

    # C assembly:
    # C11 = X0 + X3 - X4 + X6
    # C12 = X2 + X4
    # C21 = X1 + X3
    # C22 = X0 - X1 + X2 + X5

    # C11 is a 2x2 block: C[0,0], C[0,1], C[1,0], C[1,1]
    # C12 is a 2x2 block: C[0,2], C[0,3], C[1,2], C[1,3]
    # C21 is a 2x2 block: C[2,0], C[2,1], C[3,0], C[3,1]
    # C22 is a 2x2 block: C[2,2], C[2,3], C[3,2], C[3,3]

    # C block assembly coefficients: C_block[ci][k]
    # C11: k=0 -> +1, k=3 -> +1, k=4 -> -1, k=6 -> +1
    # C12: k=2 -> +1, k=4 -> +1
    # C21: k=1 -> +1, k=3 -> +1
    # C22: k=0 -> +1, k=1 -> -1, k=2 -> +1, k=5 -> +1

    C_block_coeffs = [
        # C11 (block at rows 0-1, cols 0-1)
        {0: 1, 3: 1, 4: -1, 6: 1},
        # C12 (block at rows 0-1, cols 2-3)
        {2: 1, 4: 1},
        # C21 (block at rows 2-3, cols 0-1)
        {1: 1, 3: 1},
        # C22 (block at rows 2-3, cols 2-3)
        {0: 1, 1: -1, 2: 1, 5: 1},
    ]

    # Now build U, V, W.
    # U[row, col]: row = i*4+j for A[i,j], col = k*7+l
    # V[row, col]: row = j*4+k for B[j,k], col = k*7+l
    # W[row, col]: row = k*4+i for C[k,i] (trace convention: W[k*n+i, c]), col = k*7+l

    # Wait, the trace convention says W[k*n+i, r] where the output is C[i,j] = sum_r U[i*r_a, r] * V[j*r_b, r] * W[k*n+i, r]
    # Actually, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    # This means W is indexed by [output_col * n + output_row, rank] or similar.
    # Standard: C[i,j] = sum over (a,b) A[i,a] B[a,j]
    # Tensor: T[i,j,a] = C_ij / (A_i * B_j) where the tensor is T[i,a,j] such that C[i,j] = sum_a T[i,a,j] A[i',a] B[a,j']
    # 
    # Actually, the matrix multiplication tensor T[i,j,k] where C[i,j] = sum_k A[i,k] B[k,j]
    # The decomposition: T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]... 
    # 
    # Let me think more carefully. The task says:
    # U: shape (n*m, rank) = (16, rank) - corresponds to A indices
    # V: shape (m*p, rank) = (16, rank) - corresponds to B indices  
    # W: shape (p*n, rank) = (16, rank) - corresponds to C indices, with W[k*n+i, r]
    #
    # So U[i*m+j, r] = U[i,j,r] where (i,j) indexes A[i,j]
    # V[j*p+k, r] = V[j,k,r] where (j,k) indexes B[j,k]
    # W[k*n+i, r] = W[k,i,r] where (k,i) indexes... C[i,k]? 
    #
    # The multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,k,j] = 1 if this is a valid multiplication term
    # T[i,k,j] = U[i,j] * V[j,k] * W[k,i]
    # 
    # So W[k*n+i, r] means the first index is k (column of C) and second is i (row of C).
    # W[row_in_W, r] where row_in_W = k*4 + i, and this corresponds to C[i,k].

    # Let me verify with the seed:
    # In the seed, for each (i,j,k):
    #   U[i*m+j][column] = 1  -> U[i*4+j] corresponds to A[i,j]
    #   V[j*p+k][column] = 1  -> V[j*4+k] corresponds to B[j,k]
    #   W[k*n+i][column] = 1  -> W[k*4+i] corresponds to C[i,k] (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Yes, W[k*4+i] = W for C[i,k].

    # So in our construction:
    # For column col = k_top*7 + l_bottom:
    # U[i*4+j, col] = coefficient of A[i,j] in the P part
    # V[j*4+k, col] = coefficient of B[j,k] in the Q part
    # W[k*4+i, col] = coefficient for C[i,k]

    # Let's build this.
    idx_map = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}

    for k_top in range(7):
        if time.time() > deadline:
            break
        for l_bot in range(7):
            col = k_top * 7 + l_bot

            # Compute U: linear combo of A elements
            # P_part = sum over (a,b) of P_parts[l_bot][idx(a,b)] * S_k_top[a,b]
            # S_k_top[a,b] = sum over (i,j) of S_combos[k_top][a][b][(i,j)] * A[i,j]
            
            # U[i*4+j, col] = sum over (a,b) of P_parts[l_bot][idx_map[(a,b)]] * S_combos[k_top][a][b].get((i,j), 0)
            for i in range(4):
                for j in range(4):
                    val = Fraction(0)
                    for a in range(2):
                        for b in range(2):
                            coeff_p = P_parts[l_bot][idx_map[(a, b)]]
                            if coeff_p != 0:
                                coeff_s = S_combos[k_top][a][b].get((i, j), 0)
                                if coeff_s != 0:
                                    val += coeff_p * coeff_s
                    U[i * 4 + j][col] = val

            # Compute V: linear combo of B elements
            # Q_part = sum over (a,b) of Q_parts[l_bot][idx(a,b)] * T_k_top[a,b]
            # T_k_top[a,b] = sum over (j,k) of T_combos[k_top][a][b][(j,k)] * B[j,k]
            
            for j in range(4):
                for k in range(4):
                    val = Fraction(0)
                    for a in range(2):
                        for b in range(2):
                            coeff_q = Q_parts[l_bot][idx_map[(a, b)]]
                            if coeff_q != 0:
                                coeff_t = T_combos[k_top][a][b].get((j, k), 0)
                                if coeff_t != 0:
                                    val += coeff_q * coeff_t
                    V[j * 4 + k][col] = val

            # Compute W: for C[i,k], the coefficient is
            # sum over (xk_a, xk_b) of C_block_coeffs[C_block][k_top] * Xk_W[xk_a][l_bot]
            # where C_block is determined by (i,k):
            # C[i,k] belongs to:
            #   i in {0,1}, k in {0,1} -> C11 (block 0)
            #   i in {0,1}, k in {2,3} -> C12 (block 1)
            #   i in {2,3}, k in {0,1} -> C21 (block 2)
            #   i in {2,3}, k in {2,3} -> C22 (block 3)
            # And within the block, the position determines which Xk element:
            #   C11[0,0] = X0[0,0] -> i=0,k=0 -> block 0, pos (0,0) -> xk_a=0
            #   C11[0,1] = X0[0,1] -> i=0,k=1 -> block 0, pos (0,1) -> xk_a=1
            #   C11[1,0] = X0[1,0] -> i=1,k=0 -> block 0, pos (1,0) -> xk_a=2
            #   C11[1,1] = X0[1,1] -> i=1,k=1 -> block 0, pos (1,1) -> xk_a=3
            #   C12[0,0] = X2[0,0] -> i=0,k=2 -> block 1, pos (0,0) -> xk_a=0
            #   etc.

            for i in range(4):
                for k in range(4):
                    # Determine C block
                    if i < 2 and k < 2:
                        block = 0
                        pos_a = i  # 0 or 1
                        pos_b = k  # 0 or 1
                    elif i < 2 and k >= 2:
                        block = 1
                        pos_a = i
                        pos_b = k - 2
                    elif i >= 2 and k < 2:
                        block = 2
                        pos_a = i - 2
                        pos_b = k
                    else:
                        block = 3
                        pos_a = i - 2
                        pos_b = k - 2

                    xk_a = pos_a * 2 + pos_b  # 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)

                    block_coeff = C_block_coeffs[block].get(k_top, 0)
                    if block_coeff == 0:
                        W[k * 4 + i][col] = Fraction(0)
                    else:
                        W[k * 4 + i][col] = block_coeff * Xk_W[xk_a][l_bot]

    # Convert to lists (already lists of Fractions, but let's ensure plain ints where possible)
    def convert_matrix(M):
        result = []
        for row in M:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(val.numerator)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    return {"U": convert_matrix(U), "V": convert_matrix(V), "W": convert_matrix(W)}
# EVOLVE-BLOCK-END
