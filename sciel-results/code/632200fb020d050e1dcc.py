import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    t0 = time.time()
    deadline = t0 + 110.0

    # Design: 4x4x6 = 4x4x4 (recursive Strassen, rank 49) + 4x4x2 (Strassen on A, schoolbook on B, rank 28) = 77
    #
    # Part 1: 4x4x4 Strassen (recursive)
    #   Partition A(4x4) into 2x2 blocks of 2x2, B[:,:4](4x4) into 2x2 blocks of 2x2
    #   7 Strassen products, each (2x2)*(2x2) = 7 scalar mults (inner Strassen)
    #   Total: 7*7 = 49
    #
    # Part 2: 4x4x2
    #   A(4x4) partitioned into 2x2 blocks of 2x2
    #   B[:,4:6](4x2) partitioned into 2x2 blocks of 2x1
    #   7 Strassen products on A side, each (2x2)*(2x1) = 4 scalar mults (schoolbook)
    #   Total: 7*4 = 28
    #
    # Grand total: 49 + 28 = 77

    R = 77
    U = np.zeros((n * m, R))  # (16, 77)
    V = np.zeros((m * p, R))  # (24, 77)
    W = np.zeros((p * n, R))  # (24, 77)

    # Strassen coefficients for 2x2x2
    alpha = [
        [1, 0, 0, 1],   # s=0: A00 + A11
        [0, 0, 1, 1],   # s=1: A10 + A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00 + A01
        [-1, 0, 1, 0],  # s=5: A10 - A00
        [0, 1, 0, -1]   # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],   # s=0: B00 + B11
        [1, 0, 0, 0],   # s=1: B00
        [0, 1, 0, -1],  # s=2: B01 - B11
        [-1, 0, 1, 0],  # s=3: B10 - B00
        [0, 0, 0, 1],   # s=4: B11
        [1, 1, 0, 0],   # s=5: B00 + B01
        [0, 0, 1, 1]    # s=6: B10 + B11
    ]
    # Reconstruction for 2x2x2: C_ik = sum_s recon[s][ik] * M_s
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    col = 0

    # Part 1: 4x4x4 recursive Strassen
    for s in range(7):
        for si in range(7):
            # U: X_s[i,j] combined with inner Strassen
            for a in range(4):
                a_coeff = alpha[si][a]
                if a_coeff == 0:
                    continue
                x_row = a // 2
                x_col = a % 2
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    global_i = (r // 2) * 2 + x_row
                    global_j = (r % 2) * 2 + x_col
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += a_coeff * r_coeff

            # V: Y_s[j,k] for k in {0,1} (first 4 columns of B)
            for b in range(4):
                b_coeff = beta[si][b]
                if b_coeff == 0:
                    continue
                y_row = b // 2
                y_inner_col = b % 2
                for c in range(4):
                    c_coeff = beta[s][c]
                    if c_coeff == 0:
                        continue
                    global_j = (c // 2) * 2 + y_row
                    global_k = (c % 2) * 2 + y_inner_col
                    v_idx = global_j * p + global_k
                    V[v_idx, col] += b_coeff * c_coeff

            # W: reconstruction
            for (si_row, si_col, si_coeff) in recon[s]:
                for (inner_ii, inner_kk, inner_coeff) in recon[si]:
                    global_i = si_row * 2 + inner_ii
                    global_k = si_col * 2 + inner_kk
                    w_idx = global_k * n + global_i
                    W[w_idx, col] += si_coeff * inner_coeff

            col += 1

    # Part 2: 4x4x2 (Strassen on A, schoolbook on B)
    for s in range(7):
        for i in range(2):
            for j in range(2):
                # U: X_s[i,j]
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    global_i = (r // 2) * 2 + i
                    global_j = (r % 2) * 2 + j
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += r_coeff

                # V: Y_s[j, 0] for B[:,4:6] (2x1 blocks)
                # B is 4x2, partitioned into 2x2 blocks of 2x1
                # Block c (0-3): row block c//2, col block c%2 (but only 1 col)
                # Actually B[:,4:6] is 4x2. Partition into 2 row-blocks of 2x1 each.
                # So 2 blocks: B_0 (rows 0-1, cols 4-5) and B_1 (rows 2-3, cols 4-5)
                # Wait, that's 2x2 blocks. Let me reconsider.
                # B[:,4:6] is 4x2. For Strassen on A side, we need B blocks to match.
                # A is partitioned into 2x2 blocks of 2x2 (4 blocks).
                # B[:,4:6] should be partitioned into 2x2 blocks of 2x1 (4 blocks).
                # Block (r_b, c_b): rows [2*r_b, 2*r_b+2), cols [4+1*c_b, 4+1*c_b+1)
                # So 4 blocks: (0,0), (0,1), (1,0), (1,1)
                # Each is 2x1.
                
                for c in range(4):
                    c_coeff = beta[s][c]
                    if c_coeff == 0:
                        continue
                    global_j = (c // 2) * 2 + j
                    global_k = 4 + (c % 2)
                    v_idx = global_j * p + global_k
                    V[v_idx, col] += c_coeff

                # W: reconstruction for C_s[i, k]
                # C_s is 2x2 (since B block is 2x1, product is 2x1)
                # Actually X_s is 2x2, Y_s is 2x1, so X_s * Y_s is 2x1.
                # recon[s] maps C_s to C.
                for (si_row, si_col, si_coeff) in recon[s]:
                    global_i = si_row * 2 + i
                    global_k = 4 + si_col
                    w_idx = global_k * n + global_i
                    W[w_idx, col] += si_coeff

                col += 1

    # Verify
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, R)
    Vr = Va.reshape(m, p, R)
    Wr = Wa.reshape(p, n, R)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))

    if err > 1e-6:
        # Fallback to schoolbook
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

    # Convert to lists with integer values where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rounded = round(x)
                if abs(x - rounded) < 1e-9:
                    new_row.append(int(rounded))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
