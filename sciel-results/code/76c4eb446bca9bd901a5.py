import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS

    # Use Strassen-based block decomposition for 4x4 * 4x6
    # Partition A (4x4) into 2x2 blocks of 2x2
    # Partition B (4x6) into 2x3 blocks of 2x2
    # C (4x6) will have 2x3 blocks of 2x2
    #
    # For each C block C_ik (i in {0,1}, k in {0,1,2}):
    #   C_ik = A_i0 * B_0k + A_i1 * B_1k
    # Each block product (2x2)*(2x2) uses 7 Strassen multiplications
    # Total: 2 * 3 * 2 * 7 = 84 rank terms

    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84

    col = 0
    for i in range(2):  # A row block index (0 or 1)
        for k in range(3):  # C col block index (0, 1, or 2)
            # C_ik = A_i0 * B_0k + A_i1 * B_1k
            for j in range(2):  # Sum over j: A_ij * B_jk
                # Block A_ij: rows [2*i, 2*i+1], cols [2*j, 2*j+1]
                # Block B_jk: rows [2*j, 2*j+1], cols [2*k, 2*k+1]

                # A_ij elements:
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1

                # B_jk elements:
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1

                # C_ik block: rows [2*i, 2*i+1], cols [2*k, 2*k+1]
                # W index: W[col*n + row, r]
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1

                # Strassen 7 multiplications
                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1

                # M2 = (c+d)e
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1

                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1

                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1

                # M5 = (a+b)h
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1

                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1

                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1

    # Verify the decomposition
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        return schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_mat = np.array(U).T  # (rank, n*m)
    V_mat = np.array(V).T  # (rank, m*p)
    W_mat = np.array(W).T  # (rank, p*n)

    recon = np.zeros((n, m, p), dtype=int)
    for c in range(rank):
        u_vec = U_mat[c].reshape(n, m)
        v_vec = V_mat[c].reshape(m, p)
        w_vec = W_mat[c].reshape(p, n)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]

    if not np.all(recon == 1):
        return False
    return True
