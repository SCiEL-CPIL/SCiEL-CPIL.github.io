import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: 9 blocks of 2x2, each multiplied via Strassen's 7-product algorithm.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # Decompose each 6x6 matrix into nine 2x2 blocks.
    # A = sum_{i,k} A[i,k] E_{i,k} where A[i,k] is a 2x2 block (rows 2i..2i+1, cols 2k..2k+1)
    # B = sum_{k,j} B[k,j] E_{k,j} where B[k,j] is a 2x2 block (rows 2k..2k+1, cols 2j..2j+1)
    # C = A*B = sum_{i,j} (sum_k A[i,k]*B[k,j]) E_{i,j}
    #
    # For each (i,j,k), we need to compute A[i,k]*B[k,j], which is a 2x2 product.
    # Using Strassen's algorithm for 2x2 multiplication, we need 7 scalar multiplications per block product.
    # Total rank = 3*3*3*7 = 189.

    # Strassen's 7 products for 2x2 matrices:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    rank = 189
    U = np.zeros((n * m, rank), dtype=np.float64)  # 36 x 189
    V = np.zeros((m * p, rank), dtype=np.float64)  # 36 x 189
    W = np.zeros((p * n, rank), dtype=np.float64)  # 36 x 189

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                # Strassen products for A[i_block,k_block] * B[k_block,j_block]
                # A block entries: A11 = A[i_block,k_block][0,0], A12 = A[i_block,k_block][0,1]
                #                  A21 = A[i_block,k_block][1,0], A22 = A[i_block,k_block][1,1]
                # B block entries: B11 = B[k_block,j_block][0,0], B12 = B[k_block,j_block][0,1]
                #                  B21 = B[k_block,j_block][1,0], B22 = B[k_block,j_block][1,1]

                # Define helper to get U row index for A[i_block,k_block][a,b]
                def u_row(a, b):
                    return (i_block * 2 + a) * m + (k_block * 2 + b)

                # Define helper to get V row index for B[k_block,j_block][c,d]
                def v_row(c, d):
                    return (k_block * 2 + c) * p + (j_block * 2 + d)

                # Define helper to get W row index for C[i_block,j_block][e,f]
                # W[k*n+i] corresponds to C[i,k], so k=j_block*2+f, i=i_block*2+e
                def w_row(e, f):
                    return (j_block * 2 + f) * n + (i_block * 2 + e)

                # M1 = (A11 + A22) * (B11 + B22)
                U[u_row(0, 0)][col] = 1  # A11
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(0, 0)][col] = 1  # B11
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = 1  # C11
                W[w_row(1, 1)][col] = 1  # C22
                col += 1

                # M2 = (A21 + A22) * B11
                U[u_row(1, 0)][col] = 1  # A21
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(0, 0)][col] = 1  # B11
                W[w_row(1, 0)][col] = 1  # C21
                W[w_row(1, 1)][col] = -1  # C22
                col += 1

                # M3 = A11 * (B12 - B22)
                U[u_row(0, 0)][col] = 1  # A11
                V[v_row(0, 1)][col] = 1  # B12
                V[v_row(1, 1)][col] = -1  # B22
                W[w_row(0, 1)][col] = 1  # C12
                W[w_row(1, 1)][col] = 1  # C22
                col += 1

                # M4 = A22 * (B21 - B11)
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(1, 0)][col] = 1  # B21
                V[v_row(0, 0)][col] = -1  # B11
                W[w_row(0, 0)][col] = 1  # C11
                W[w_row(1, 0)][col] = 1  # C21
                col += 1

                # M5 = (A11 + A12) * B22
                U[u_row(0, 0)][col] = 1  # A11
                U[u_row(0, 1)][col] = 1  # A12
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = -1  # C11
                W[w_row(0, 1)][col] = 1  # C12
                col += 1

                # M6 = (A21 - A11) * (B11 + B12)
                U[u_row(1, 0)][col] = 1  # A21
                U[u_row(0, 0)][col] = -1  # A11
                V[v_row(0, 0)][col] = 1  # B11
                V[v_row(0, 1)][col] = 1  # B12
                W[w_row(1, 1)][col] = 1  # C22
                col += 1

                # M7 = (A12 - A22) * (B21 + B22)
                U[u_row(0, 1)][col] = 1  # A12
                U[u_row(1, 1)][col] = -1  # A22
                V[v_row(1, 0)][col] = 1  # B21
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = 1  # C11
                col += 1

    # Verify the decomposition with a quick check
    # Check a few random indices to ensure correctness
    import random
    random.seed(42)
    valid = True
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = 0.0
        for r in range(rank):
            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook if verification fails
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback), dtype=np.float64)
        V_f = np.zeros((m * p, rank_fallback), dtype=np.float64)
        W_f = np.zeros((p * n, rank_fallback), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
