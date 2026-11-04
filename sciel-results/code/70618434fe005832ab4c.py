import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Build target tensor
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Strategy: Hybrid decomposition
    # Part 1: 4x4x4 multiplication (k=0..3) using Strassen Kronecker (Rank 49)
    # Part 2: 4x4x3 multiplication (k=4..6) using Schoolbook (Rank 48)
    # Total Rank: 97

    U_2, V_2, W_2 = _strassen_2x2x2()

    # --- Part 1: 4x4x4 Strassen Kronecker ---
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))  # Local 4x4 indexing: j*4+k
    W_444 = np.zeros((16, rank_444))  # Local 4x4 indexing: k*4+i

    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V_444[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W_444[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])

    # Embed Part 1 into full 4x4x7 space
    V_444_full = np.zeros((28, rank_444))
    for j in range(4):
        for k in range(4):
            V_444_full[j * 7 + k, :] = V_444[j * 4 + k, :]

    W_444_full = np.zeros((28, rank_444))
    for k in range(4):
        for i in range(4):
            W_444_full[k * 4 + i, :] = W_444[k * 4 + i, :]

    # --- Part 2: 4x4x3 Schoolbook ---
    # This handles k=4,5,6.
    # For each k in {4,5,6}, we perform standard matrix multiplication.
    # Rank per k is 4*4 = 16. Total 48.
    rank_443 = 48
    U_443 = np.zeros((16, rank_443))
    V_443_full = np.zeros((28, rank_443))
    W_443_full = np.zeros((28, rank_443))

    col_idx = 0
    for k_global in range(4, 7):
        for i in range(4):
            for j in range(4):
                a_idx = i * 4 + j
                b_idx = j * 7 + k_global
                c_idx = k_global * 4 + i
                
                U_443[a_idx, col_idx] = 1
                V_443_full[b_idx, col_idx] = 1
                W_443_full[c_idx, col_idx] = 1
                col_idx += 1

    # Combine
    total_rank = rank_444 + rank_443  # 97
    U_total = np.hstack([U_444, U_443])
    V_total = np.hstack([V_444_full, V_443_full])
    W_total = np.hstack([W_444_full, W_443_full])

    # Verify
    recon = np.einsum('ak,bk,ck->abc', U_total, V_total, W_total)
    if np.allclose(recon, T, atol=1e-10):
        return {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }

    # Fallback: schoolbook
    return _schoolbook(n, m, p)


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
