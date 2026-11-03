"""Program for 2x4 * 4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0

def _construct_strassen():
    """Rank-28 Strassen-based decomposition for 2x4 * 4x8 multiplication."""
    n, m, p = 2, 4, 8
    rank = 28

    # Zero matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 32 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28

    # Strassen formulas for 2x2 * 2x2 = 2x2
    # P1 = (A11-A21)*(B21-B22) -> C12
    # P2 = (A11+A12)*B22        -> C22
    # P3 = A21*(B11-B22)        -> C21
    # P4 = A11*B11              -> C11
    # P5 = (A11+A21)*(B11-B21)  -> C11
    # P6 = (A21-A11)*(B12+B21)  -> C22
    # P7 = (A12-A21)*(B21-B22)  -> C12
    #
    # For block b (0 or 1), j_base = 2*b, k_base = 4*b
    # U indices: i*m + j = 2*i + j_base + dj
    # V indices: j*p + k = (j_base + dj)*8 + k_base + dk
    # W indices: k*n + i = (k_base + dk)*2 + i

    strassen_terms = [
        # (U_row_offsets, U_vals, V_row_offsets, V_vals, W_row_offsets, W_vals)
        # P1 = (A11-A21)*(B21-B22) -> C12
        ((0, 1), (1, -1), (1, 2), (1, -1), (0, 1), (1, 1)),
        # P2 = (A11+A12)*B22 -> C22
        ((0, 1), (1, 1), (2,), (1,), (1,), (1,)),
        # P3 = A21*(B11-B22) -> C21
        ((1,), (1,), (0, 2), (1, -1), (1,), (1,)),
        # P4 = A11*B11 -> C11
        ((0,), (1,), (0,), (1,), (0,), (1,)),
        # P5 = (A11+A21)*(B11-B21) -> C11
        ((0, 1), (1, 1), (0, 1), (1, -1), (0,), (1,)),
        # P6 = (A21-A11)*(B12+B21) -> C22
        ((1, 0), (1, -1), (1, 0), (1, 1), (1,), (1,)),
        # P7 = (A12-A21)*(B21-B22) -> C12
        ((1, 0), (-1, 1), (2, 1), (-1, 1), (0, 1), (1, 1)),
    ]

    for b in range(2):  # block 0 and 1
        j_base = 2 * b
        k_base = 4 * b
        for t in range(7):
            col = b * 7 + t
            u_rows, u_vals, v_rows, v_vals, w_rows, w_vals = strassen_terms[t]
            for ur, uv in zip(u_rows, u_vals):
                U[ur * 2 + j_base + ur % 2][col] += uv  # i=ur, j=j_base+ur
            for vr, vv in zip(v_rows, v_vals):
                V[(j_base + vr) * 8 + k_base + vr][col] += vv  # j=j_base+vr, k=k_base+vr
            for wr, wv in zip(w_rows, w_vals):
                W[(k_base + wr) * 2 + wr][col] += wv  # k=k_base+wr, i=wr

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition produces exact matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # must equal 1 for all (i,j,k) (identity tensor for matmul)

    # Build T from decomposition
    # T[i,j,k] = sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])

    # The identity tensor for matmul is all ones
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-10):
        max_err = np.max(np.abs(T - expected))
        print(f"Verification failed! Max error: {max_err}")
        return False
    return True


def construct():
    """Construct the rank-minimal decomposition for 2x4 * 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS

    # Try the Strassen-based rank-28 decomposition
    result = _construct_strassen()

    # Verify correctness
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result

    # Fallback: schoolbook decomposition (rank 64)
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
# EVOLVE-BLOCK-END
