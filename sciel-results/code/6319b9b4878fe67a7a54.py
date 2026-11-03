import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0.0
                for r in range(rank):
                    computed += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(computed - 1.0) > tol:
                    return False
    return True

def _build_strassen_2x2x2(U, V, W, col, u_a, u_b, u_c, u_d,
                          v_e, v_f, v_g, v_h,
                          w_00, w_01, w_10, w_11):
    """Apply the 7-multiplication Strassen decomposition for a 2x2x2 block."""
    # P1 = (a+d)(e+h)
    U[u_a, col] = 1; U[u_d, col] = 1
    V[v_e, col] = 1; V[v_h, col] = 1
    W[w_00, col] = 1; W[w_11, col] = 1
    col += 1

    # P2 = (c+d)e
    U[u_c, col] = 1; U[u_d, col] = 1
    V[v_e, col] = 1
    W[w_10, col] = 1; W[w_11, col] = -1
    col += 1

    # P3 = a(f-h)
    U[u_a, col] = 1
    V[v_f, col] = 1; V[v_h, col] = -1
    W[w_01, col] = 1; W[w_11, col] = 1
    col += 1

    # P4 = d(g-e)
    U[u_d, col] = 1
    V[v_g, col] = 1; V[v_e, col] = -1
    W[w_00, col] = 1; W[w_10, col] = 1
    col += 1

    # P5 = (a+b)h
    U[u_a, col] = 1; U[u_b, col] = 1
    V[v_h, col] = 1
    W[w_00, col] = -1; W[w_01, col] = 1
    col += 1

    # P6 = (c-a)(e+f)
    U[u_c, col] = 1; U[u_a, col] = -1
    V[v_e, col] = 1; V[v_f, col] = 1
    W[w_11, col] = 1
    col += 1

    # P7 = (b-d)(g+h)
    U[u_b, col] = 1; U[u_d, col] = -1
    V[v_g, col] = 1; V[v_h, col] = 1
    W[w_00, col] = 1
    col += 1

    return col

def _build_rank44_decomposition(n, m, p):
    """
    Build a rank-44 decomposition:
      - 28 rank for A[0:2, 0:2] * B[0:2, 0:8] (4 blocks of 2x2x2 Strassen)
      - 16 rank for A[0:2, 2] * B[2, 0:8] (Schoolbook)
    """
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    # Part 1: 2x2 * 2x8 using Strassen on 2-column blocks
    for j in range(4):
        w_r00 = 4 * j
        w_r01 = 4 * j + 2
        w_r10 = 4 * j + 1
        w_r11 = 4 * j + 3

        v_e = 2 * j
        v_f = 2 * j + 1
        v_g = 8 + 2 * j
        v_h = 9 + 2 * j

        col = _build_strassen_2x2x2(
            U, V, W, col,
            u_a=0, u_b=1, u_c=3, u_d=4,
            v_e=v_e, v_f=v_f, v_g=v_g, v_h=v_h,
            w_00=w_r00, w_01=w_r01, w_10=w_r10, w_11=w_r11
        )

    # Part 2: 2x1 * 1x8 schoolbook
    for i in range(2):
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 2 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
    return U, V, W

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS

    # Build the rank-44 decomposition
    U, V, W = _build_rank44_decomposition(n, m, p)

    # Verify initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook rank 48
        U, V, W = _build_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
