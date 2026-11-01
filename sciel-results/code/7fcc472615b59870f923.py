import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x5 matrix.

    Strategy:
    - The matrix multiplication tensor T in F_{2,3,5} has structure:
      T_{i,j,k} = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
      with the convention that the output index is (k,i) and input indices are (i,j) and (j,k).

    - We use a block decomposition:
      * For j=0: schoolbook 2x1x5 -> rank 10
      * For j=1,2 with k=0,1: Strassen 2x2x2 -> rank 7
      * For j=1,2 with k=2: schoolbook 2x2x1 -> rank 4
      * For j=1,2 with k=3,4: Strassen 2x2x2 -> rank 7
      Total: 10 + 7 + 4 + 7 = 28

    - The W matrix uses the official trace convention: W[k*n + i, r]
    """
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s limit

    n, m, p = 2, 3, 5  # dimensions

    # Attempt rank-28 decomposition using block structure
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Strassen 2x2x2 helper: adds 7 terms
    # Local indices: i in {0,1}, j' in {0,1}, k' in {0,1}
    # Global mapping: i_global = i, j_global = j_offset + j', k_global = k_offset + k'
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M2: (A10+A11)B00 -> C10, -C11
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1

        # M3: A00(B01-B11) -> C01, C11
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M4: A11(B10-B00) -> C00, C10
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1

        # M5: (A00+A01)B11 -> -C00, C01
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01) -> C11
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11) -> C00
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1

    # Block 2a: j=1,2, k=0,1 (Strassen 2x2x2)
    add_strassen(i_off=0, j_off=1, k_off=0)

    # Block 2b: j=1,2, k=2 (schoolbook 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1

    # Block 2c: j=1,2, k=3,4 (Strassen 2x2x2)
    add_strassen(i_off=0, j_off=1, k_off=3)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    # Verify decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test

    C_decomp = np.zeros((n, p))
    for t in range(rank):
        u_t = U[:, t].reshape(n, m)
        v_t = V[:, t].reshape(m, p)
        w_t = W[:, t].reshape(p, n)
        s1 = np.sum(A_test * u_t)
        s2 = np.sum(B_test * v_t)
        C_decomp += s1 * s2 * w_t.T

    if not np.allclose(C_test, C_decomp, atol=1e-10):
        # Fallback: return schoolbook rank-30
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1.0
                    V_sb[j * p + k, col_sb] = 1.0
                    W_sb[k * n + i, col_sb] = 1.0
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
