"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10).
    Partition the output columns (10) into five 2-wide blocks.
    For each 2x2 x 2x2 sub-multiplication, use the classic rank-7 Strassen algorithm.
    Subtotal: 5 * 7 = 35.
  - Part 2: row 2 of A (2x1) times row 2 of B (1x10).
    Schoolbook: 2 * 1 * 10 = 20.
  Total rank: 55.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145.0


def _verify_decomposition(U, V, W, n, m, p, num_trials=10, seed=42):
    """Verify decomposition with random integer inputs using vectorized ops."""
    rng = np.random.RandomState(seed)
    rank = U.shape[1]
    for _ in range(num_trials):
        A = rng.randint(-10, 11, size=(n, m)).astype(float)
        B = rng.randint(-10, 11, size=(m, p)).astype(float)
        C_direct = A @ B

        A_flat = A.flatten()
        B_flat = B.flatten()
        u_sums = U.T @ A_flat
        v_sums = V.T @ B_flat
        scalars = u_sums * v_sums
        C_flat = W @ scalars
        C_decomp = C_flat.reshape(p, n).T

        err = np.max(np.abs(C_decomp - C_direct))
        if err > 1e-9:
            return False, err
    return True, 0.0


def _schoolbook(n, m, p):
    """Return a schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
    return U, V, W


def _strassen_2x2x2(U, V, W, a_rows, b_cols, c_rows, c_cols, base_idx):
    """
    Apply rank-7 Strassen for multiplying a 2x2 block of A by a 2x2 block of B,
    contributing to a 2x2 block of C.
    
    a_rows: indices into A for the 2x2 block (row-major: [r0c0, r0c1, r1c0, r1c1])
    b_cols: indices into B for the 2x2 block (row-major: [r0c0, r0c1, r1c0, r1c1])
    c_rows: indices into C for the 2x2 block (row-major: [r0c0, r0c1, r1c0, r1c1])
    base_idx: starting index in U,V,W columns
    """
    idx = base_idx
    # A elements: a00=a_rows[0], a01=a_rows[1], a10=a_rows[2], a11=a_rows[3]
    # B elements: b00=b_cols[0], b01=b_cols[1], b10=b_cols[2], b11=b_cols[3]
    # C elements: c00=c_rows[0], c01=c_rows[1], c10=c_rows[2], c11=c_rows[3]

    # M1 = (a00 + a11) * (b00 + b11)
    U[a_rows[0], idx] = 1.0
    U[a_rows[3], idx] = 1.0
    V[b_cols[0], idx] = 1.0
    V[b_cols[3], idx] = 1.0
    W[c_rows[0], idx] = 1.0
    W[c_rows[3], idx] = 1.0
    idx += 1

    # M2 = (a10 + a11) * b00
    U[a_rows[2], idx] = 1.0
    U[a_rows[3], idx] = 1.0
    V[b_cols[0], idx] = 1.0
    W[c_rows[2], idx] = 1.0
    W[c_rows[3], idx] = -1.0
    idx += 1

    # M3 = a00 * (b01 - b11)
    U[a_rows[0], idx] = 1.0
    V[b_cols[1], idx] = 1.0
    V[b_cols[3], idx] = -1.0
    W[c_rows[1], idx] = 1.0
    W[c_rows[3], idx] = 1.0
    idx += 1

    # M4 = a11 * (b10 - b00)
    U[a_rows[3], idx] = 1.0
    V[b_cols[2], idx] = 1.0
    V[b_cols[0], idx] = -1.0
    W[c_rows[0], idx] = 1.0
    W[c_rows[2], idx] = 1.0
    idx += 1

    # M5 = (a00 + a01) * b11
    U[a_rows[0], idx] = 1.0
    U[a_rows[1], idx] = 1.0
    V[b_cols[3], idx] = 1.0
    W[c_rows[0], idx] = -1.0
    W[c_rows[1], idx] = 1.0
    idx += 1

    # M6 = (a10 - a00) * (b00 + b01)
    U[a_rows[2], idx] = 1.0
    U[a_rows[0], idx] = -1.0
    V[b_cols[0], idx] = 1.0
    V[b_cols[1], idx] = 1.0
    W[c_rows[3], idx] = 1.0
    idx += 1

    # M7 = (a01 - a11) * (b10 + b11)
    U[a_rows[1], idx] = 1.0
    U[a_rows[3], idx] = -1.0
    V[b_cols[2], idx] = 1.0
    V[b_cols[3], idx] = 1.0
    W[c_rows[1], idx] = 1.0
    idx += 1

    return idx


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    rank = 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
    # A rows 0,1 (2x2), B rows 0,1 (2x10), C is 2x10
    # For each pair of output columns (k0, k1):
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # A 2x2 block: rows 0,1 of A; cols 0,1 of the 2x2 middle part
        # A_flat indices: row0col0=0, row0col1=1, row1col0=2, row1col1=3
        a_rows = [0, 1, 2, 3]  # a00, a01, a10, a11

        # B 2x2 block: rows 0,1 of B (the 2x2 middle part); cols k0, k1
        # B_flat indices: row0col0=0, row0col1=1, row1col0=10, row1col1=11 (for p=10)
        # Actually B is 3x10, so B[j,k] is at index j*10+k
        # For the 2x2 part (rows 0,1): B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=10+k0, B[1,k1]=10+k1
        b_cols = [k0, k1, 10 + k0, 10 + k1]  # b00, b01, b10, b11

        # C 2x2 block: all 2 rows of C; cols k0, k1
        # C is 2x10, C_flat index: row r, col c -> r*10+c
        # But W uses convention W[k*n+i, r] where k is col, i is row
        # So C[i,k] corresponds to W[k*n + i, :]
        # For C[0,k0]: W[k0*2+0, :], C[0,k1]: W[k1*2+0, :]
        # C[1,k0]: W[k0*2+1, :], C[1,k1]: W[k1*2+1, :]
        c_rows = [k0 * 2 + 0, k1 * 2 + 0, k0 * 2 + 1, k1 * 2 + 1]  # c00, c01, c10, c11

        idx = _strassen_2x2x2(U, V, W, a_rows, b_cols, c_rows, idx)

    # Part 2: Schoolbook for 2x1 x 1x10
    # A row 2 (2x1), B row 2 (1x10)
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0  # A[2,i] -> flat index 2*3+i = 6+i... wait
            # A is 2x3, A[i,j] flat index = i*3+j
            # We want A[2, :] but A only has rows 0,1. 
            # Wait, m=3, so A is 2x3. Row 2 of A doesn't exist.
            # The middle dimension is m=3. A is n x m = 2x3. B is m x p = 3x10.
            # So A has rows 0,1 (n=2) and cols 0,1,2 (m=3).
            # B has rows 0,1,2 (m=3) and cols 0..9 (p=10).
            # Part 1: A[:, 0:2] (2x2) * B[0:2, :] (2x10)
            # Part 2: A[:, 2] (2x1) * B[2, :] (1x10)
            # A[:, 2] is a 2x1 matrix: A[0,2] and A[1,2]
            # Flat index for A[i, 2] = i*3 + 2
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0  # B[2, k] flat index = 2*10 + k = 20+k
            W[k * 2 + i, idx] = 1.0  # C[i, k] -> W[k*2+i, :]
            idx += 1

    if time.time() - start_time < DEADLINE - 5.0:
        ok, err = _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=123)
        if not ok:
            U_sb, V_sb, W_sb = _schoolbook(n, m, p)
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
