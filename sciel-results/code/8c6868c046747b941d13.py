"""
Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Design: Strassen-based decomposition with rank 55.
  - Partition middle dimension 3 into 2+1.
  - Part 1: 2x2 * 2x10 using 5 independent 2x2*2x2 Strassen blocks (rank 35).
  - Part 2: 2x1 * 1x10 schoolbook (rank 20).
  - Total: rank 55.

The W matrix follows the official trace convention W[k*n+i, r] where k is the column index
and i is the row index of the output matrix C (n x p).
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145.0  # Safety margin before 150s kill


def _verify_decomposition(U, V, W, n, m, p, num_trials=10, seed=42):
    """Verify decomposition with random integer inputs using vectorized ops."""
    rng = np.random.RandomState(seed)
    rank = U.shape[1]
    for _ in range(num_trials):
        A = rng.randint(-10, 11, size=(n, m)).astype(float)
        B = rng.randint(-10, 11, size=(m, p)).astype(float)
        C_direct = A @ B

        # Vectorized decomposition evaluation
        A_flat = A.flatten()  # shape (n*m,)
        B_flat = B.flatten()  # shape (m*p,)
        u_sums = U.T @ A_flat  # (rank,)
        v_sums = V.T @ B_flat  # (rank,)
        scalars = u_sums * v_sums  # (rank,)
        C_flat = W @ scalars  # (p*n,)
        C_decomp = C_flat.reshape(p, n).T  # (n, p)

        err = np.max(np.abs(C_decomp - C_direct))
        if err > 1e-9:
            return False, err
    return True, 0.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Partition middle dim: 3 = 2 + 1
    # Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10) -> 5 Strassen blocks of 2x2x2
    # Part 2: row 2 of A (2x1) times row 2 of B (1x10) -> schoolbook rank 20

    rank = 35 + 20  # 55

    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
    # For each block, we multiply A_block (2x2) by B_block (2x2) to get C_block (2x2)
    # A_block elements: A[0,k0], A[0,k1], A[1,k0], A[1,k1] where k0=2*block, k1=2*block+1
    # B_block elements: B[k0,0], B[k0,1], B[k1,0], B[k1,1]
    # C_block elements: C[0,k0], C[0,k1], C[1,k0], C[1,k1]

    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # Strassen's 7 products for 2x2 * 2x2:
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        #
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C[0,0] = M1 - M4 + M5 - M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M6 + M7

        # In our notation:
        # a = A[0,k0], b = A[0,k1], c = A[1,k0], d = A[1,k1]
        # e = B[k0,0], f = B[k0,1], g = B[k1,0], h = B[k1,1]

        # M1 = (A[0,k0] + A[1,k1]) * (B[k0,0] + B[k1,1])
        U[0 * 3 + k0, idx] = 1.0   # A[0,k0]
        U[1 * 3 + k1, idx] = 1.0   # A[1,k1]
        V[k0 * 10 + 0, idx] = 1.0  # B[k0,0]
        V[k1 * 10 + 1, idx] = 1.0  # B[k1,1]
        W[k0 * 2 + 0, idx] = 1.0   # C[0,k0]
        W[k1 * 2 + 1, idx] = 1.0   # C[1,k1]
        idx += 1

        # M2 = (A[1,k0] + A[1,k1]) * B[k0,0]
        U[1 * 3 + k0, idx] = 1.0   # A[1,k0]
        U[1 * 3 + k1, idx] = 1.0   # A[1,k1]
        V[k0 * 10 + 0, idx] = 1.0  # B[k0,0]
        W[k0 * 2 + 1, idx] = 1.0   # C[0,k1]
        W[k1 * 2 + 1, idx] = -1.0  # C[1,k1]
        idx += 1

        # M3 = A[0,k0] * (B[k0,1] - B[k1,1])
        U[0 * 3 + k0, idx] = 1.0   # A[0,k0]
        V[k0 * 10 + 1, idx] = 1.0  # B[k0,1]
        V[k1 * 10 + 1, idx] = -1.0 # B[k1,1]
        W[k1 * 2 + 0, idx] = 1.0   # C[0,k1]
        W[k1 * 2 + 1, idx] = 1.0   # C[1,k1]
        idx += 1

        # M4 = A[1,k1] * (B[k1,0] - B[k0,0])
        U[1 * 3 + k1, idx] = 1.0   # A[1,k1]
        V[k1 * 10 + 0, idx] = 1.0  # B[k1,0]
        V[k0 * 10 + 0, idx] = -1.0 # B[k0,0]
        W[k0 * 2 + 0, idx] = 1.0   # C[0,k0]
        W[k0 * 2 + 1, idx] = 1.0   # C[0,k1]
        idx += 1

        # M5 = (A[0,k0] + A[0,k1]) * B[k1,1]
        U[0 * 3 + k0, idx] = 1.0   # A[0,k0]
        U[0 * 3 + k1, idx] = 1.0   # A[0,k1]
        V[k1 * 10 + 1, idx] = 1.0  # B[k1,1]
        W[k0 * 2 + 0, idx] = -1.0  # C[0,k0]
        W[k1 * 2 + 0, idx] = 1.0   # C[0,k1]
        idx += 1

        # M6 = (A[1,k0] - A[0,k0]) * (B[k0,0] + B[k0,1])
        U[1 * 3 + k0, idx] = 1.0   # A[1,k0]
        U[0 * 3 + k0, idx] = -1.0  # A[0,k0]
        V[k0 * 10 + 0, idx] = 1.0  # B[k0,0]
        V[k0 * 10 + 1, idx] = 1.0  # B[k0,1]
        W[k1 * 2 + 1, idx] = 1.0   # C[1,k1]
        idx += 1

        # M7 = (A[0,k1] - A[1,k1]) * (B[k1,0] + B[k1,1])
        U[0 * 3 + k1, idx] = 1.0   # A[0,k1]
        U[1 * 3 + k1, idx] = -1.0  # A[1,k1]
        V[k1 * 10 + 0, idx] = 1.0  # B[k1,0]
        V[k1 * 10 + 1, idx] = 1.0  # B[k1,1]
        W[k0 * 2 + 0, idx] = 1.0   # C[0,k0]
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10
    # A[0,2] * B[2,k] contributes to C[0,k]
    # A[1,2] * B[2,k] contributes to C[1,k]
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0   # A[i,2]
            V[2 * 10 + k, idx] = 1.0  # B[2,k]
            W[k * 2 + i, idx] = 1.0   # C[i,k]
            idx += 1

    # Verify correctness
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    ok, err = _verify_decomposition(U_arr, V_arr, W_arr, n, m, p, num_trials=5, seed=42)
    if not ok:
        raise RuntimeError(f"Decomposition verification failed: err={err}")

    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    # Verify correctness with random integer inputs (exact arithmetic)
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]

        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
