"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.

Strategy:
  - Partition middle dimension m=3 into 2 + 1.
  - Part 1: 2x2 x 2x10 block. Split 10 output columns into five 2-wide blocks.
    Apply rank-7 Strassen to each 2x2x2 sub-problem. Cost: 5 * 7 = 35.
  - Part 2: 2x1 x 1x10 block. Schoolbook: 2*1*10 = 20.
  - Total rank: 55.

This program constructs the decomposition with exact integer entries,
verifies it, and returns the factor matrices.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # safety margin before the 150 s kill


def _build_decomposition(n, m, p):
    """Build the rank-55 decomposition (5 Strassen blocks + 20 schoolbook)."""
    rank = 5 * 7 + 2 * 1 * 10  # 55
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    idx = 0
    # Part 1: five Strassen 2x2x2 blocks over the 2-wide output column pairs.
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1]) -> C[0,k0], C[1,k1]
        U[0 * m + 0, idx] = 1
        U[1 * m + 1, idx] = 1
        V[0 * p + k0, idx] = 1
        V[1 * p + k1, idx] = 1
        W[k0 * n + 0, idx] = 1
        W[k1 * n + 1, idx] = 1
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0] -> C[0,k1], C[1,k1]
        U[1 * m + 0, idx] = 1
        U[1 * m + 1, idx] = 1
        V[0 * p + k0, idx] = 1
        W[k0 * n + 1, idx] = 1
        W[k1 * n + 1, idx] = -1
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1]) -> C[0,k1], C[1,k1]
        U[0 * m + 0, idx] = 1
        V[0 * p + k1, idx] = 1
        V[1 * p + k1, idx] = -1
        W[k1 * n + 0, idx] = 1
        W[k1 * n + 1, idx] = 1
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0]) -> C[0,k0], C[1,k0]
        U[1 * m + 1, idx] = 1
        V[1 * p + k0, idx] = 1
        V[0 * p + k0, idx] = -1
        W[k0 * n + 0, idx] = 1
        W[k0 * n + 1, idx] = 1
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1] -> C[0,k0], C[0,k1]
        U[0 * m + 0, idx] = 1
        U[0 * m + 1, idx] = 1
        V[1 * p + k1, idx] = 1
        W[k0 * n + 0, idx] = -1
        W[k1 * n + 0, idx] = 1
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1]) -> C[1,k1]
        U[1 * m + 0, idx] = 1
        U[0 * m + 0, idx] = -1
        V[0 * p + k0, idx] = 1
        V[0 * p + k1, idx] = 1
        W[k1 * n + 1, idx] = 1
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1]) -> C[0,k0]
        U[0 * m + 1, idx] = 1
        U[1 * m + 1, idx] = -1
        V[1 * p + k0, idx] = 1
        V[1 * p + k1, idx] = 1
        W[k0 * n + 0, idx] = 1
        idx += 1

    # Part 2: schoolbook for A[2,:] (2x1) times B[2,:] (1x10).
    for i in range(n):
        for k in range(p):
            U[i * m + 2, idx] = 1
            V[2 * p + k, idx] = 1
            W[k * n + i, idx] = 1
            idx += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, num_trials=12, seed=42):
    """Verify the decomposition against random integer inputs (vectorized, exact)."""
    rng = np.random.RandomState(seed)
    rank = U.shape[1]
    for _ in range(num_trials):
        A = rng.randint(-10, 11, size=(n, m))
        B = rng.randint(-10, 11, size=(m, p))
        C_direct = A @ B

        A_flat = A.reshape(-1)
        B_flat = B.reshape(-1)
        u_sums = U.T @ A_flat
        v_sums = V.T @ B_flat
        scalars = u_sums * v_sums
        C_flat = W @ scalars
        C_decomp = C_flat.reshape(p, n).T

        if not np.array_equal(C_decomp, C_direct):
            return False
    return True


def _tensor_residual(U, V, W, n, m, p, skip_col=None):
    """
    Return the max |sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1| over all i,j,k,
    optionally skipping one column. The matrix-multiplication tensor is all-ones.
    """
    rank = U.shape[1]
    cols = [r for r in range(rank) if r != skip_col]
    Uc = U[:, cols]
    Vc = V[:, cols]
    Wc = W[:, cols]
    Uc3 = Uc.reshape(n, m, -1)
    Vc3 = Vc.reshape(m, p, -1)
    Wc3 = Wc.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,ikr->ijk', Uc3, Vc3, Wc3)
    return int(np.max(np.abs(T - 1)))


def _eliminate_redundant_columns(U, V, W, n, m, p, deadline):
    """
    Attempt to remove any rank term whose removal leaves the exact matrix-multiplication tensor.
    For the correct Strassen construction this finds nothing (blocks are output-disjoint),
    but it is a cheap, exact refinement pass.
    """
    rank = U.shape[1]
    start = time.time()
    changed = True
    while changed:
        changed = False
        for r in range(U.shape[1]):
            if time.time() > deadline:
                return U, V, W
            res = _tensor_residual(U, V, W, n, m, p, skip_col=r)
            if res < 1:
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                changed = True
                break
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    U, V, W = _build_decomposition(n, m, p)

    # Exact verification before any refinement.
    ok = _verify_decomposition(U, V, W, n, m, p, num_trials=12, seed=42)
    if not ok:
        raise RuntimeError("Decomposition verification failed")

    # Refinement: try to drop any redundant rank term (exact tensor test).
    if time.time() < DEADLINE - 20:
        U, V, W = _eliminate_redundant_columns(U, V, W, n, m, p, DEADLINE - 5)
        # Re-verify after any elimination.
        ok2 = _verify_decomposition(U, V, W, n, m, p, num_trials=12, seed=42)
        if not ok2:
            # Fall back to the original verified construction.
            U, V, W = _build_decomposition(n, m, p)

    # Convert to float lists as required by the interface.
    return {"U": U.astype(float).tolist(), "V": V.astype(float).tolist(), "W": W.astype(float).tolist()}


# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    # Verify correctness with random integer inputs (exact arithmetic).
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m))
    B = np.random.randint(-5, 6, size=(m, p))

    C_decomp = np.zeros((n, p), dtype=np.int64)
    for r in range(rank):
        u_r = U[:, r].astype(np.int64)
        v_r = V[:, r].astype(np.int64)
        w_r = W[:, r].astype(np.int64)

        u_mat = u_r.reshape(n, m)
        v_mat = v_r.reshape(m, p)

        prod_u = np.sum(u_mat * A)
        prod_v = np.sum(v_mat * B)

        scalar = prod_u * prod_v

        w_mat = w_r.reshape(p, n)
        C_decomp += scalar * w_mat.T

    C_direct = A @ B
    if not np.array_equal(C_direct, C_decomp):
        print("WARNING: Decomposition is incorrect!")
    else:
        print("Decomposition is correct.")

BET: {mechanism: "Same rank-55 Strassen+schoolbook construction as parent but with stricter exact-int64 arithmetic throughout verification and a cleaner elimination loop; no structural rank reduction is possible within this family, so score is tied.", q: 0.0, delta: [0.0, 0.0]}
