"""Matrix multiplication 4x4x8: search for a sparse rank-20 4x4x4 core, then embed."""
import time
import numpy as np

_DIMENSIONS = (4, 4, 8)
_START = time.time()
_DEADLINE = _START + 135  # 15s safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _verify(U, V, W, n, m, p, n_trials=8, tol=1e-8):
    """Verify decomposition: C[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    rank = U.shape[1]
    for trial in range(n_trials):
        np.random.seed(1000 + trial * 7919)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            C_decomp += w.T * (u @ v)
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    return True


def _verify_fast(U, V, W, n, m, p, n_trials=12, tol=1e-7):
    """Faster verification using einsum-like accumulation."""
    rank = U.shape[1]
    for trial in range(n_trials):
        np.random.seed(2000 + trial * 104729)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        # Vectorized: for each r, C += W[:,r].T @ (U[:,r] @ V[:,r])
        # U[:,r] is (n*m,) -> reshape (n,m); V[:,r] is (m*p,) -> reshape (m,p)
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            C_decomp += w_mat.T @ (u_mat @ v_mat)
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    return True


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _build_4x4x8_from_core(U4, V4, W4):
    """
    Embed a 4x4x4 core (rank r) into 4x4x8.
    A is 4x4, B is 4x8, C is 4x8.
    Split B into B1 (4x4, cols 0-3) and B2 (4x4, cols 4-7).
    C = A*B1 + A*B2, where each product is 4x4.
    Use the same core for both halves.
    Rank = 2*r.
    """
    r = U4.shape[1]
    rank = 2 * r
    n, m, p = 4, 4, 8
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # First half: B[:, 0:4] -> C[:, 0:4]
    for col in range(r):
        U[:, col] = U4[:, col]
        V[:m * 4, col] = V4[:, col]
        W[:4 * n, col] = W4[:, col]

    # Second half: B[:, 4:8] -> C[:, 4:8]
    for col in range(r):
        c2 = r + col
        U[:, c2] = U4[:, col]
        V[m * 4:, c2] = V4[:, col]
        W[4 * n:, c2] = W4[:, col]

    return U, V, W


def _random_sparse_init(rank, n, m, p, rng):
    """
    Initialize a sparse 4x4x4 decomposition.
    Each term r has exactly one non-zero in U, one in V, one in W.
    The pattern (which index is non-zero) defines the "structure".
    """
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for r in range(rank):
        ui = rng.randint(0, n * m)
        vi = rng.randint(0, m * p)
        wi = rng.randint(0, p * n)
        U[ui, r] = rng.choice([-1.0, 1.0])
        V[vi, r] = rng.choice([-1.0, 1.0])
        W[wi, r] = rng.choice([-1.0, 1.0])

    return U, V, W


def _compute_residual(U, V, W, A, B, n, m, p):
    """Compute C_decomp - C_expected for given A, B."""
    C_expected = A @ B
    C_decomp = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n)
        C_decomp += w_mat.T @ (u_mat @ v_mat)
    return C_decomp - C_expected


def _local_search_4x4x4(rank, max_iters, rng, time_budget):
    """
    Local search for a rank-r 4x4x4 decomposition.
    Strategy: start from random sparse init, then iteratively:
    1. Compute residual on a few random (A,B) pairs.
    2. For each term, try changing the non-zero index and sign.
    3. Accept if residual decreases.
    Also try adding/removing non-zeros (up to 2 per factor per term).
    """
    n, m, p = 4, 4, 4
    U, V, W = _random_sparse_init(rank, n, m, p, rng)

    # Pre-generate test matrices
    n_tests = 6
    test_As = []
    test_Bs = []
    for t in range(n_tests):
        rng2 = np.random.RandomState(5000 + t * 31)
        test_As.append(rng2.randn(n, m))
        test_Bs.append(rng2.randn(m, p))

    def total_residual():
        tot = 0.0
        for t in range(n_tests):
            res = _compute_residual(U, V, W, test_As[t], test_Bs[t], n, m, p)
            tot += np.sum(res * res)
        return tot

    best_res = total_residual()
    # Print for debugging
    # print(f"  Initial residual: {best_res:.6f}")

    for it in range(max_iters):
        if _elapsed() > time_budget:
            break

        # Pick a random term
        r = rng.randint(0, rank)

        # Current non-zero positions
        u_nz = np.nonzero(U[:, r])[0]
        v_nz = np.nonzero(V[:, r])[0]
        w_nz = np.nonzero(W[:, r])[0]

        # Try perturbing one factor of this term
        factor = rng.randint(0, 3)  # 0=U, 1=V, 2=W
        if factor == 0:
            old_val = U[u_nz, r].copy() if len(u_nz) > 0 else np.array([0.0])
            old_idx = u_nz.copy() if len(u_nz) > 0 else np.array([])
            # New random position
            new_idx = rng.randint(0, n * m)
            new_sign = rng.choice([-1.0, 1.0])
            U[u_nz, r] = 0.0
            U[new_idx, r] = new_sign
            new_res = total_residual()
            if new_res < best_res - 1e-12:
                best_res = new_res
            else:
                U[new_idx, r] = 0.0
                if len(old_idx) > 0:
                    U[old_idx, r] = old_val
        elif factor == 1:
            old_val = V[v_nz, r].copy() if len(v_nz) > 0 else np.array([0.0])
            old_idx = v_nz.copy() if len(v_nz) > 0 else np.array([])
            new_idx = rng.randint(0, m * p)
            new_sign = rng.choice([-1.0, 1.0])
            V[v_nz, r] = 0.0
            V[new_idx, r] = new_sign
            new_res = total_residual()
            if new_res < best_res - 1e-12:
                best_res = new_res
            else:
                V[new_idx, r] = 0.0
                if len(old_idx) > 0:
                    V[old_idx, r] = old_val
        else:
            old_val = W[w_nz, r].copy() if len(w_nz) > 0 else np.array([0.0])
            old_idx = w_nz.copy() if len(w_nz) > 0 else np.array([])
            new_idx = rng.randint(0, p * n)
            new_sign = rng.choice([-1.0, 1.0])
            W[w_nz, r] = 0.0
            W[new_idx, r] = new_sign
            new_res = total_residual()
            if new_res < best_res - 1e-12:
                best_res = new_res
            else:
                W[new_idx, r] = 0.0
                if len(old_idx) > 0:
                    W[old_idx, r] = old_val

        # Occasionally try flipping sign of an existing non-zero
        if rng.random() < 0.3:
            factor = rng.randint(0, 3)
            if factor == 0 and len(u_nz) > 0:
                idx = u_nz[rng.randint(0, len(u_nz))]
                U[idx, r] *= -1.0
                new_res = total_residual()
                if new_res < best_res - 1e-12:
                    best_res = new_res
                else:
                    U[idx, r] *= -1.0
            elif factor == 1 and len(v_nz) > 0:
                idx = v_nz[rng.randint(0, len(v_nz))]
                V[idx, r] *= -1.0
                new_res = total_residual()
                if new_res < best_res - 1e-12:
                    best_res = new_res
                else:
                    V[idx, r] *= -1.0
            elif factor == 2 and len(w_nz) > 0:
                idx = w_nz[rng.randint(0, len(w_nz))]
                W[idx, r] *= -1.0
                new_res = total_residual()
                if new_res < best_res - 1e-12:
                    best_res = new_res
                else:
                    W[idx, r] *= -1.0

    return U, V, W, best_res


def _try_rank20_444(rng, time_budget):
    """Try to find a rank-20 4x4x4 decomposition via local search."""
    rank = 20
    n_restarts = 50
    max_iters_per_restart = 3000

    best_U, best_V, best_W = None, None, None
    best_score = np.inf

    for restart in range(n_restarts):
        if _elapsed() > time_budget:
            break

        U, V, W, score = _local_search_4x4x4(
            rank, max_iters_per_restart, rng, time_budget
        )

        if score < best_score:
            best_score = score
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # Check if we found an exact solution
        if best_score < 1e-10:
            # Verify thoroughly
            if _verify_fast(best_U, best_V, best_W, 4, 4, 4, n_trials=20, tol=1e-9):
                return best_U, best_V, best_W

    if best_score < 1e-10:
        if _verify_fast(best_U, best_V, best_W, 4, 4, 4, n_trials=20, tol=1e-9):
            return best_U, best_V, best_W

    return None, None, None


def _try_rank21_444(rng, time_budget):
    """Try to find a rank-21 4x4x4 decomposition via local search."""
    rank = 21
    n_restarts = 30
    max_iters_per_restart = 2000

    best_U, best_V, best_W = None, None, None
    best_score = np.inf

    for restart in range(n_restarts):
        if _elapsed() > time_budget:
            break

        U, V, W, score = _local_search_4x4x4(
            rank, max_iters_per_restart, rng, time_budget
        )

        if score < best_score:
            best_score = score
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if best_score < 1e-10:
            if _verify_fast(best_U, best_V, best_W, 4, 4, 4, n_trials=20, tol=1e-9):
                return best_U, best_V, best_W

    if best_score < 1e-10:
        if _verify_fast(best_U, best_V, best_W, 4, 4, 4, n_trials=20, tol=1e-9):
            return best_U, best_V, best_W

    return None, None, None


def construct():
    n, m, p = _DIMENSIONS
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 128

    # Strategy 1: Try to find a rank-20 4x4x4 core -> rank-40 for 4x4x8
    # This would be a massive improvement over 98.
    time_budget_1 = _START + 100  # Give 100s for rank-20 search
    if _elapsed() < time_budget_1 - 5:
        rng = np.random.RandomState(42)
        U4, V4, W4 = _try_rank20_444(rng, time_budget_1)
        if U4 is not None:
            U, V, W = _build_4x4x8_from_core(U4, V4, W4)
            if _verify(U, V, W, n, m, p, n_trials=10, tol=1e-8):
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Strategy 2: Try rank-21 4x4x4 core -> rank-42 for 4x4x8
    time_budget_2 = _START + 120
    if _elapsed() < time_budget_2 - 5:
        rng = np.random.RandomState(123)
        U4, V4, W4 = _try_rank21_444(rng, time_budget_2)
        if U4 is not None:
            U, V, W = _build_4x4x8_from_core(U4, V4, W4)
            if _verify(U, V, W, n, m, p, n_trials=10, tol=1e-8):
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: Strassen-based rank-98
    if _elapsed() < _DEADLINE - 10:
        try:
            U, V, W = _build_strassen_4x4x8()
            if _verify(U, V, W, n, m, p, n_trials=5, tol=1e-8):
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
        except Exception:
            pass

    # Fallback: Schoolbook rank-128
    if best_U is None or _elapsed() < _DEADLINE - 5:
        U, V, W = _build_schoolbook(n, m, p)
        if best_U is None or U.shape[1] < best_rank:
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    rank = 7
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)

    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0

    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0

    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0

    U[3, 3] = 1.0
    V[1, 3] = 1.0; V[2, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0

    U[0, 4] = 1.0; U[2, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0

    U[1, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[2, 5] = 1.0
    W[3, 5] = 1.0

    U[2, 6] = 1.0; U[3, 6] = -1.0
    V[1, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = -1.0; W[3, 6] = 1.0

    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using Strassen at two levels. Rank = 49."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)

    U2, V2, W2 = _strassen_2x2x2()

    def a_block_idx(bi, bj, li, lj):
        gi = bi * 2 + li
        gj = bj * 2 + lj
        return gi * 4 + gj

    def b_block_idx(bi, bj, lj, lk):
        gi = bi * 2 + lj
        gj = bj * 2 + lk
        return gi * 4 + gj

    def c_block_idx(ci, cj, lk, li):
        gi = ci * 2 + lk
        gj = cj * 2 + li
        return gi * 4 + gj

    outer_A = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): 1.0},
        {(0,0): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,1): 1.0, (0,0): -1.0},
        {(1,0): 1.0, (1,1): -1.0},
    ]
    outer_B = [
        {(0,0): 1.0, (1,1): 1.0},
        {(0,0): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,0): -1.0},
        {(1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,1): 1.0, (1,1): 1.0},
    ]
    outer_C = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): -1.0, (1,1): 1.0},
    ]

    for t in range(7):
        for s in range(7):
            g = t * 7 + s
            for li in range(2):
                for lj in range(2):
                    coeff = U2[li * 2 + lj, s]
                    if coeff != 0:
                        for (bi, bj), a_coeff in outer_A[t].items():
                            idx = a_block_idx(bi, bj, li, lj)
                            U[idx, g] += coeff * a_coeff
            for lj in range(2):
                for lk in range(2):
                    coeff = V2[lj * 2 + lk, s]
                    if coeff != 0:
                        for (bi, bj), b_coeff in outer_B[t].items():
                            idx = b_block_idx(bi, bj, lj, lk)
                            V[idx, g] += coeff * b_coeff
            for lk in range(2):
                for li in range(2):
                    coeff = W2[lk * 2 + li, s]
                    if coeff != 0:
                        for (ci, cj), c_coeff in outer_C[t].items():
                            idx = c_block_idx(ci, cj, lk, li)
                            W[idx, g] += coeff * c_coeff

    return U, V, W


def _build_strassen_4x4x8():
    """Build 4x4x8 decomposition using Strassen. Rank = 98."""
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4x4_strassen()
    rank4 = U4.shape[1]
    rank = 2 * rank4

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for r in range(rank4):
        U[:, r] = U4[:, r]
        V[:m * 4, r] = V4[:, r]
        W[:4 * n, r] = W4[:, r]

    for r in range(rank4):
        col = rank4 + r
        U[:, col] = U4[:, r]
        V[m * 4:, col] = V4[:, col]
        W[4 * n:, col] = W4[:, col]

    return U, V, W
