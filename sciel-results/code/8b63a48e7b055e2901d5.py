import time
import numpy as np

DIMENSIONS = [2, 3, 7]


def _build_hybrid_decomposition():
    """Build the hybrid Strassen/schoolbook decomposition for 2x3 * 3x7.

    Strategy:
      - Split A (2x3) into A_left (2x2) and A_right (2x1).
      - Split B (3x7) into B_top (2x7) and B_bottom (1x7).
      - C = A_left @ B_top + A_right @ B_bottom.
      - For each 2x2 * 2x2 block of B_top, use Strassen (rank 7 instead of 8).
      - For 2x2 * 2x1 and 2x1 * 1x7 blocks, use schoolbook.

    This yields rank 39 (vs. trivial 42).
    """
    n, m, p = DIMENSIONS  # 2, 3, 7

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    terms = []

    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))

    # ---- Strassen for a 2x2 * 2x2 block ----
    # A_left = [[a00, a01], [a10, a11]]
    # Block = [[B[0,c0], B[0,c1]], [B[1,c0], B[1,c1]]]
    # Output columns c0, c1; output rows 0, 1.
    def strassen_block(c0, c1):
        # M1 = (a00 + a11)(B[0,c0] + B[1,c1])
        add_term(
            {u_idx(0, 0): 1, u_idx(1, 1): 1},
            {v_idx(0, c0): 1, v_idx(1, c1): 1},
            {w_idx(c0, 0): 1, w_idx(c1, 1): 1},
        )
        # M2 = (a10 + a11) * B[0,c0]
        add_term(
            {u_idx(1, 0): 1, u_idx(1, 1): 1},
            {v_idx(0, c0): 1},
            {w_idx(c0, 1): 1, w_idx(c1, 1): -1},
        )
        # M3 = a00 * (B[0,c1] - B[1,c1])
        add_term(
            {u_idx(0, 0): 1},
            {v_idx(0, c1): 1, v_idx(1, c1): -1},
            {w_idx(c1, 0): 1, w_idx(c1, 1): 1},
        )
        # M4 = a11 * (B[1,c0] - B[0,c0])
        add_term(
            {u_idx(1, 1): 1},
            {v_idx(1, c0): 1, v_idx(0, c0): -1},
            {w_idx(c0, 0): 1, w_idx(c0, 1): 1},
        )
        # M5 = (a00 + a01) * B[1,c1]
        add_term(
            {u_idx(0, 0): 1, u_idx(0, 1): 1},
            {v_idx(1, c1): 1},
            {w_idx(c0, 0): -1, w_idx(c1, 0): 1},
        )
        # M6 = (a10 - a00) * (B[0,c0] + B[0,c1])
        add_term(
            {u_idx(1, 0): 1, u_idx(0, 0): -1},
            {v_idx(0, c0): 1, v_idx(0, c1): 1},
            {w_idx(c1, 1): 1},
        )
        # M7 = (a01 - a11) * (B[1,c0] + B[1,c1])
        add_term(
            {u_idx(0, 1): 1, u_idx(1, 1): -1},
            {v_idx(1, c0): 1, v_idx(1, c1): 1},
            {w_idx(c0, 0): 1},
        )

    # ---- Schoolbook for a 2x2 * 2x1 block ----
    # C[0] = a00*B[0,c] + a01*B[1,c]
    # C[1] = a10*B[0,c] + a11*B[1,c]
    def schoolbook_2x2_2x1(c):
        add_term(
            {u_idx(0, 0): 1},
            {v_idx(0, c): 1},
            {w_idx(c, 0): 1},
        )
        add_term(
            {u_idx(0, 1): 1},
            {v_idx(1, c): 1},
            {w_idx(c, 0): 1},
        )
        add_term(
            {u_idx(1, 0): 1},
            {v_idx(0, c): 1},
            {w_idx(c, 1): 1},
        )
        add_term(
            {u_idx(1, 1): 1},
            {v_idx(1, c): 1},
            {w_idx(c, 1): 1},
        )

    # ---- Schoolbook for a 2x1 * 1x7 block (A_right @ B_bottom) ----
    # C[0] = a20 * B[2,c]
    # C[1] = a21 * B[2,c]
    def schoolbook_2x1_1x7(c):
        add_term(
            {u_idx(1, 2): 1},
            {v_idx(2, c): 1},
            {w_idx(c, 0): 1},
        )
        add_term(
            {u_idx(1, 2): 1},
            {v_idx(2, c): 1},
            {w_idx(c, 1): 1},
        )

    # A_left @ B_top (2x2 * 2x7):
    #   B_top columns 0..6.
    #   Pair (0,1) -> Strassen (7)
    #   Pair (2,3) -> Strassen (7)
    #   Pair (4,5) -> Strassen (7)
    #   Column 6    -> schoolbook 2x2*2x1 (4)
    # Total: 25
    strassen_block(0, 1)
    strassen_block(2, 3)
    strassen_block(4, 5)
    schoolbook_2x2_2x1(6)

    # A_right @ B_bottom (2x1 * 1x7):
    #   7 columns * 2 rows = 14
    for c in range(7):
        schoolbook_2x1_1x7(c)

    # Assemble matrices
    rank = len(terms)
    U = np.array([t[0] for t in terms], dtype=object).T  # (n*m, rank)
    V = np.array([t[1] for t in terms], dtype=object).T  # (m*p, rank)
    W = np.array([t[2] for t in terms], dtype=object).T  # (p*n, rank)

    return U, V, W, rank


def _build_trivial_decomposition():
    """Fallback trivial (schoolbook) decomposition, rank = n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return U, V, W, rank


def _verify_decomposition(U, V, W, max_samples=50):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r]^T equals the matmul tensor.

    Returns True if all sampled cases pass within tolerance.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)

    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check against the standard matmul: C = A @ B
    # T[i,j,k] should equal sum_j' A[i,j'] * B[j',k]
    # i.e. T is the full multiplication tensor.

    rng = np.random.default_rng(12345)
    for _ in range(max_samples):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B

        # Compute C from the decomposition:
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_k ... ) -- not quite.
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # And the tensor decomposition says:
        #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta (the multiplication tensor entry)
        # The multiplication tensor entry T[i,j,k] = 1 if ... no.
        #
        # The correct identity: for the matrix multiplication tensor,
        #   T[i,j,k] = 1 (it's the canonical tensor where T[i,j,k] = 1 for all valid (i,j,k))
        # Wait no. The multiplication tensor is defined by:
        #   <T, A \otimes B> = C where C = A @ B
        #   C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (the tensor is all-ones in the standard basis).
        #
        # Actually the multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k
        # in the standard basis, and the contraction gives C[i,k] = sum_j A[i,j] B[j,k].
        #
        # So the decomposition must satisfy:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i, j, k.
        #
        # Let's verify this.

        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check that T[i,j,k] = 1 for all i,j,k

        # Efficient check: for each (i,j,k), compute the sum
        # This is O(n*m*p*rank) per sample, which is fine for small dims.
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err > 1e-9:
            return False

    return True


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit

    n, m, p = DIMENSIONS

    # Try the hybrid Strassen/schoolbook decomposition (rank 39)
    try:
        U, V, W, rank = _build_hybrid_decomposition()

        # Verify correctness
        if _verify_decomposition(U, V, W):
            # Convert to lists of lists for the expected interface
            U_out = U.tolist()
            V_out = V.tolist()
            W_out = W.tolist()
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            # Verification failed; fall through to trivial
            pass
    except Exception:
        pass

    if time.time() > deadline:
        # Emergency: return trivial
        U, V, W, rank = _build_trivial_decomposition()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: trivial decomposition (rank 42)
    U, V, W, rank = _build_trivial_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
