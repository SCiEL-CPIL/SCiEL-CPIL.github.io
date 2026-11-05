import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication.

    Base construction (rank 36): split the 4x4 middle dimension into two 2x2
    blocks. Each block uses two Strassen 2x2x2 blocks (rank 7 each) plus one
    schoolbook 2x2x1 block (rank 4). Total = 2*(7+7+4) = 36.

    Refinement: attempt an exact rank reduction by finding a non-trivial linear
    combination of the 36 rank-1 terms whose tensor is identically zero. Because
    the two A-blocks share the same row indices (i in {0,1}) and the output W
    shares (k,i) entries across blocks, there may be a dependency allowing one
    term to be dropped. We solve for the null space of the coefficient tensor
    and, if found, subtract the dependency to zero out a column and drop it.
    """
    t0 = time.time()
    DEADLINE = 120.0  # evaluator kills at 150s; keep safety margin

    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    # ---- Build the target tensor T (ground truth) ----
    # T[a,b,c] = 1 iff (a%m == b//p) and (b%p == c//n) and (a//m == c%n)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for a in range(nm):
        i = a // m
        j = a % m
        for b in range(mp):
            jp = b // p
            k = b % p
            for c in range(pn):
                kp = c // n
                ip = c % n
                if j == jp and k == kp and i == ip:
                    T[a, b, c] = 1.0

    def verify(U, V, W):
        """Check that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T exactly (within tolerance)."""
        recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(U.shape[1]):
            recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        return np.allclose(recon, T, atol=1e-9)

    # ---- Base rank-36 construction ----
    rank = 36
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)

    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        for (i, j), val in u_coeffs.items():
            U[i * m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i, idx] += val

    def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
        # M1 = (A00 + A11) * (B00 + B11)
        add_term(idx_start,
                 {(0, a_row_offset): 1, (1, a_row_offset + 1): 1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                 {(c_col_offset, 0): 1, (c_col_offset + 1, 1): 1})
        # M2 = (A10 + A11) * B00
        add_term(idx_start + 1,
                 {(1, a_row_offset): 1, (1, a_row_offset + 1): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 1): 1, (c_col_offset + 1, 1): -1})
        # M3 = A00 * (B01 - B11)
        add_term(idx_start + 2,
                 {(0, a_row_offset): 1},
                 {(b_row_offset, c_col_offset + 1): 1, (b_row_offset + 1, c_col_offset + 1): -1},
                 {(c_col_offset + 1, 0): 1, (c_col_offset + 1, 1): 1})
        # M4 = A11 * (B10 - B00)
        add_term(idx_start + 3,
                 {(1, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                 {(c_col_offset, 0): 1, (c_col_offset + 1, 1): 1})
        # M5 = (A00 + A01) * B11
        add_term(idx_start + 4,
                 {(0, a_row_offset): 1, (0, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset + 1): 1},
                 {(c_col_offset, 0): -1, (c_col_offset + 1, 0): 1})
        # M6 = (A10 - A00) * (B00 + B01)
        add_term(idx_start + 5,
                 {(1, a_row_offset): 1, (0, a_row_offset): -1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset + 1): 1},
                 {(c_col_offset + 1, 1): 1})
        # M7 = (A01 - A11) * (B10 + B11)
        add_term(idx_start + 6,
                 {(0, a_row_offset + 1): 1, (1, a_row_offset + 1): -1},
                 {(b_row_offset + 1, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                 {(c_col_offset, 0): 1})

    def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
        add_term(idx_start,
                 {(0, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 0): 1})
        add_term(idx_start + 1,
                 {(0, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset): 1},
                 {(c_col_offset, 0): 1})
        add_term(idx_start + 2,
                 {(1, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 1): 1})
        add_term(idx_start + 3,
                 {(1, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset): 1},
                 {(c_col_offset, 1): 1})

    # Block 1: A[:, 0:2] * B[0:2, :]
    idx = 0
    strassen_block(0, 0, 0, idx); idx += 7
    strassen_block(0, 0, 2, idx); idx += 7
    schoolbook_2x2x1_block(0, 0, 4, idx); idx += 4
    # Block 2: A[:, 2:4] * B[2:4, :]
    strassen_block(2, 2, 0, idx); idx += 7
    strassen_block(2, 2, 2, idx); idx += 7
    schoolbook_2x2x1_block(2, 2, 4, idx); idx += 4

    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")

    # ---- Verify base construction ----
    if not verify(U, V, W):
        # Fallback: trivial rank-40 decomposition
        U_t = np.zeros((nm, 40), dtype=np.float64)
        V_t = np.zeros((mp, 40), dtype=np.float64)
        W_t = np.zeros((pn, 40), dtype=np.float64)
        cidx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i * m + j, cidx] = 1.0
                    V_t[j * p + k, cidx] = 1.0
                    W_t[k * n + i, cidx] = 1.0
                    cidx += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    # ---- Attempt exact rank reduction via null-space dependency ----
    if time.time() - t0 < DEADLINE - 20:
        U, V, W = _try_reduce_rank(U, V, W, T)

    # ---- Final verification with fallback ----
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank-36 (rebuild)
    U = np.zeros((nm, 36), dtype=np.float64)
    V = np.zeros((mp, 36), dtype=np.float64)
    W = np.zeros((pn, 36), dtype=np.float64)
    idx = 0
    strassen_block(0, 0, 0, idx); idx += 7
    strassen_block(0, 0, 2, idx); idx += 7
    schoolbook_2x2x1_block(0, 0, 4, idx); idx += 4
    strassen_block(2, 2, 0, idx); idx += 7
    strassen_block(2, 2, 2, idx); idx += 7
    schoolbook_2x2x1_block(2, 2, 4, idx); idx += 4
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_reduce_rank(U, V, W, T):
    """
    Try to find a linear dependency among the rank-1 terms:
    sum_r alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
    If found, subtract the dependency to zero out one column, then drop it.
    """
    rank = U.shape[1]
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    total = nm * mp * pn  # 8*20*10 = 1600

    # Build matrix M of shape (total, rank): M[(a,b,c), r] = U[a,r]*V[b,r]*W[c,r]
    # Flatten (a,b,c) -> a*mp*pn + b*pn + c
    M = np.zeros((total, rank), dtype=np.float64)
    for r in range(rank):
        # outer product of U[:,r], V[:,r], W[:,r]
        tensor_r = U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        M[:, r] = tensor_r.reshape(total)

    # Find null space of M (rows = tensor entries, cols = rank terms)
    # We want alpha such that M @ alpha = 0
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W

    # Null space = rows of Vt corresponding to near-zero singular values
    tol = 1e-10
    rank_sv = np.sum(S > tol)
    nullity = rank - rank_sv

    if nullity < 1:
        return U, V, W

    # Null space vectors are the last (rank - rank_sv) rows of Vt
    null_vec = Vt[rank_sv:, :]  # shape (nullity, rank)

    # For each null vector, try to zero out a column
    best_U, best_V, best_W = None, None, None
    for nv in null_vec:
        # Find the column with largest |alpha| to target for zeroing
        abs_alpha = np.abs(nv)
        target = int(np.argmax(abs_alpha))
        if abs_alpha[target] < 1e-12:
            continue

        # Subtract (alpha[target]) * (column target) from each column r:
        # New column r = column r - (alpha[r]/alpha[target]) * column target
        # This makes the new column target = 0
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        for r in range(rank):
            if r == target:
                continue
            factor = nv[r] / nv[target]
            U_new[:, r] -= factor * U_new[:, target]
            V_new[:, r] -= factor * V_new[:, target]
            W_new[:, r] -= factor * W_new[:, target]
        # Zero out target column
        U_new[:, target] = 0.0
        V_new[:, target] = 0.0
        W_new[:, target] = 0.0

        # Drop the target column
        keep = [r for r in range(rank) if r != target]
        U_new = U_new[:, keep]
        V_new = V_new[:, keep]
        W_new = W_new[:, keep]

        # Verify
        recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(U_new.shape[1]):
            recon += U_new[:, r][:, None, None] * V_new[:, r][None, :, None] * W_new[:, r][None, None, :]
        if np.allclose(recon, T, atol=1e-9):
            best_U, best_V, best_W = U_new, V_new, W_new
            break

    if best_U is not None:
        return best_U, best_V, best_W
    return U, V, W
# EVOLVE-BLOCK-END
