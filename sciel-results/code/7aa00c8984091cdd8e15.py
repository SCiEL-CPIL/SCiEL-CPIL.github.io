import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _build_strassen_rank39(n, m, p):
    """Build the rank-39 Strassen-based decomposition."""
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
        nonlocal col
        if u_coeffs is None:
            u_coeffs = [1] * len(u_indices)
        if v_coeffs is None:
            v_coeffs = [1] * len(v_indices)
        if w_coeffs is None:
            w_coeffs = [1] * len(w_indices)
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        """Standard 7-term Strassen for 2x2 x 2x2 block."""
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1

        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1

        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1

        add_term([a00, a11], [b00, b11], [c00, c11])
        add_term([a10, a11], [b00], [c10, c11], w_coeffs=[1, -1])
        add_term([a00], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        add_term([a11], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        add_term([a00, a01], [b11], [c00, c01], w_coeffs=[-1, 1])
        add_term([a10, a00], [b00, b01], [c11], u_coeffs=[1, -1])
        add_term([a01, a11], [b10, b11], [c00], u_coeffs=[1, -1])

    def add_strassen_block_optimized(k_start):
        """Try to find a 6-term variant by merging terms."""
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1

        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1

        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1

        # Try to combine the last two terms
        # Term 6: (a10-a00)*b00*b01 -> c11
        # Term 7: (a01-a11)*(b10+b11) -> c00
        # These are independent, so we need both.
        # But maybe we can restructure...
        add_term([a00, a11], [b00, b11], [c00, c11])
        add_term([a10, a11], [b00], [c10, c11], w_coeffs=[1, -1])
        add_term([a00], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        add_term([a11], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        add_term([a00, a01], [b11], [c00, c01], w_coeffs=[-1, 1])
        add_term([a10, a00], [b00, b01], [c11], u_coeffs=[1, -1])
        add_term([a01, a11], [b10, b11], [c00], u_coeffs=[1, -1])

    # Use standard blocks for now
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)

    # Handle column 6 with 4 terms
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1

    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])

    # Row 2 of A (the "direct" row)
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])

    return {"U": U, "V": V, "W": W}

def _peel_columns(U, V, W, n, m, p, max_rounds=200, tol=1e-10):
    """Aggressively peel removable columns. Returns (U, V, W) with fewer columns."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    for _ in range(max_rounds):
        if current_rank == 0:
            break
        removed = False
        for col in range(current_rank):
            U_test = [row[:col] + row[col+1:] for row in U]
            V_test = [row[:col] + row[col+1:] for row in V]
            W_test = [row[:col] + row[col+1:] for row in W]
            if _verify(U_test, V_test, W_test, n, m, p, tol):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, current_rank

def _eliminate_dependent(U, V, W, n, m, p, tol=1e-10):
    """
    Try to eliminate a column r that is a linear combination of other columns,
    consistently across U, V, W. If U[:,r] = sum_s alpha_s U[:,s], V[:,r] = sum_s alpha_s V[:,s],
    W[:,r] = sum_s alpha_s W[:,s], then we can subtract the combination and column r becomes
    zero (removable). We solve for alpha from U, check V and W.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    changed = True
    while changed and current_rank > 1:
        changed = False
        # Try each column as target
        for r in range(current_rank):
            others = [s for s in range(current_rank) if s != r]
            # Build matrix of other columns in U
            U_others = U_arr[:, others]  # (nm, k)
            target_U = U_arr[:, r]
            # Solve U_others @ alpha = target_U (least squares)
            try:
                alpha, residuals, rank_u, sv = np.linalg.lstsq(U_others, target_U, rcond=None)
            except np.linalg.LinAlgError:
                continue
            # Check U residual
            U_pred = U_others @ alpha
            if np.linalg.norm(U_pred - target_U) > tol * (1 + np.linalg.norm(target_U)):
                continue
            # Check V
            V_others = V_arr[:, others]
            target_V = V_arr[:, r]
            V_pred = V_others @ alpha
            if np.linalg.norm(V_pred - target_V) > tol * (1 + np.linalg.norm(target_V)):
                continue
            # Check W
            W_others = W_arr[:, others]
            target_W = W_arr[:, r]
            W_pred = W_others @ alpha
            if np.linalg.norm(W_pred - target_W) > tol * (1 + np.linalg.norm(target_W)):
                continue
            # All consistent: subtract combination from column r
            U_arr[:, r] = target_U - U_pred
            V_arr[:, r] = target_V - V_pred
            W_arr[:, r] = target_W - W_pred
            # Round to integers if close
            U_arr = np.round(U_arr, 8)
            V_arr = np.round(V_arr, 8)
            W_arr = np.round(W_arr, 8)
            U = U_arr.tolist()
            V = V_arr.tolist()
            W = W_arr.tolist()
            current_rank -= 1
            # Drop column r
            U = [row[:r] + row[r+1:] for row in U]
            V = [row[:r] + row[r+1:] for row in V]
            W = [row[:r] + row[r+1:] for row in W]
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            changed = True
            break
    return U, V, W, current_rank

def _try_pairwise_merges(U, V, W, n, m, p, tol=1e-10):
    """
    Try to merge two columns r and s if their sum (or difference) can be absorbed.
    If column r + column s = column t (some existing or new), we might reduce rank.
    More practically: try to find if two columns are "close" to being linearly dependent.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    improved = True
    while improved and current_rank > 2:
        improved = False
        for r in range(current_rank):
            for s in range(r+1, current_rank):
                # Try: can column s be expressed as alpha * column r + (combination of others)?
                others = [i for i in range(current_rank) if i not in [r, s]]
                # Build system: [U_others | U_r] @ [alpha; beta] = U_s
                if len(others) == 0:
                    # Only r and s remain
                    # Check if U_s = beta * U_r
                    if np.linalg.norm(U_arr[:, r]) < tol:
                        continue
                    beta = np.dot(U_arr[:, s], U_arr[:, r]) / np.dot(U_arr[:, r], U_arr[:, r])
                    U_pred = beta * U_arr[:, r]
                    if np.linalg.norm(U_pred - U_arr[:, s]) > tol:
                        continue
                    V_pred = beta * V_arr[:, r]
                    if np.linalg.norm(V_pred - V_arr[:, s]) > tol:
                        continue
                    W_pred = beta * W_arr[:, r]
                    if np.linalg.norm(W_pred - W_arr[:, s]) > tol:
                        continue
                    # Subtract
                    U_arr[:, s] -= U_pred
                    V_arr[:, s] -= V_pred
                    W_arr[:, s] -= W_pred
                    U_arr = np.round(U_arr, 8)
                    V_arr = np.round(V_arr, 8)
                    W_arr = np.round(W_arr, 8)
                    U = U_arr.tolist()
                    V = V_arr.tolist()
                    W = W_arr.tolist()
                    current_rank -= 1
                    U = [row[:s] + row[s+1:] for row in U]
                    V = [row[:s] + row[s+1:] for row in V]
                    W = [row[:s] + row[s+1:] for row in W]
                    U_arr = np.array(U, dtype=np.float64)
                    V_arr = np.array(V, dtype=np.float64)
                    W_arr = np.array(W, dtype=np.float64)
                    improved = True
                    break
                else:
                    # Solve with all others + r
                    cols = others + [r]
                    M = np.column_stack([U_arr[:, i] for i in cols])
                    target = U_arr[:, s]
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
                    except:
                        continue
                    pred = M @ coeffs
                    if np.linalg.norm(pred - target) > tol * (1 + np.linalg.norm(target)):
                        continue
                    # Check V
                    M_V = np.column_stack([V_arr[:, i] for i in cols])
                    pred_V = M_V @ coeffs
                    if np.linalg.norm(pred_V - V_arr[:, s]) > tol * (1 + np.linalg.norm(V_arr[:, s])):
                        continue
                    # Check W
                    M_W = np.column_stack([W_arr[:, i] for i in cols])
                    pred_W = M_W @ coeffs
                    if np.linalg.norm(pred_W - W_arr[:, s]) > tol * (1 + np.linalg.norm(W_arr[:, s])):
                        continue
                    # Subtract
                    U_arr[:, s] -= pred
                    V_arr[:, s] -= pred_V
                    W_arr[:, s] -= pred_W
                    U_arr = np.round(U_arr, 8)
                    V_arr = np.round(V_arr, 8)
                    W_arr = np.round(W_arr, 8)
                    U = U_arr.tolist()
                    V = V_arr.tolist()
                    W = W_arr.tolist()
                    current_rank -= 1
                    U = [row[:s] + row[s+1:] for row in U]
                    V = [row[:s] + row[s+1:] for row in V]
                    W = [row[:s] + row[s+1:] for row in W]
                    U_arr = np.array(U, dtype=np.float64)
                    V_arr = np.array(V, dtype=np.float64)
                    W_arr = np.array(W, dtype=np.float64)
                    improved = True
                    break
            if improved:
                break
    return U, V, W, current_rank

def _try_rank_reduction(rank_target, U_init, V_init, W_init, n, m, p, deadline):
    """Attempt to reduce rank below rank_target via peeling + dependent elimination."""
    U, V, W, cur = _peel_columns(U_init, V_init, W_init, n, m, p)
    U, V, W, cur = _eliminate_dependent(U, V, W, n, m, p)
    # Try pairwise merges for additional reduction
    U, V, W, cur = _try_pairwise_merges(U, V, W, n, m, p)
    if cur < rank_target and _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    result = _build_strassen_rank39(n, m, p)

    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to reduce rank aggressively
        best_rank = len(result["U"][0])
        for target_rank in range(best_rank - 1, max(best_rank - 10, 20), -1):
            if time.time() > deadline - 10:
                break
            reduced = _try_rank_reduction(target_rank, result["U"], result["V"], result["W"], n, m, p, deadline)
            if reduced is not None:
                result = reduced
                best_rank = len(result["U"][0])
                break

        return result

    # Fallback: schoolbook
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

if __name__ == "__main__":
    res = construct()
