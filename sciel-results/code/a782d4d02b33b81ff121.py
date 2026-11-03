import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _remaining():
    return _DEADLINE - time.time()


def _verify(U, V, W, n, m, p, rank):
    """Exact verification: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    acc += U[ui][r] * V[vj][r] * W[wk][r]
                if acc != 1:
                    return False
    return True


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.

    Strategy:
      1. Build a rank-36 block Strassen decomposition.
      2. Search for rank reduction by identifying linearly dependent columns
         or columns that can be removed/merged while preserving exactness.
      3. Return the best (lowest-rank) valid decomposition found.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank_base = 36

    # ---- Build the base rank-36 Strassen decomposition ----
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M2 = (A10 + A11) * B00
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        c += 1
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M4 = A11 * (B10 - B00)
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        c += 1
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        c += 1
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        col = c
        return c

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        col = c
        return c

    # Four Strassen 2x2x2 blocks
    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    # Schoolbook for the last column
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank_base, f"Expected {rank_base}, got {col}"

    # Verify base decomposition
    assert _verify(U, V, W, n, m, p, rank_base), "Base decomposition failed verification"

    best_rank = rank_base
    best_U, best_V, best_W = U, V, W

    # ---- Rank reduction search ----
    # Try removing individual columns and checking if the decomposition still holds.
    # A column r can be removed if for every (i,j,k):
    #   sum_{r' != r} U[i,j,r']*V[j,k,r']*W[k,i,r'] == 1
    # Equivalently, the column r must not contribute to any (i,j,k) triple.
    # More generally, try small subsets of columns to remove.

    if _remaining() > 2.0:
        # Build numpy arrays for faster checks
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)

        # Precompute the full tensor: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Shape: (n, m, p)
        T = np.zeros((n, m, p), dtype=np.int64)
        for r in range(rank_base):
            T += U_np[:, r, None, None] * V_np[None, :, r, None] * W_np[None, None, :, r]
            # Actually let's do it properly
        # Recompute properly
        T = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    T[i, j, k] = np.sum(U_np[ui] * V_np[vj] * W_np[wk])
        assert np.all(T == 1), "Tensor verification failed"

        # For each column r, compute its contribution to T
        # contrib_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
        # A column can be removed if its contribution is zero everywhere,
        # OR if removing it still leaves T == 1 (which requires the contribution to be 0).
        
        # Actually, we need sum over r' != r of contrib_r'[i,j,k] = 1 for all (i,j,k)
        # This means contrib_r[i,j,k] must be 0 for all (i,j,k).
        
        # Let's check which columns have zero contribution
        zero_cols = []
        for r in range(rank_base):
            is_zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = i * m + j
                        vj = j * p + k
                        wk = k * n + i
                        if U_np[ui, r] * V_np[vj, r] * W_np[wk, r] != 0:
                            is_zero = False
                            break
                    if not is_zero:
                        break
                if not is_zero:
                    break
            if is_zero:
                zero_cols.append(r)

        # Remove zero-contribution columns
        if zero_cols:
            keep = [r for r in range(rank_base) if r not in zero_cols]
            new_rank = len(keep)
            if new_rank < best_rank:
                new_U = [U[r] for r in keep]
                new_V = [V[r] for r in keep]
                new_W = [W[r] for r in keep]
                if _verify(new_U, new_V, new_W, n, m, p, new_rank):
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Now try pairs of columns: can we remove two columns whose combined
        # contribution is zero? This requires solving a system.
        # For each pair (r1, r2), check if there exist alpha, beta (non-zero)
        # such that alpha*contrib_r1 + beta*contrib_r2 = 0 everywhere.
        # This is a linear dependency check.
        
        # Build the contribution matrix: for each r, flatten contrib_r to a vector of length n*m*p
        nmk = n * m * p
        contribs = np.zeros((rank_base, nmk), dtype=np.int64)
        for r in range(rank_base):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = i * m + j
                        vj = j * p + k
                        wk = k * n + i
                        contribs[r, i * m * p + j * p + k] = U_np[ui, r] * V_np[vj, r] * W_np[wk, r]

        # Check for linear dependencies among the remaining columns
        # If columns r1, r2, ..., rk are linearly dependent with integer coefficients,
        # we might be able to reduce rank.
        
        # For simplicity, check pairs: for each pair (r1, r2) not in zero_cols,
        # check if contribs[r1] and contribs[r2] are linearly dependent.
        if _remaining() > 1.0:
            removed_set = set(zero_cols)
            current_rank = rank_base - len(removed_set)
            
            for r1 in range(rank_base):
                if r1 in removed_set:
                    continue
                for r2 in range(r1 + 1, rank_base):
                    if r2 in removed_set:
                        continue
                    # Check if contribs[r1] is a scalar multiple of contribs[r2]
                    # or vice versa, or if their sum/difference is zero
                    if np.array_equal(contribs[r1], contribs[r2]):
                        # They are identical, so we can remove one
                        removed_set.add(r2)
                        current_rank -= 1
                        if current_rank < best_rank:
                            keep = [r for r in range(rank_base) if r not in removed_set]
                            new_U = [U[r] for r in keep]
                            new_V = [V[r] for r in keep]
                            new_W = [W[r] for r in keep]
                            if _verify(new_U, new_V, new_W, n, m, p, len(keep)):
                                best_rank = len(keep)
                                best_U, best_V, best_W = new_U, new_V, new_W
                    elif np.array_equal(contribs[r1], -contribs[r2]):
                        # They cancel each other, remove both
                        removed_set.add(r1)
                        removed_set.add(r2)
                        current_rank -= 2
                        if current_rank < best_rank:
                            keep = [r for r in range(rank_base) if r not in removed_set]
                            new_U = [U[r] for r in keep]
                            new_V = [V[r] for r in keep]
                            new_W = [W[r] for r in keep]
                            if _verify(new_U, new_V, new_W, n, m, p, len(keep)):
                                best_rank = len(keep)
                                best_U, best_V, best_W = new_U, new_V, new_W

    return {"U": best_U, "V": best_V, "W": best_W}
