"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _build_target_mask(n, m, p):
    """Precompute the target mask and index arrays for verification."""
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    idx_a, idx_b, idx_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = True
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    return target, np.array(idx_a), np.array(idx_b), np.array(idx_c)

def _verify(U, V, W, tol=1e-9, target_mask=None, idx_a=None, idx_b=None, idx_c=None):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Build tensor via sum of outer products
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    if target_mask is None:
        target_mask, idx_a, idx_b, idx_c = _build_target_mask(n, m, p)

    target_vals = T[idx_a, idx_b, idx_c]
    max_err = np.max(np.abs(target_vals - 1.0))
    non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0

    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5, target_mask=None, idx_a=None, idx_b=None, idx_c=None):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol,
               target_mask=target_mask, idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()

    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol,
               target_mask=target_mask, idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()

    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    Split B (3x8) into 4 blocks of 3x2.
    Each block: Strassen rank-7 for 2x2*2x2 part + 4 rank-1 terms for extra row.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))

    col_offset = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1

        # Strassen rank-7 for A[0:2,0:2] * B[0:2,k0:k1+1]
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1

        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1

        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1

        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1

        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1

        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1

        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1

        # 4 rank-1 terms: A[:,2] * B[2,k] -> C[:,k]
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1

    assert col_offset == 44, f"Expected 44, got {col_offset}"
    return U.tolist(), V.tolist(), W.tolist()

def _als_decompose(rank, max_iters=80, deadline=None, seed=None):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    Uses batched least-squares for speed.
    """
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(seed)

    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))

    # Precompute index structures
    # For updating U[a]: we need V[j*p+k] and W[k*n+i] for all k, where a = i*m+j
    # For updating V[b]: b = j*p+k, need U[i*m+j] and W[k*n+i] for all i
    # For updating W[c]: c = k*n+i, need U[i*m+j] and V[j*p+k] for all j

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # --- Update U ---
        for i in range(n):
            for j in range(m):
                a = i * m + j
                # For each k: V[j*p+k] * W[k*n+i], target = 1
                M = np.zeros((p, rank))
                rhs = np.ones(p)
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

        # --- Update V ---
        for j in range(m):
            for k in range(p):
                b = j * p + k
                # For each i: U[i*m+j] * W[k*n+i], target = 1
                M = np.zeros((n, rank))
                rhs = np.ones(n)
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

        # --- Update W ---
        for k in range(p):
            for i in range(n):
                c = k * n + i
                # For each j: U[i*m+j] * V[j*p+k], target = 1
                M = np.zeros((m, rank))
                rhs = np.ones(m)
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

        # Check convergence (every few iterations to save time)
        if it % 5 == 4 or it == max_iters - 1:
            T = np.einsum('ar,br,cr->abc', U, V, W)
            target = np.zeros((n*m, m*p, p*n), dtype=bool)
            idx_a, idx_b, idx_c = [], [], []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = True
                        idx_a.append(i*m+j)
                        idx_b.append(j*p+k)
                        idx_c.append(k*n+i)
            idx_a = np.array(idx_a); idx_b = np.array(idx_b); idx_c = np.array(idx_c)
            target_err = np.max(np.abs(T[idx_a, idx_b, idx_c] - 1.0))
            non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
            if target_err < 1e-8 and non_target_err < 1e-8:
                return U.tolist(), V.tolist(), W.tolist()

    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48

    # Precompute target mask
    target_mask, idx_a, idx_b, idx_c = _build_target_mask(n, m, p)

    # Schoolbook fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1

    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank

    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Step 1: Try the explicit rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6, target_mask=target_mask,
                   idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
            rounded = _round_and_verify(U44, V44, W44, tol=1e-5,
                                        target_mask=target_mask,
                                        idx_a=idx_a, idx_b=idx_b, idx_c=idx_c)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception:
        pass

    # Step 2: Try to find better rank via ALS
    if best_rank <= 44:
        # Search ranks from 30 up to best_rank-1
        for rank in range(30, best_rank):
            if time.time() > deadline:
                break

            found = False
            # Multiple random restarts
            max_trials = 5
            for trial in range(max_trials):
                if time.time() > deadline:
                    break
                seed = rank * 100 + trial
                result = _als_decompose(rank, max_iters=60, deadline=deadline, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6, target_mask=target_mask,
                               idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
                        rounded = _round_and_verify(U, V, W, tol=1e-5,
                                                    target_mask=target_mask,
                                                    idx_a=idx_a, idx_b=idx_b, idx_c=idx_c)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        found = True
                        break

            if found:
                # Try to go even lower
                continue
            else:
                # If we can't find rank, try next higher
                continue

    return {"U": best_U, "V": best_V, "W": best_W}
