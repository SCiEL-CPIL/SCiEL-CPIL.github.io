def construct():
    deadline = time.time() + 110.0  # safety margin under 120s cap
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Guaranteed valid schoolbook solution (rank 54).
    U, V, W = _schoolbook(n, m, p)
    assert _error(U, V, W, T) < 1e-9, "schoolbook failed"
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try to beat schoolbook with ALS for decreasing ranks.
    # Lower bound on tensor rank for n x m x p is at least max(n*m, m*p, p*n)
    # and at least max(n, m, p) etc.  Known: rank >= max(n*m, m*p, p*n) is NOT
    # true in general; a safe lower bound is max(n, m, p).  We'll just scan down.
    lower = max(n, m, p)  # = 9
    for r in range(lower, best_rank):
        if time.time() > deadline - 30.0:
            break
        res = _als_decompose(T, r, max_iter=200, time_limit=deadline - 30.0)
        if res is not None:
            U, V, W = res
            if _error(U, V, W, T) < 1e-9:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # found the smallest r in this scan that works

    # Try column-merging reduction on the best solution found.
    if time.time() < deadline - 10.0:
        merged = _try_column_merge(best_U, best_V, best_W, T, time_limit=deadline - 10.0)
        if merged is not None:
            best_U, best_V, best_W = merged
            best_rank = best_U.shape[1]

    # Snap near-integer / half-integer values to exact rationals for cleanliness.
    def snap(M):
        M = np.array(M, dtype=np.float64)
        out = np.round(M * 2.0) / 2.0
        # If values are very close to integers, prefer integers.
        close_int = np.abs(out - np.round(out)) < 1e-6
        out = np.where(close_int, np.round(out), out)
        return out

    Uf, Vf, Wf = snap(best_U), snap(best_V), snap(best_W)
    # Verify the snapped solution is still exact.
    if _error(Uf, Vf, Wf, T) < 1e-9:
        best_U, best_V, best_W = Uf, Vf, Wf

    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
