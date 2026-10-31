<<<<<<< SEARCH
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.clip(np.minimum(r, lo), 1e-6, None)
    out = np.column_stack([cx, cy, r])

    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        for s0, W, H, g0 in pool[:5]:
            cx2 = np.clip(g0[:, 0], 0.0, W)
            cy2 = np.clip(g0[:, 1], 0.0, H)
            r2 = g0[:, 2].copy()
            lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
            r2 = np.clip(np.minimum(r2, lo2), 1e-6, None)
            out2 = np.column_stack([cx2, cy2, r2])
            if _evaluate(out2, W, H) is not None:
                return out2
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return out
=======
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.clip(np.minimum(r, lo), 1e-6, None)
    out = np.column_stack([cx, cy, r])

    # Strict final validation: check all pairs for overlap
    def _is_valid(arr, W, H):
        cx_v = arr[:, 0]
        cy_v = arr[:, 1]
        r_v = arr[:, 2]
        n_v = len(cx_v)
        # Check boundary
        for i in range(n_v):
            if cx_v[i] < r_v[i] - 1e-9 or cx_v[i] > W - r_v[i] + 1e-9:
                return False
            if cy_v[i] < r_v[i] - 1e-9 or cy_v[i] > H - r_v[i] + 1e-9:
                return False
        # Check all pairs
        for i in range(n_v):
            for j in range(i + 1, n_v):
                d = np.hypot(cx_v[i] - cx_v[j], cy_v[i] - cy_v[j])
                if d < r_v[i] + r_v[j] - 1e-9:
                    return False
        return True

    if _is_valid(out, best_W, best_H):
        return out

    # Try pool entries
    for s0, W, H, g0 in pool[:10]:
        cx2 = np.clip(g0[:, 0], 0.0, W)
        cy2 = np.clip(g0[:, 1], 0.0, H)
        r2 = g0[:, 2].copy()
        lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
        r2 = np.clip(np.minimum(r2, lo2), 1e-6, None)
        out2 = np.column_stack([cx2, cy2, r2])
        if _is_valid(out2, W, H):
            return out2

    # Guaranteed valid: 5x5 grid in 1x1 square
    W_f, H_f = 1.0, 1.0
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cx_c = (i + 0.5) / side
            cy_c = (j + 0.5) / side
            r_c = 0.5 / side - 1e-4
            cells.append([cx_c, cy_c, r_c])
    return np.array(cells[:N])
>>>>>>> REPLACE
