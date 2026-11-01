<<<<<<< SEARCH
    if time.time() < start_time + 148:
        cd_final = _coordinate_descent_polish(np.concatenate([x, y, r, [w]]), start_time + 148)
        if cd_final is not None:
            x2 = cd_final[0::3][:N]
            y2 = cd_final[1::3][:N]
            r2 = cd_final[2::3][:N]
            w2 = cd_final[-1]
            h2 = 2.0 - w2
            if _is_valid(x2, y2, r2, w2, h2) and np.sum(r2) > np.sum(r):
                x, y, r, w, h = x2, y2, r2, w2, h2

    return np.column_stack((x, y, r))
=======
    if time.time() < start_time + 148:
        cd_final = _coordinate_descent_polish(np.concatenate([x, y, r, [w]]), start_time + 148)
        if cd_final is not None:
            x2 = cd_final[0::3][:N]
            y2 = cd_final[1::3][:N]
            r2 = cd_final[2::3][:N]
            w2 = cd_final[-1]
            h2 = 2.0 - w2
            if _is_valid(x2, y2, r2, w2, h2) and np.sum(r2) > np.sum(r):
                x, y, r, w, h = x2, y2, r2, w2, h2

    # Final high-precision polish: tight uniform growth + per-circle expansion
    if time.time() < start_time + 149:
        for _ in range(50):
            if time.time() > start_time + 149:
                break
            lo, hi = 1.0, 1.01
            for _ in range(20):
                mid = (lo + hi) / 2.0
                if _is_valid(x, y, r * mid, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > 1.0 + 1e-12:
                r = r * lo
            else:
                break
        # Per-circle final expansion
        for i in range(N):
            if time.time() > start_time + 149:
                break
            r_max = _max_radius_for(i, x, y, r, w, h)
            if r_max > r[i] + 1e-12:
                lo_r, hi_r = r[i], r_max
                for _ in range(25):
                    mid_r = (lo_r + hi_r) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid_r
                    if _is_valid(x, y, r_test, w, h):
                        lo_r = mid_r
                    else:
                        hi_r = mid_r
                r[i] = lo_r

    return np.column_stack((x, y, r))
>>>>>>> REPLACE
