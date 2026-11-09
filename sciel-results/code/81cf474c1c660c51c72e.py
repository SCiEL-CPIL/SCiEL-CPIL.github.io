<<<<<<< SEARCH
    # ---- Parse distribution ----
    p = None
    if isinstance(distribution_spec, dict):
        for k in ("prob", "probs", "distribution", "pdf", "weights", "mass", "P", "p"):
            if k in distribution_spec:
                p = np.asarray(distribution_spec[k], dtype=float)
                break
        if p is None:
            # maybe a 2D array under a key
            for k, v in distribution_spec.items():
                try:
                    arr = np.asarray(v, dtype=float)
                    if arr.ndim == 2 and arr.shape == (4, 4):
                        p = arr
                        break
                except Exception:
                    pass
    if p is None:
        p = np.asarray(distribution_spec, dtype=float)
    if p.ndim != 2 or p.shape != (4, 4):
        # try flatten 16
        pf = np.asarray(distribution_spec, dtype=float).ravel()
        if pf.size == 16:
            p = pf.reshape(4, 4)
        else:
            p = np.full((4, 4), 1.0 / 16.0)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s > 0:
        p = p / s
=======
    # ---- Parse distribution ----
    p = None
    try:
        if isinstance(distribution_spec, dict):
            for k in ("prob", "probs", "distribution", "pdf", "weights", "mass", "P", "p"):
                if k in distribution_spec:
                    p = np.asarray(distribution_spec[k], dtype=float)
                    break
            if p is None:
                for k, v in distribution_spec.items():
                    try:
                        arr = np.asarray(v, dtype=float)
                        if arr.ndim == 2 and arr.shape == (4, 4):
                            p = arr
                            break
                    except Exception:
                        pass
        else:
            p = np.asarray(distribution_spec, dtype=float)
    except Exception:
        p = None

    if p is None or p.size != 16:
        p = np.full((4, 4), 1.0 / 16.0)
    else:
        p = p.reshape(4, 4)
        
    # Ensure valid probability distribution
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s > 0:
        p = p / s
    else:
        p = np.full((4, 4), 1.0 / 16.0)
>>>>>>> REPLACE

This change ensures that `p` is always a valid 4x4 array of non-negative floats summing to 1. It handles cases where `distribution_spec` is a dict, a list, a numpy array, or even something unexpected by falling back to a uniform distribution. This prevents `TypeError` or `ValueError` in subsequent steps.

Additionally, the `_lp_optimum` function uses `linprog`. If `p` is all zeros (which we prevented) or very small, it might have issues, but we normalized it.

One more potential issue: `support` parsing. If `support` is a dict or something else, the `for item in support` loop might fail. Let's make that robust too.

