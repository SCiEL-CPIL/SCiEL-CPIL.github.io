def _run():
    t0 = time.time()
    deadline = t0 + 580.0  # 20s safety margin
    best_c = np.ones(2 * _K + 1, dtype=np.float64)
    best_R = _ratios(best_c)[0]

    rng = np.random.default_rng(20240517)
    dim = 2 * _K + 1

    # Generate diverse initializations
    inits = []
    inits.append(np.ones(dim))
    for m in range(-_K, _K + 1):
        v = np.zeros(dim); v[m + _K] = 1.0; inits.append(v)
    for _ in range(50):
        inits.append(rng.normal(size=dim))
    
    # Score and sort initializations
    scored = []
    for v in inits:
        if not np.all(np.isfinite(v)): continue
        R = _ratios(v)[0]
        scored.append((R, v))
    scored.sort(key=lambda x: x[0])

    # Optimization loop
    for i, (R0, v0) in enumerate(scored[:20]): # Take top 20
        if time.time() > deadline - 30.0:
            break
        
        # Try L-BFGS-B first
        try:
            res = minimize(_objective, v0, jac=_gradient, method="L-BFGS-B",
                           options=dict(maxiter=200, ftol=1e-12, gtol=1e-8, maxls=20))
            if res.fun < _objective(v0):
                Rn = _ratios(res.x)[0]
                if Rn < best_R:
                    best_R = Rn
                    best_c = res.x.copy()
        except Exception:
            pass

        # If L-BFGS-B failed or didn't improve, try Nelder-Mead (robust)
        if time.time() < deadline - 30.0:
            try:
                res_nm = minimize(_objective, v0, method="Nelder-Mead",
                                  options=dict(maxiter=500, xatol=1e-8, fatol=1e-8))
                if res_nm.fun < _objective(v0):
                    Rn = _ratios(res_nm.x)[0]
                    if Rn < best_R:
                        best_R = Rn
                        best_c = res_nm.x.copy()
            except Exception:
                pass

    # Final polish with L-BFGS-B if time permits
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_objective, best_c, jac=_gradient, method="L-BFGS-B",
                           options=dict(maxiter=300, ftol=1e-14, gtol=1e-10, maxls=20))
            if res.fun < _objective(best_c):
                Rn = _ratios(res.x)[0]
                if Rn < best_R:
                    best_R = Rn
                    best_c = res.x.copy()
        except Exception:
            pass

    # Validity guard
    c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(c)):
        c = np.ones(dim)
    
    # Scale to valid range
    max_abs = np.max(np.abs(c))
    if max_abs == 0:
        c = np.ones(dim)
    else:
        c = c / max_abs
        # Ensure max|c| is in [1e-3, 1e5]
        if np.max(np.abs(c)) < 1e-3:
            c *= 1e-2
        if np.max(np.abs(c)) > 1e5:
            c /= np.max(np.abs(c))
            c *= 1e4

    return c
