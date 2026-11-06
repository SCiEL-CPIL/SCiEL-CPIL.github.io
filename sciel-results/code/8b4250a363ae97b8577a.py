<<<<<<< SEARCH
def _optimize():
    """Run global and local optimization."""
    # Physics-based initial guess
    # c0 ~ 0.55 (base index)
    # c1 ~ 0.0 (linear matter correction)
    # c2 ~ 0.05 (quadratic curvature)
    # c3 ~ -0.05 (linear u correction)
    # c4 ~ 0.0 (explicit wa*a)
    # c5 ~ 0.03 (u*q cross term)
    seed = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_c = seed.copy()
    best_score = _score(seed)
    
    bounds = [(-10.0, 10.0)] * 6
    
    # 1. Differential Evolution for global exploration
    if time.time() < _DEADLINE - 30.0:
        try:
            de_res = differential_evolution(
                _score, bounds, 
                maxiter=100, popsize=20, 
                tol=1e-6, 
                seed=42, 
                polish=False
            )
            if de_res.fun < best_score:
                best_c = de_res.x
                best_score = de_res.fun
        except Exception:
            pass

    # 2. Multi-start Nelder-Mead for local refinement
    if time.time() < _DEADLINE - 15.0:
        # Diverse starting points based on physics intuition
        starts = [
            best_c,
            np.array([0.5, 0.1, -0.05, 0.1, 0.0, 0.0]),
            np.array([0.6, -0.1, 0.1, 0.0, 0.0, 0.05]),
            np.array([0.55, 0.05, 0.0, 0.05, -0.05, 0.0]),
            np.array([0.5, 0.0, 0.1, -0.1, 0.05, -0.05]),
            np.array([0.55, -0.05, 0.05, 0.0, 0.0, 0.05]),
            np.array([0.6, 0.0, 0.0, 0.1, 0.0, 0.0]),
            np.array([0.5, 0.1, 0.0, 0.0, 0.05, 0.05]),
            np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.6, 0.05, -0.05, 0.05, 0.05, -0.03]),
            np.array([0.5, -0.05, 0.08, -0.08, 0.02, 0.04]),
            np.array([0.58, 0.02, 0.03, -0.02, -0.03, 0.02]),
        ]
        
        for start in starts:
            if time.time() > _DEADLINE - 15.0:
                break
            try:
                nm_res = minimize(
                    _score, start, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-11, 'adaptive': True}
                )
                if nm_res.fun < best_score:
                    best_c = nm_res.x
                    best_score = nm_res.fun
            except Exception:
                pass

    # 3. Random restarts around best solution
    if time.time() < _DEADLINE - 10.0:
        rng = np.random.default_rng(12345)
        for _ in range(150):
            if time.time() > _DEADLINE - 10.0:
                break
            c = rng.normal(loc=best_c, scale=[0.1, 0.3, 0.1, 0.3, 0.3, 0.2])
            c = np.clip(c, -10, 10)
            s = _score(c)
            if s < best_score:
                best_score = s
                best_c = c.copy()
                # Quick local polish
                if time.time() < _DEADLINE - 10.0:
                    try:
                        res = minimize(
                            _score, best_c, 
                            method='Nelder-Mead',
                            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-11, 'adaptive': True}
                        )
                        if res.fun < best_score:
                            best_score = res.fun
                            best_c = np.clip(res.x, -10, 10)
                    except Exception:
                        pass

    # 4. Final high-precision polish with L-BFGS-B
    if time.time() < _DEADLINE - 2.0:
        try:
            lbfgs_res = minimize(
                _score, best_c, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if lbfgs_res.fun < best_score:
                best_c = lbfgs_res.x
                best_score = lbfgs_res.fun
        except Exception:
            pass
        
        # Additional Nelder-Mead polish
        try:
            nm_res2 = minimize(
                _score, best_c, 
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_res2.fun < best_score:
                best_c = nm_res2.x
                best_score = nm_res2.fun
        except Exception:
            pass

    # Ensure bounds are respected
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return best_c
=======
def _optimize():
    """Run global and local optimization."""
    # Physics-based initial guess
    # c0 ~ 0.55 (base index)
    # c1 ~ 0.0 (linear matter correction)
    # c2 ~ 0.05 (quadratic curvature)
    # c3 ~ -0.05 (linear u correction)
    # c4 ~ 0.0 (explicit wa*a)
    # c5 ~ 0.03 (u*q cross term)
    seed = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_c = seed.copy()
    best_score = _score(seed)
    
    bounds = [(-10.0, 10.0)] * 6
    
    # 1. Differential Evolution for global exploration (use smooth surrogate)
    if time.time() < _DEADLINE - 35.0:
        try:
            de_res = differential_evolution(
                _score_smooth, bounds, 
                maxiter=150, popsize=25, 
                tol=1e-8, 
                seed=42, 
                polish=False
            )
            if de_res.fun < best_score:
                best_c = de_res.x
                best_score = _score(best_c)
        except Exception:
            pass

    # 2. Smooth-surrogate local refinement with multiple starts
    if time.time() < _DEADLINE - 25.0:
        starts = [
            best_c,
            np.array([0.5, 0.1, -0.05, 0.1, 0.0, 0.0]),
            np.array([0.6, -0.1, 0.1, 0.0, 0.0, 0.05]),
            np.array([0.55, 0.05, 0.0, 0.05, -0.05, 0.0]),
            np.array([0.5, 0.0, 0.1, -0.1, 0.05, -0.05]),
            np.array([0.55, -0.05, 0.05, 0.0, 0.0, 0.05]),
            np.array([0.6, 0.0, 0.0, 0.1, 0.0, 0.0]),
            np.array([0.5, 0.1, 0.0, 0.0, 0.05, 0.05]),
            np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.6, 0.05, -0.05, 0.05, 0.05, -0.03]),
            np.array([0.5, -0.05, 0.08, -0.08, 0.02, 0.04]),
            np.array([0.58, 0.02, 0.03, -0.02, -0.03, 0.02]),
            np.array([0.52, 0.08, 0.02, -0.03, 0.04, 0.01]),
            np.array([0.57, -0.02, 0.06, -0.04, -0.02, 0.04]),
        ]
        
        for start in starts:
            if time.time() > _DEADLINE - 25.0:
                break
            try:
                nm_res = minimize(
                    _score_smooth, start, 
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-13, 'adaptive': True}
                )
                if nm_res.fun < best_score:
                    best_c = nm_res.x
                    best_score = _score(best_c)
            except Exception:
                pass

    # 3. Random restarts with smooth surrogate, then exact polish
    if time.time() < _DEADLINE - 15.0:
        rng = np.random.default_rng(12345)
        for _ in range(200):
            if time.time() > _DEADLINE - 15.0:
                break
            c = rng.normal(loc=best_c, scale=[0.15, 0.4, 0.15, 0.4, 0.4, 0.25])
            c = np.clip(c, -10, 10)
            s_smooth = _score_smooth(c)
            if s_smooth < best_score:
                # Polish with exact score
                try:
                    res = minimize(
                        _score, c, 
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-13, 'adaptive': True}
                    )
                    if res.fun < best_score:
                        best_score = res.fun
                        best_c = np.clip(res.x, -10, 10)
                except Exception:
                    s_exact = _score(c)
                    if s_exact < best_score:
                        best_score = s_exact
                        best_c = c.copy()

    # 4. Final high-precision polish: L-BFGS-B then Nelder-Mead
    if time.time() < _DEADLINE - 3.0:
        try:
            lbfgs_res = minimize(
                _score, best_c, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if lbfgs_res.fun < best_score:
                best_c = lbfgs_res.x
                best_score = lbfgs_res.fun
        except Exception:
            pass
        
        # Additional Nelder-Mead polish with very tight tolerances
        try:
            nm_res2 = minimize(
                _score, best_c, 
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-14, 'adaptive': True}
            )
            if nm_res2.fun < best_score:
                best_c = nm_res2.x
                best_score = nm_res2.fun
        except Exception:
            pass

    # 5. Verify gamma constraints and final clip
    best_c = np.clip(best_c, -10.0, 10.0)
    # Quick check: if gamma is violated, nudge toward valid region
    refs = _build_refs()
    for m in refs:
        Om0, w0, wa = m['Om0'], m['w0'], m['wa']
        a = m['a']
        g = _gamma_vec(a, Om0, w0, wa, best_c)
        if np.any(g < 0.1) or np.any(g > 1.2):
            # Simple projection: scale down coefficients that push gamma out of range
            # This is a heuristic; the optimizer should have handled this, but as safety
            best_c = best_c * 0.95
            break
    
    return best_c
>>>>>>> REPLACE

