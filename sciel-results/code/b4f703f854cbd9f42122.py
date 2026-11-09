<<<<<<< SEARCH
    # Multiple Nelder-Mead refinement passes from different starting points
    if (time.time() - start) < deadline - 60.0:
        # Pass 1: From DE result
        try:
            x0 = best_offsets.copy()
            r2 = minimize(
                lambda x: _obj(x, smooth=True),
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": 30000,
                    "xatol": 1e-12,
                    "fatol": 1e-14,
                    "adaptive": True
                }
            )
            if r2 is not None and r2.x is not None:
                pts = _build_polygon(r2.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
        except Exception:
            pass
        
        # Pass 2: From best_offsets with exact objective
        if (time.time() - start) < deadline - 30.0:
            try:
                x0 = best_offsets.copy()
                r3 = minimize(
                    lambda x: _obj(x, smooth=False),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 50000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r3 is not None and r3.x is not None:
                    pts = _build_polygon(r3.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
            except Exception:
                pass
        
        # Pass 3: Try a few random perturbations of best_offsets
        if (time.time() - start) < deadline - 15.0:
            rng = np.random.RandomState(42)
            for trial in range(5):
                if (time.time() - start) >= deadline - 10.0:
                    break
                try:
                    x0 = best_offsets + rng.normal(0, 0.01, size=4)
                    x0 = np.sort(x0)
                    if x0[0] < 0 or x0[-1] >= 2 * np.pi / 5:
                        continue
                    if np.any(np.diff(x0) < 1e-10):
                        continue
                    wrap_gap = (x0[0] + 2 * np.pi / 5) - x0[-1]
                    if wrap_gap < 1e-10:
                        continue
                    r4 = minimize(
                        lambda x: _obj(x, smooth=False),
                        x0,
                        method="Nelder-Mead",
                        options={
                            "maxiter": 20000,
                            "xatol": 1e-14,
                            "fatol": 1e-16,
                            "adaptive": True
                        }
                    )
                    if r4 is not None and r4.x is not None:
                        pts = _build_polygon(r4.x)
                        if pts is not None:
                            sc = _polygon_score(pts)
                            if sc > best_score:
                                best_pts, best_score = pts, sc
                except Exception:
                    pass
=======
    # Multiple Nelder-Mead refinement passes from different starting points
    if (time.time() - start) < deadline - 60.0:
        # Pass 1: From DE result
        try:
            x0 = best_params.copy()
            r2 = minimize(
                lambda x: _obj(x, smooth=True),
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": 30000,
                    "xatol": 1e-12,
                    "fatol": 1e-14,
                    "adaptive": True
                }
            )
            if r2 is not None and r2.x is not None:
                pts = _build_polygon(r2.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
                        best_params = r2.x.copy()
        except Exception:
            pass
        
        # Pass 2: From best_params with exact objective
        if (time.time() - start) < deadline - 30.0:
            try:
                x0 = best_params.copy()
                r3 = minimize(
                    lambda x: _obj(x, smooth=False),
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 50000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r3 is not None and r3.x is not None:
                    pts = _build_polygon(r3.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
                            best_params = r3.x.copy()
            except Exception:
                pass
        
        # Pass 3: Try a few random perturbations of best_params
        if (time.time() - start) < deadline - 15.0:
            rng = np.random.RandomState(42)
            for trial in range(5):
                if (time.time() - start) >= deadline - 10.0:
                    break
                try:
                    x0 = best_params.copy()
                    x0[:4] = x0[:4] + rng.normal(0, 0.01, size=4)
                    x0[4:] = x0[4:] + rng.normal(0, 0.05, size=4)
                    x0[:4] = np.sort(x0[:4])
                    if x0[0] < 0 or x0[3] >= 2 * np.pi / 5:
                        continue
                    if np.any(np.diff(x0[:4]) < 1e-10):
                        continue
                    wrap_gap = (x0[0] + 2 * np.pi / 5) - x0[3]
                    if wrap_gap < 1e-10:
                        continue
                    if np.any(x0[4:] < 0.3) or np.any(x0[4:] > 3.0):
                        continue
                    r4 = minimize(
                        lambda x: _obj(x, smooth=False),
                        x0,
                        method="Nelder-Mead",
                        options={
                            "maxiter": 20000,
                            "xatol": 1e-14,
                            "fatol": 1e-16,
                            "adaptive": True
                        }
                    )
                    if r4 is not None and r4.x is not None:
                        pts = _build_polygon(r4.x)
                        if pts is not None:
                            sc = _polygon_score(pts)
                            if sc > best_score:
                                best_pts, best_score = pts, sc
                                best_params = r4.x.copy()
                except Exception:
                    pass
>>>>>>> REPLACE
