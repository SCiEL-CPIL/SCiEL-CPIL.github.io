<<<<<<< SEARCH
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 8 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
=======
    # Phase 3: Final high-precision polish with tighter tolerances
    if time.time() < deadline - 5 and best_params is not None:
        def obj_polish_tight(p):
            il = []
            for i in range(N):
                il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
            ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
            side = compute_required_side_fast(il, (ocx, ocy), oa)
            pen = overlap_penalty(il)
            return side + 10000.0 * pen
        
        try:
            res = minimize(obj_polish_tight, best_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 8000})
            polish_inner = []
            for i in range(N):
                polish_inner.append([float(res.x[i * 3]), float(res.x[i * 3 + 1]), float(res.x[i * 3 + 2])])
            polish_oc = (float(res.x[N * 3]), float(res.x[N * 3 + 1]))
            polish_oa = float(res.x[N * 3 + 2])
            
            if check_disjoint(polish_inner, eps=1e-6):
                polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
                if polish_side < best_side:
                    best_inner = polish_inner
                    best_outer_center = polish_oc
                    best_outer_angle = polish_oa
                    best_side = polish_side
                    best_params = res.x.copy()
        except Exception:
            # Fallback to original polish
            polished = final_polish(best_params, deadline)
            polish_inner = []
            for i in range(N):
                polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
            polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
            polish_oa = float(polished[N * 3 + 2])
            
            if check_disjoint(polish_inner, eps=1e-5):
                polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
                if polish_side < best_side:
                    best_inner = polish_inner
                    best_outer_center = polish_oc
                    best_outer_angle = polish_oa
                    best_side = polish_side
>>>>>>> REPLACE
