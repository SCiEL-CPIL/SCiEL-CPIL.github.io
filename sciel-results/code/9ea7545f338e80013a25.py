<<<<<<< SEARCH
    def optimize_shape(pts):
        best_D = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Coarse grid search for beta
        for i in range(13): # 0 to pi/6 step pi/72
            beta = i * (math.pi / 6) / 12
            # Minimize over center
            # Initial guess: centroid
            cx0, cy0 = np.mean(pts, axis=0)
            res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_D:
                best_D = res.fun
                best_beta = beta
                best_center = (res.x[0], res.x[1])
        
        # Refine beta
        def obj_beta(beta):
            # Re-optimize center for this beta
            cx0, cy0 = best_center
            res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
            return res.fun
        
        res_beta = minimize_scalar(obj_beta, bounds=(max(0, best_beta - 0.1), min(math.pi/6, best_beta + 0.1)), method='bounded')
        if res_beta.fun < best_D:
            best_D = res_beta.fun
            best_beta = res_beta.x
            # Update center
            cx0, cy0 = best_center
            res_c = minimize(objective_center, [cx0, cy0], args=(pts, best_beta), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            best_center = (res_c.x[0], res_c.x[1])
            best_D = res_c.fun

        S = (best_D + APOTHEM) / APOTHEM
        return S, best_beta, best_center
=======
    def optimize_shape(pts):
        best_D = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Precompute a good initial center using minimum enclosing circle
        mec_center = min_enclosing_circle(pts)
        
        # Coarse grid search for beta with more points
        for i in range(25): # 0 to pi/6 step pi/144
            beta = i * (math.pi / 6) / 24
            # Minimize over center - try multiple initial guesses
            best_res_for_beta = None
            for cx0, cy0 in [(mec_center[0], mec_center[1]), 
                             (np.mean(pts, axis=0)[0], np.mean(pts, axis=0)[1]),
                             (mec_center[0] + 0.1, mec_center[1]),
                             (mec_center[0], mec_center[1] + 0.1)]:
                res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
                if best_res_for_beta is None or res.fun < best_res_for_beta.fun:
                    best_res_for_beta = res
            
            if best_res_for_beta.fun < best_D:
                best_D = best_res_for_beta.fun
                best_beta = beta
                best_center = (best_res_for_beta.x[0], best_res_for_beta.x[1])
        
        # Refine beta
        def obj_beta(beta):
            # Re-optimize center for this beta
            cx0, cy0 = best_center
            res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            return res.fun
        
        res_beta = minimize_scalar(obj_beta, bounds=(max(0, best_beta - 0.15), min(math.pi/6, best_beta + 0.15)), method='bounded')
        if res_beta.fun < best_D:
            best_D = res_beta.fun
            best_beta = res_beta.x
            # Update center with high precision
            cx0, cy0 = best_center
            res_c = minimize(objective_center, [cx0, cy0], args=(pts, best_beta), method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14})
            best_center = (res_c.x[0], res_c.x[1])
            best_D = res_c.fun
        
        # Final polish pass with even tighter tolerances
        res_final = minimize(objective_center, list(best_center), args=(pts, best_beta), method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-16, 'fatol': 1e-16})
        if res_final.fun < best_D:
            best_D = res_final.fun
            best_center = (res_final.x[0], res_final.x[1])

        S = (best_D + APOTHEM) / APOTHEM
        return S, best_beta, best_center
>>>>>>> REPLACE
