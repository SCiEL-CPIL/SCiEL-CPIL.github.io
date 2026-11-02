    def optimize_shape(pts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        def exact_S_for_beta(beta):
            S, center, D = calc_S_for_beta(pts, beta)
            return S
        
        # Grid search
        n_grid = 100
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S = exact_S_for_beta(beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                _, best_center, _ = calc_S_for_beta(pts, beta)
        
        if best_center is None:
            return float('inf'), 0.0, None
            
        # Refine with minimize_scalar
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(exact_S_for_beta, 
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
            
        return best_S, best_beta, best_center
   