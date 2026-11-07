<<<<<<< SEARCH
        # --- Strategy: Halton initialization + Hypervolume-guided DE + Local Polish ---
        
        def halton_sequence(n, dim):
            """Generate Halton sequence points in [0,1]^dim"""
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, dim + 2):
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    point.append(r)
                points.append(point)
            return np.array(points)
        
        # 1. Halton initialization for better coverage (40% of budget)
        n_init = int(budget * 0.4)  # 80 points
        halton_pts = halton_sequence(n_init, dim)
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * halton_pts[i]
            evaluate(x)
        
        # 2. Additional random sampling to fill remaining init budget
        n_rand_init = int(budget * 0.1)  # 20 more points
        for _ in range(n_rand_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)
=======
        # --- Strategy: Halton initialization + Grid-based decomposition + Local Polish ---
        
        def halton_sequence(n, dim):
            """Generate Halton sequence points in [0,1]^dim"""
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, dim + 2):
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    point.append(r)
                points.append(point)
            return np.array(points)
        
        def nelder_mead(problem, x0, bounds, max_iter=50, tol=1e-8):
            """Nelder-Mead simplex method for multi-objective optimization"""
            dim = len(x0)
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            
            # Initialize simplex
            simplex = [np.clip(x0.copy(), lo, hi)]
            for i in range(dim):
                dx = x0.copy()
                dx[i] += 0.05 * (hi[i] - lo[i])
                simplex.append(np.clip(dx, lo, hi))
            
            # Evaluate initial simplex
            fvals = [np.array(problem(s), dtype=float) for s in simplex]
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Sort by hypervolume contribution (use weighted sum as proxy)
                # Use lexicographic order: minimize first obj, then second
                order = np.argsort([fvals[i][0] + fvals[i][1] for i in range(len(simplex))])
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if np.max([np.linalg.norm(fvals[0] - fvals[i]) for i in range(1, len(fvals))]) < tol:
                    break
                
                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = np.array(problem(x_r), dtype=float)
                
                if f_r[0] < fvals[0][0] and f_r[1] < fvals[0][1]:
                    # Expansion
                    x_e = centroid + 2.0 * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = np.array(problem(x_e), dtype=float)
                    if f_e[0] < f_r[0] and f_e[1] < f_r[1]:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif not (f_r[0] >= fvals[-1][0] and f_r[1] >= fvals[-1][1]):
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = np.array(problem(x_c), dtype=float)
                    if f_c[0] < fvals[-1][0] and f_c[1] < fvals[-1][1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = np.array(problem(simplex[i]), dtype=float)
            
            return simplex[0], fvals[0]
        
        # 1. Halton initialization for better coverage (30% of budget)
        n_init = int(budget * 0.3)  # 60 points
        halton_pts = halton_sequence(n_init, dim)
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * halton_pts[i]
            evaluate(x)
        
        # 2. Additional random sampling to fill remaining init budget
        n_rand_init = int(budget * 0.1)  # 20 more points
        for _ in range(n_rand_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)
>>>>>>> REPLACE

