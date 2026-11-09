    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def nm_optimize(x0, max_calls):
            nonlocal calls_made, best_x, best_f
            n = dim
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                if xi[i] == 0:
                    xi[i] = 0.00025
                else:
                    xi[i] *= 1 + 0.00025
                simplex.append(xi)
            
            fvals = []
            for i in range(n + 1):
                if calls_made >= budget: return
                fvals.append(func(simplex[i]))
                calls_made += 1
            
            # Sort
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    if fvals[i] > fvals[j]:
                        fvals[i], fvals[j] = fvals[j], fvals[i]
                        simplex[i], simplex[j] = simplex[j], simplex[i]
            
            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls_made < budget and (time.time() < deadline):
                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-12:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls_made >= budget: break
                fr = func(xr)
                calls_made += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls_made >= budget: break
                    fe = func(xe)
                    calls_made += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls_made >= budget: break
                    fc = func(xc)
                    calls_made += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls_made >= budget: break
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                        
                        # Re-sort
                        for i in range(n + 1):
                            for j in range(i + 1, n + 1):
                                if fvals[i] > fvals[j]:
                                    fvals[i], fvals[j] = fvals[j], fvals[i]
                                    simplex[i], simplex[j] = simplex[j], simplex[i]
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

        # Phase 1: Random Search to find good start
        n_init = 10
        for _ in range(n_init):
            if calls_made >= budget: break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead
        if best_x is not None:
            nm_optimize(best_x, budget - calls_made)
        
        # Phase 3: If budget remains, try another restart?
        if calls_made < budget and (time.time() < deadline) and best_x is not None:
            # Try one more restart from a different random point
            x_start = rng.uniform(-5, 5, dim)
            nm_optimize(x_start, budget - calls_made)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
    