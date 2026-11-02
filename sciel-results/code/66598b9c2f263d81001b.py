<<<<<<< SEARCH
def construct():
    t0 = time.time()
    deadline = t0 + 190  # safety margin before 210s kill
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective_exact(best)
    
    # Phase 1: Local refinement from seed with smooth objective
    try:
        res = minimize(objective, seed, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol':1e-10, 'fatol':1e-12})
        if time.time() < deadline and res.fun < best_val:
            best = res.x.copy()
            best_val = objective_exact(best)
    except Exception:
        pass
    
    # Phase 2: Multi-start with diverse initial points
    if time.time() < deadline - 80:
        rng = np.random.default_rng(42)
        # Generate diverse starting points in multiple scales
        starts = []
        # Small perturbations around seed
        for _ in range(10):
            starts.append(seed + rng.normal(0, 0.1, size=3))
        # Medium range
        for _ in range(10):
            starts.append(rng.uniform(-1, 1, size=3))
        # Larger range
        for _ in range(10):
            starts.append(rng.uniform(-3, 3, size=3))
        # Structured points
        for scale in [0.01, 0.1, 1.0]:
            for i in range(3):
                pt = np.zeros(3)
                pt[i] = scale
                starts.append(pt)
                pt[i] = -scale
                starts.append(pt)
        
        for start in starts:
            if time.time() > deadline - 40:
                break
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol':1e-9, 'fatol':1e-11})
                if time.time() > deadline - 40:
                    break
                val = objective_exact(res.x)
                if val < best_val:
                    best = res.x.copy()
                    best_val = val
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish with exact objective
    if time.time() < deadline - 15:
        try:
            res = minimize(objective_exact, best, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol':1e-11, 'fatol':1e-13})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 4: One more very tight polish if time remains
    if time.time() < deadline - 5:
        try:
            res = minimize(objective_exact, best, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol':1e-12, 'fatol':1e-14})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return np.array(best)
=======
def construct():
    t0 = time.time()
    deadline = t0 + 195  # safety margin before 210s kill
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective_exact(best)
    
    # Phase 1: Local refinement from seed with smooth objective
    try:
        res = minimize(objective, seed, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol':1e-10, 'fatol':1e-12})
        if time.time() < deadline and res.fun < best_val:
            best = res.x.copy()
            best_val = objective_exact(best)
    except Exception:
        pass
    
    # Phase 2: CMA-ES global search with multiple restarts
    if time.time() < deadline - 100:
        rng = np.random.default_rng(42)
        
        # Define CMA-ES options for each restart
        cma_opts = {
            'maxfevals': 300,
            'verbose': -9,
            'bounds': [-10, 10],
            'seed': None,
        }
        
        # Multiple CMA-ES restarts from different centers
        centers = [
            seed.copy(),
            seed + np.array([0.1, -0.05, 0.02]),
            seed + np.array([-0.1, 0.05, -0.02]),
            np.array([0.0, 0.0, 0.0]),
            np.array([-0.5, 0.1, 0.05]),
            np.array([0.5, -0.1, -0.05]),
            np.array([-1.0, 0.2, 0.1]),
            np.array([1.0, -0.2, -0.1]),
        ]
        
        for i, center in enumerate(centers):
            if time.time() > deadline - 50:
                break
            try:
                opts = cma_opts.copy()
                opts['seed'] = 42 + i
                es = cma.CMAEvolutionStrategy(center, 0.3, opts)
                while not es.stop() and time.time() < deadline - 50:
                    solutions = es.ask()
                    fitnesses = [objective(s) for s in solutions]
                    es.tell(solutions, fitnesses)
                xopt = es.result.xbest
                val = objective_exact(xopt)
                if val < best_val:
                    best = xopt.copy()
                    best_val = val
            except Exception:
                pass
    
    # Phase 3: Multi-start Nelder-Mead with diverse initial points
    if time.time() < deadline - 40:
        rng = np.random.default_rng(123)
        # Generate diverse starting points in multiple scales
        starts = []
        # Small perturbations around current best
        for _ in range(8):
            starts.append(best + rng.normal(0, 0.05, size=3))
        # Medium range
        for _ in range(8):
            starts.append(rng.uniform(-1, 1, size=3))
        # Larger range
        for _ in range(8):
            starts.append(rng.uniform(-3, 3, size=3))
        # Structured points
        for scale in [0.01, 0.1, 0.5]:
            for i in range(3):
                pt = np.zeros(3)
                pt[i] = scale
                starts.append(pt)
                pt[i] = -scale
                starts.append(pt)
        
        for start in starts:
            if time.time() > deadline - 25:
                break
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol':1e-9, 'fatol':1e-11})
                if time.time() > deadline - 25:
                    break
                val = objective_exact(res.x)
                if val < best_val:
                    best = res.x.copy()
                    best_val = val
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish with exact objective
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_exact, best, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol':1e-11, 'fatol':1e-13})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 5: One more very tight polish if time remains
    if time.time() < deadline - 3:
        try:
            res = minimize(objective_exact, best, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol':1e-12, 'fatol':1e-14})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return np.array(best)
>>>>>>> REPLACE
