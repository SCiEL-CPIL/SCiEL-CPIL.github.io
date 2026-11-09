import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            try:
                y = func(x)
            except Exception:
                return None
            if not np.isfinite(y):
                return None
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # CMA-ES implementation (single population)
        def cma_es_single(mean, sigma, max_calls, island_rng):
            nonlocal best_x, best_y, calls_made
            
            if max_calls <= 0:
                return
            
            n = d
            lam = int(4 + 3 * np.log(n))
            mu = lam // 2
            
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + c_sigma
            
            mean = np.array(mean, dtype=float).copy()
            sigma = float(sigma)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            def update_sqrt_C():
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                sqrt_inv_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return sqrt_C, sqrt_inv_C
            
            sqrt_C, sqrt_inv_C = update_sqrt_C()
            
            gen = 0
            while calls_made < budget and time_ok() and gen < 200:
                # Check if we've used up our allocated calls
                if calls_made >= max_calls:
                    break
                
                # Generate population
                pop = []
                for i in range(lam):
                    z = island_rng.standard_normal(n)
                    x = mean + sigma * (sqrt_C @ z)
                    pop.append(x)
                
                # Evaluate population
                fvals = []
                for x in pop:
                    y = call_func(x)
                    if y is None:
                        return
                    fvals.append(y)
                
                if len(fvals) < lam:
                    return
                
                # Sort by fitness
                idx = np.argsort(fvals)
                pop = [pop[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Compute weighted mean
                new_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                
                # Update evolution path
                pc = (1 - c_c) * pc + c_c * (sqrt_inv_C @ (new_mean - mean) / sigma)
                
                # Update covariance matrix
                delta_hs = 1.0 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(gen + 1)) < (1.4 + 2/(n+1)) * chi_n else 0.0
                ps = (1 - c_sigma) * ps + delta_hs * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (sqrt_inv_C @ (new_mean - mean) / sigma)
                
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - C)
                C += c_mu * sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu))
                
                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                mean = new_mean
                
                # Update sqrt(C) periodically
                if gen % 5 == 0:
                    sqrt_C, sqrt_inv_C = update_sqrt_C()
                
                # Convergence check
                if sigma < 1e-15:
                    break
                
                gen += 1
            
            return mean, sigma

        # Nelder-Mead simplex method
        def nelder_mead(x0, max_calls, nm_rng):
            nonlocal best_x, best_y, calls_made
            
            if max_calls <= 0 or calls_made >= budget:
                return
            
            n = d
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(xi)
            
            fvals = []
            for x in simplex:
                y = call_func(x)
                if y is None:
                    return
                fvals.append(y)
            
            nm_calls = n + 1
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            max_iter = max_calls - nm_calls
            if max_iter <= 0:
                return
            
            for _ in range(max_iter):
                if calls_made >= budget or not time_ok():
                    break
                
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = call_func(xr)
                if fr is None:
                    break
                nm_calls += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = call_func(xe)
                    if fe is None:
                        break
                    nm_calls += 1
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
                    if fr < fvals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                        fc = call_func(xc)
                        if fc is None:
                            break
                        nm_calls += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fi = call_func(simplex[i])
                                if fi is None:
                                    return
                                fvals[i] = fi
                                nm_calls += 1
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = call_func(xc)
                        if fc is None:
                            break
                        nm_calls += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fi = call_func(simplex[i])
                                if fi is None:
                                    return
                                fvals[i] = fi
                                nm_calls += 1
                
                if nm_calls >= max_calls:
                    break
            
            # Find best in simplex
            best_idx = np.argmin(fvals)
            if fvals[best_idx] < best_y:
                best_y = fvals[best_idx]
                best_x = simplex[best_idx].copy()

        # Phase 1: Initial sampling to find good starting points
        n_init = 30
        for _ in range(n_init):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)
        
        if best_x is None:
            return best_x, best_y
        
        # Phase 2: Island-based CMA-ES
        n_islands = 4
        remaining = budget - calls_made
        
        if remaining > 40 and time_ok():
            # Create island initial points
            island_starts = [best_x.copy()]
            for i in range(n_islands - 1):
                # Perturb best point to create diversity
                perturbation = rng.standard_normal(d) * 1.5
                island_starts.append(np.clip(best_x + perturbation, -5, 5))
            
            # Distribute budget among islands
            calls_per_island = remaining // n_islands
            migration_interval = max(5, calls_per_island // 4)
            
            # Run islands in a round-robin fashion
            island_states = []
            for i in range(n_islands):
                is_rng = np.random.default_rng(42 + i * 1000)
                island_states.append({
                    'mean': island_starts[i].copy(),
                    'sigma': 1.5,
                    'calls_used': 0,
                    'rng': is_rng,
                    'best_x': island_starts[i].copy(),
                    'best_y': np.inf,
                    'gen': 0
                })
            
            # Track global best for migration
            global_best_x = best_x.copy()
            global_best_y = best_y
            
            # Run islands round-robin
            total_calls_this_phase = 0
            max_rounds = remaining
            
            while total_calls_this_phase < remaining and time_ok():
                for i in range(n_islands):
                    if total_calls_this_phase >= remaining or not time_ok():
                        break
                    
                    state = island_states[i]
                    # Run one generation of CMA-ES on this island
                    n = d
                    lam = int(4 + 3 * np.log(n))
                    mu = lam // 2
                    
                    weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mueff = 1.0 / np.sum(weights**2)
                    
                    c_sigma = (mueff + 2) / (n + mueff + 5)
                    c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                    c_1 = 2 / ((n + 1.3)**2 + mueff)
                    c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
                    
                    chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                    d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + c_sigma
                    
                    mean = state['mean']
                    sigma = state['sigma']
                    
                    # Initialize C matrices if first gen
                    if state['gen'] == 0:
                        C = np.eye(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-20)
                        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                        sqrt_inv_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                        state['C'] = C
                        state['pc'] = pc
                        state['ps'] = ps
                        state['sqrt_C'] = sqrt_C
                        state['sqrt_inv_C'] = sqrt_inv_C
                    else:
                        C = state['C']
                        pc = state['pc']
                        ps = state['ps']
                        sqrt_C = state['sqrt_C']
                        sqrt_inv_C = state['sqrt_inv_C']
                    
                    # Generate population
                    pop = []
                    for j in range(lam):
                        z = state['rng'].standard_normal(n)
                        x = mean + sigma * (sqrt_C @ z)
                        pop.append(x)
                    
                    # Evaluate population
                    fvals = []
                    for x in pop:
                        y = call_func(x)
                        if y is None:
                            break
                        fvals.append(y)
                    
                    if len(fvals) < lam:
                        break
                    
                    total_calls_this_phase += lam
                    
                    # Sort by fitness
                    idx = np.argsort(fvals)
                    pop = [pop[i] for i in idx]
                    fvals = [fvals[i] for i in idx]
                    
                    # Update island best
                    if fvals[0] < state['best_y']:
                        state['best_y'] = fvals[0]
                        state['best_x'] = pop[0].copy()
                    
                    # Update global best
                    if fvals[0] < global_best_y:
                        global_best_y = fvals[0]
                        global_best_x = pop[0].copy()
                    
                    # Compute weighted mean
                    new_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                    
                    # Update evolution path
                    pc = (1 - c_c) * pc + c_c * (sqrt_inv_C @ (new_mean - mean) / sigma)
                    
                    # Update covariance matrix
                    delta_hs = 1.0 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(state['gen'] + 1)) < (1.4 + 2/(n+1)) * chi_n else 0.0
                    ps = (1 - c_sigma) * ps + delta_hs * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (sqrt_inv_C @ (new_mean - mean) / sigma)
                    
                    C = (1 - c_1 - c_mu) * C
                    C += c_1 * (np.outer(pc, pc) - C)
                    C += c_mu * sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu))
                    
                    # Ensure positive semi-definite
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                    
                    # Update sigma
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    
                    # Migration: replace worst island with global best periodically
                    if state['gen'] > 0 and state['gen'] % migration_interval == 0:
                        # Replace this island's mean with global best with some noise
                        noise = state['rng'].standard_normal(n) * 0.3
                        new_mean = np.clip(global_best_x + noise, -5, 5)
                        sigma = max(sigma, 0.5)
                        # Reset C matrix
                        C = np.eye(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-20)
                        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                        sqrt_inv_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    
                    # Update state
                    state['mean'] = new_mean
                    state['sigma'] = sigma
                    state['C'] = C
                    state['pc'] = pc
                    state['ps'] = ps
                    state['sqrt_C'] = sqrt_C
                    state['sqrt_inv_C'] = sqrt_inv_C
                    state['gen'] += 1
            
            # Update global best from islands
            for state in island_states:
                if state['best_y'] < global_best_y:
                    global_best_y = state['best_y']
                    global_best_x = state['best_x'].copy()
            
            if global_best_y < best_y:
                best_y = global_best_y
                best_x = global_best_x.copy()
        
        # Phase 3: Nelder-Mead polish
        remaining = budget - calls_made
        if remaining > 10 and best_x is not None and time_ok():
            nm_rng = np.random.default_rng(12345)
            nelder_mead(best_x, min(remaining, 50), nm_rng)
        
        return best_x, best_y
