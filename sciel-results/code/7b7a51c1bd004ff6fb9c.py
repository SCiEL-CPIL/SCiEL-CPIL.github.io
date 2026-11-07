"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with random restarts and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return float(func(x))

        # (1+1)-ES implementation
        def es_11(func_eval, x0, sigma0, max_calls, deadline):
            nonlocal best_x, best_f, calls_made
            n = dim
            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            rng = np.random.default_rng(42)
            
            # Adaptation rate for sigma
            # Standard (1+1)-ES update: sigma *= exp((1/n) * (1 if success else -1))
            # Or more commonly: sigma *= exp((1/(10*n)) * (1 if success else -1))
            # Let's use the standard 1/10n rate for stability
            rate = 1.0 / (10.0 * n)
            
            while calls_made < max_calls and time.time() < deadline:
                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                
                # Evaluate
                fx = func_eval(x)
                fy = func_eval(y)
                calls_made += 2
                
                if fy < fx:
                    # Success
                    x = y
                    sigma *= np.exp(rate)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    # Failure
                    sigma *= np.exp(-rate)
                    # Keep x
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Early stopping if sigma is very small or large
                if sigma < 1e-6:
                    break
            return best_x, best_f

        # Nelder-Mead polish
        def nelder_mead(func_eval, x0, max_calls, deadline):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            n = dim
            # Initialize simplex
            simplex = [np.array(x0, dtype=np.float64)]
            for i in range(n):
                point = simplex[0].copy()
                if point[i] != 0:
                    point[i] *= 1.05
                else:
                    point[i] = 0.01
                point = np.clip(point, -5.0, 5.0)
                simplex.append(point)

            fvals = [func_eval(p) for p in simplex]
            calls_made += n + 1
            for f, p in zip(fvals, simplex):
                if f < best_f:
                    best_f = f
                    best_x = p.copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_iter = max(1, (max_calls - calls_made) // (n + 1))

            for _ in range(max_iter):
                if time.time() >= deadline or calls_made >= max_calls:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.sum(simplex[:n], axis=0) / n

                # Reflection
                xr = centroid + alpha * (centroid - simplex[n])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func_eval(xr)
                calls_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[n])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func_eval(xe)
                    calls_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                elif fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                else:
                    # Contraction
                    if fr < fvals[n]:
                        # Outside contraction
                        xc = centroid + 0.5 * (simplex[n] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[n])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func_eval(xc)
                    calls_made += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()

                    if fc < min(fr, fvals[n]):
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = func_eval(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

            return best_x, best_f

        # Strategy:
        # 1. Global random sampling to find a decent start (20 calls)
        # 2. Run (1+1)-ES from that start with a large budget
        # 3. Polish with Nelder-Mead
        
        rng = np.random.default_rng(123)
        
        # Phase 1: Global Search
        global_calls = min(20, budget)
        for _ in range(global_calls):
            if calls_made >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # Phase 2: (1+1)-ES
        # Reserve some budget for NM polish
        polish_budget = min(100, budget // 5)
        es_budget = budget - polish_budget
        
        if es_budget > 10 and time.time() < deadline:
            # Start ES from best point found
            # Use a moderate initial sigma
            es_11(safe_func, best_x, 1.0, calls_made + es_budget, deadline)
            
        # Phase 3: Nelder-Mead Polish
        if calls_made < budget and time.time() < deadline:
            if best_x is not None:
                nelder_mead(safe_func, best_x, budget, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
