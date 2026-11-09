"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive step size and restarts for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_used += 1
            if not np.isfinite(val):
                return 1e10
            return float(val)

        # Phase 1: Initial exploration with multiple random starts + Nelder-Mead
        # For Bent Cigar, we need to find the flat direction. Use multiple restarts.
        
        def nelder_mead(x0, f0, remaining_budget, max_iter=200):
            """Run Nelder-Mead from x0 with f0. Returns (best_x, best_f, calls_used)."""
            n = dim
            # Build initial simplex
            simplex = []
            simplex.append((x0.copy(), f0))
            
            # Create other vertices with small perturbations
            step = 0.5
            for i in range(n):
                xi = x0.copy()
                xi[i] += step
                simplex.append((xi.copy(), safe_func(xi)))
            
            # Sort by function value
            simplex.sort(key=lambda t: t[1])
            
            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            local_calls = 0
            for it in range(max_iter):
                if calls_used >= remaining_budget:
                    break
                if time.time() > deadline:
                    break
                
                # Get best and worst
                best = simplex[0]
                worst = simplex[-1]
                
                # Centroid of all but worst
                centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - worst[0])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_func(xr)
                local_calls += 1
                
                if fr < simplex[0][1]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst[0])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_func(xe)
                    local_calls += 1
                    if fe < fr:
                        simplex[-1] = (xe, fe)
                    else:
                        simplex[-1] = (xr, fr)
                elif fr < simplex[-2][1]:
                    simplex[-1] = (xr, fr)
                else:
                    # Contraction
                    if fr < worst[1]:
                        # Outside contraction
                        xc = centroid + rho * (centroid - worst[0])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - worst[0])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_func(xc)
                    local_calls += 1
                    
                    if fc <= min(fr, worst[1]):
                        simplex[-1] = (xc, fc)
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = (simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0]),
                                          safe_func(simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])))
                            local_calls += 1
                
                # Sort again
                simplex.sort(key=lambda t: t[1])
                
                # Check convergence
                spread = np.max(np.abs([s[1] for s in simplex]) - np.min([s[1] for s in simplex]))
                if spread < 1e-12:
                    break
            
            # Return best
            best_s = simplex[0]
            return best_s[0], best_s[1], local_calls

        # Strategy: Use a few random starts, run Nelder-Mead from each, keep best
        # Bent Cigar is highly anisotropic: f(x) = (x1)^2 + 100*(x2-x1)^2 + ... 
        # Actually BBOB f12: f(x) = (x_1)^2 + 100 * sum_{i=2}^n (x_i - x_{i-1})^2
        # The flat direction is along the diagonal.
        
        # Try several starting points
        num_starts = min(5, budget // 100)
        
        for start_idx in range(num_starts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            
            if start_idx == 0:
                x0 = rng.uniform(-3, 3, dim)
            else:
                # Spread out starts
                x0 = rng.uniform(-5, 5, dim)
            
            f0 = safe_func(x0)
            calls_used += 1
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            remaining = budget - calls_used
            if remaining < 10:
                break
            
            # Run Nelder-Mead
            bx, bf, _ = nelder_mead(x0, f0, budget)
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        # Phase 2: Final polish with a tighter Nelder-Mead or coordinate descent
        if best_x is not None and calls_used < budget - 5 and time.time() < deadline:
            # Try a small local search around the best point
            # Use a very small step Nelder-Mead
            x0 = best_x.copy()
            f0 = best_f
            # Small initial simplex
            simplex = [(x0.copy(), f0)]
            small_step = 0.1
            for i in range(dim):
                xi = x0.copy()
                xi[i] += small_step
                simplex.append((xi.copy(), safe_func(xi)))
            
            simplex.sort(key=lambda t: t[1])
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            n = dim
            
            for it in range(100):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                best = simplex[0]
                worst = simplex[-1]
                centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)
                
                xr = centroid + alpha * (centroid - worst[0])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_func(xr)
                
                if fr < simplex[0][1]:
                    xe = centroid + gamma * (centroid - worst[0])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[-1] = (xe, fe)
                    else:
                        simplex[-1] = (xr, fr)
                elif fr < simplex[-2][1]:
                    simplex[-1] = (xr, fr)
                else:
                    if fr < worst[1]:
                        xc = centroid + rho * (centroid - worst[0])
                    else:
                        xc = centroid - rho * (centroid - worst[0])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_func(xc)
                    
                    if fc <= min(fr, worst[1]):
                        simplex[-1] = (xc, fc)
                    else:
                        for i in range(1, n + 1):
                            new_x = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                            simplex[i] = (new_x, safe_func(new_x))
                
                simplex.sort(key=lambda t: t[1])
                
                spread = np.max(np.abs([s[1] for s in simplex]) - np.min([s[1] for s in simplex]))
                if spread < 1e-14:
                    break
            
            if simplex[0][1] < best_f:
                best_f = simplex[0][1]
                best_x = simplex[0][0].copy()

        # Ensure we made at least one call
        if calls_used == 0:
            x = np.zeros(dim)
            safe_func(x)
            best_x = x
            best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
