"""Multi-start Nelder-Mead optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # 290s margin within 300s limit (evaluator kills at 330s, task says 300s)
        # Using 290s to be safe against overhead.
        deadline = start_time + 290 

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(func, x0, max_iter=50, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0, eps=1e-8):
            # Local Nelder-Mead implementation
            # x0: initial point (dim,)
            # Returns best point found
            
            # Create initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i, i - 1] = x0[i - 1] + 0.25 if x0[i - 1] < 4.75 else x0[i - 1] - 0.25
                # Ensure within bounds
                simplex[i] = np.clip(simplex[i], -5.0, 5.0)

            # Evaluate initial simplex
            fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
            local_best_x = simplex[0].copy()
            local_best_f = fvals[0]
            
            # Track local best
            if fvals[0] < local_best_f:
                local_best_f = fvals[0]
                local_best_x = simplex[0].copy()
            for i in range(1, dim+1):
                if fvals[i] < local_best_f:
                    local_best_f = fvals[i]
                    local_best_x = simplex[i].copy()

            # Sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            for _ in range(max_iter):
                # Check time and budget
                if time.time() > deadline or total_calls >= budget:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + sigma * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = func(simplex[i])
                
                # Update local best
                for i in range(dim + 1):
                    if fvals[i] < local_best_f:
                        local_best_f = fvals[i]
                        local_best_x = simplex[i].copy()

                # Sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Convergence check
                if np.std(fvals) < eps:
                    break

            return local_best_x, local_best_f

        # Strategy: Multi-start
        # Phase 1: Coarse random search to find diverse starting points
        # We use about 20% of the budget for exploration
        n_starts = 5
        exploration_budget = int(budget * 0.2)
        points = []
        
        # Initial center
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        points.append((x0, f0))
        
        # Random points in [-4, 4]^5 to avoid boundary issues
        rng = np.random.default_rng(123)
        
        for _ in range(exploration_budget - 1):
            if time.time() > deadline or total_calls >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            fx = safe_call(x)
            points.append((x, fx))

        # Select top n_starts points as starting points for Nelder-Mead
        points.sort(key=lambda p: p[1])
        start_points = [p[0] for p in points[:n_starts]]
        
        # Phase 2: Local refinement with Nelder-Mead
        # Distribute remaining budget among starts
        remaining_budget = budget - total_calls
        budget_per_start = remaining_budget // n_starts
        
        for sp in start_points:
            if time.time() > deadline or total_calls >= budget:
                break
            
            # Run Nelder-Mead
            # We pass a wrapper that respects the global budget and time
            # But Nelder-Mead is local, so we just run it.
            # The safe_call inside nelder_mead is actually func, which is safe_call.
            # Wait, nelder_mead calls func directly. We need to ensure it uses safe_call.
            # Let's pass safe_call to nelder_mead.
            
            x_opt, f_opt = nelder_mead(safe_call, sp, max_iter=budget_per_start)
            
            # Update global best
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
