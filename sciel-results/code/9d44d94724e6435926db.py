"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer with Coordinate Descent Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0

        def call_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            return func(x)

        def nelder_mead(func, x0, max_iter, initial_step=1.0):
            """
            Deterministic Nelder-Mead implementation.
            Returns (best_x, best_f, calls_used_in_this_run)
            """
            n = dim
            # Initialize simplex
            # Vertices: x0, x0 + step*e_i
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += initial_step
                simplex.append(xi)
            
            fvals = np.array([call_func(v) for v in simplex])
            local_calls = n + 1
            best_f = np.min(fvals)
            best_x = simplex[np.argmin(fvals)].copy()

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if local_calls >= budget:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # 1. Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = call_func(xr)
                local_calls += 1
                if fr < fvals[0]:
                    # 2. Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = call_func(xe)
                    local_calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # 3. Contraction (inside)
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = call_func(xc)
                    local_calls += 1
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # 4. Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = call_func(simplex[i])
                        local_calls += 1

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f, local_calls

        def coordinate_descent(func, x0, max_iter, step_size=0.1):
            """
            Coordinate descent polish.
            Returns (best_x, best_f, calls_used_in_this_run)
            """
            x = x0.copy()
            fx = call_func(x)
            local_calls = 1
            best_f = fx
            best_x = x.copy()

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if local_calls >= budget:
                    break
                improved = False
                for i in range(dim):
                    if time.time() > deadline or local_calls >= budget:
                        break
                    # Try positive step
                    xp = x.copy()
                    xp[i] += step_size
                    fp = call_func(xp)
                    local_calls += 1
                    if fp < fx:
                        x = xp
                        fx = fp
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        # Try to go further (optional, but let's keep it simple 1-step for budget)
                    else:
                        # Try negative step
                        xn = x.copy()
                        xn[i] -= step_size
                        fn = call_func(xn)
                        local_calls += 1
                        if fn < fx:
                            x = xn
                            fx = fn
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

                if not improved:
                    # No improvement in full sweep, reduce step size
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

            return best_x, best_f, local_calls

        # Strategy: 
        # 1. Use Nelder-Mead for the bulk of the search (it handles the valley well).
        # 2. If budget remains, polish with Coordinate Descent.
        
        # Initial point: Center of box is often good for Rosenbrock if shifted, 
        # but let's start at 0 and let NM explore.
        x_init = np.zeros(dim)
        
        # Reserve some budget for polish? 
        # NM needs at least n+1 calls per iteration. 
        # Let's give NM 70% of the budget, CD 30%.
        nm_budget = int(budget * 0.7)
        cd_budget = int(budget * 0.3)

        # Run Nelder-Mead
        best_x, best_f, nm_calls = nelder_mead(func, x_init, max_iter=nm_budget, initial_step=2.0)
        
        # If we have time and budget left, run Coordinate Descent from best_x
        if calls_used < budget and time.time() < deadline:
            # CD works well in the narrow valley of Rosenbrock once close
            # Start with a moderate step
            cd_x, cd_f, cd_calls = coordinate_descent(func, best_x, max_iter=cd_budget, step_size=0.5)
            
            if cd_f < best_f:
                best_x = cd_x
                best_f = cd_f

        return best_x, best_f
# EVOLVE-BLOCK-END
