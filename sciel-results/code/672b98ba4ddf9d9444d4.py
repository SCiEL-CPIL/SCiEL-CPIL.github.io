"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # We will track the best solution found
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_used += 1
            return float(val)

        # Initialize with a few diverse points to get a sense of the landscape
        # Use a fixed seed for reproducibility within a run, but different from 0
        rng = np.random.default_rng(42)

        # Generate initial candidate points: center, corners, and random
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Corners of the box
        for mask in range(2 ** dim):
            pt = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
            init_points.append(pt)
        # Some random points
        for _ in range(10):
            init_points.append(rng.uniform(-5, 5, dim))

        # Evaluate initial points to find a good starting point for NM
        # We have budget to spare, so let's use ~20-30 calls for initialization
        init_evals = min(20, budget // 10)
        # Shuffle init points to avoid bias
        idx = rng.permutation(len(init_points))
        for i in range(init_evals):
            if calls_used >= budget:
                break
            pt = init_points[idx[i]]
            f_val = safe_func(pt)
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # If we didn't find anything better than inf (shouldn't happen), pick a random point
        if np.isinf(best_f):
            pt = rng.uniform(-5, 5, dim)
            best_f = safe_func(pt)
            best_x = pt.copy()

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, max_iter=500):
            nonlocal best_x, best_f, calls_used, deadline
            dim = self.dim

            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [f0]

            # Create other vertices with a small step
            step = 0.5
            for i in range(dim):
                xi = x0.copy()
                # Determine step size based on position
                if xi[i] > 0:
                    xi[i] -= step
                else:
                    xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_func(xi)
                simplex.append(xi)
                f_simplex.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = safe_func(x_r)
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = safe_func(x_e)
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = safe_func(x_c)
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()

                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_simplex[i] = safe_func(simplex[i])
                            if f_simplex[i] < best_f:
                                best_f = f_simplex[i]
                                best_x = simplex[i].copy()

                # Sort simplex again
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Check for convergence
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                    break

            return best_x, best_f

        # Multi-start Nelder-Mead
        # We have remaining budget. Let's divide it among several restarts.
        # Start with the best point found so far, then try other promising regions.
        # We'll do a few restarts from different points.

        # Strategy:
        # 1. Run NM from the best initial point found.
        # 2. If budget remains, run NM from other diverse points.
        # We need to be careful with budget allocation.

        # Let's reserve some budget for the first NM run.
        # A good heuristic: give the first run a significant chunk, then split the rest.
        
        # Determine number of restarts
        # If we have plenty of budget, do more restarts.
        # Let's say we do up to 5 restarts.
        num_restarts = 5
        
        # Allocate budget per restart
        # First restart gets more budget
        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Split remaining budget among restarts
        # First restart gets 40%, others split the rest
        restart_budgets = []
        first_budget = int(remaining_budget * 0.4)
        restart_budgets.append(first_budget)
        rest_budget = remaining_budget - first_budget
        per_restart = rest_budget // (num_restarts - 1) if num_restarts > 1 else 0
        for i in range(num_restarts - 1):
            restart_budgets.append(per_restart)
        
        # Ensure the last one gets any leftover
        if num_restarts > 1:
            restart_budgets[-1] += remaining_budget - sum(restart_budgets)

        # Prepare starting points for restarts
        # Restart 1: best_x
        # Restart 2-5: other diverse points from initial set or new randoms
        start_points = [best_x.copy()]
        
        # Add some diverse points
        for i in range(1, num_restarts):
            if i < len(init_points):
                start_points.append(init_points[idx[i]].copy())
            else:
                start_points.append(rng.uniform(-5, 5, dim))

        for i, start_pt in enumerate(start_points):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Evaluate starting point if not already done
            if np.array_equal(start_pt, best_x) and i == 0:
                f_start = best_f
            else:
                # We might have evaluated this point during init, but let's be safe
                # and just call safe_func. It's cheap and ensures we have a value.
                # However, to save budget, we could check if it's in our init cache.
                # For simplicity, just call it.
                f_start = safe_func(start_pt)
                if f_start < best_f:
                    best_f = f_start
                    best_x = start_pt.copy()

            # Run NM with allocated budget
            # max_iter is a soft limit, budget is the hard limit
            max_iter = restart_budgets[i]
            nelder_mead(start_pt, f_start, max_iter=max_iter)

        return best_x, best_f
# EVOLVE-BLOCK-END
