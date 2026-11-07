"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and deadline guard for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        # Global best tracking
        best_x = None
        best_f = np.inf

        # Phase 1: Broad exploration with multiple random starts
        # Try several different starting points to find the basin
        starts = []
        rng = np.random.default_rng(42)
        
        # First start at origin
        starts.append(np.zeros(dim))
        
        # Add a few random starts
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Evaluate all starts
        for x0 in starts:
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: Nelder-Mead optimization from the best start
        # Build initial simplex
        alpha = 0.5
        x_best = best_x.copy()
        f_best = best_f

        # Create simplex
        simplex = [x_best.copy()]
        f_values = [f_best]
        
        for i in range(dim):
            new_point = x_best.copy()
            # If x_best[i] is near boundary, step inward; otherwise step outward
            if new_point[i] > 0:
                new_point[i] = max(-5.0, new_point[i] - alpha)
            else:
                new_point[i] = min(5.0, new_point[i] + alpha)
            new_point = np.clip(new_point, -5, 5)
            f_new = func(new_point)
            simplex.append(new_point)
            f_values.append(f_new)
            if f_new < best_f:
                best_f = f_new
                best_x = new_point.copy()

        # Nelder-Mead parameters
        gamma = 1.0   # reflection
        rho = 2.0     # expansion
        sigma = 0.5   # contraction
        xi = 0.5      # shrink

        calls_made = len(starts) + dim  # starts + simplex construction
        stall_counter = 0
        no_improve_threshold = 10 * (dim + 1)
        restarts = 0
        max_restarts = 3

        while calls_made < budget and time.time() < deadline:
            # Sort simplex by function value
            indices = np.argsort(f_values)
            simplex = [simplex[i] for i in indices]
            f_values = [f_values[i] for i in indices]

            # Check for convergence
            f_range = f_values[-1] - f_values[0]
            if f_range < 1e-10:
                break

            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + gamma * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1

            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outside contraction
                    x_c = centroid + sigma * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - sigma * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + xi * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls_made += 1

            # Update global best
            for i, f in enumerate(f_values):
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()

            # Stall detection
            if f_values[0] < best_f + 1e-12:
                stall_counter = 0
            else:
                stall_counter += 1

            if stall_counter > no_improve_threshold and restarts < max_restarts:
                # Restart from a new random point near the best
                restarts += 1
                stall_counter = 0
                x_restart = best_x + 0.5 * np.random.default_rng(restarts).standard_normal(dim)
                x_restart = np.clip(x_restart, -5, 5)
                f_restart = func(x_restart)
                calls_made += 1
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()

                # Rebuild simplex
                simplex = [best_x.copy()]
                f_values = [best_f]
                for i in range(dim):
                    new_point = best_x.copy()
                    if new_point[i] > 0:
                        new_point[i] = max(-5.0, new_point[i] - alpha)
                    else:
                        new_point[i] = min(5.0, new_point[i] + alpha)
                    new_point = np.clip(new_point, -5, 5)
                    f_new = func(new_point)
                    simplex.append(new_point)
                    f_values.append(f_new)
                    if f_new < best_f:
                        best_f = f_new
                        best_x = new_point.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
