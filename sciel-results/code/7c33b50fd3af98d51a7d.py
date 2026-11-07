"""CMA-ES with adaptive restarts and Nelder-Mead polish for f24 (Lunacek bi-Rastrigin)."""
import numpy as np
import time


class Optimizer:
    """
    CMA-ES based derivative-free optimizer with adaptive restarts.
    Uses multiple restarts with decreasing sigma, a global best tracker,
    and a final Nelder-Mead polish to refine the best solution found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        n = dim
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = max(4 + int(3 * np.log(n)), 6)
        # Initial sigma
        sigma0 = 2.0

        # We'll do multiple restarts with decreasing sigma
        # Each restart uses a portion of the budget
        max_restarts = 4
        budget_per_restart = budget // max_restarts

        global_best_x = None
        global_best_f = np.inf

        for restart in range(max_restarts):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            # For restart 0, start from center; for later restarts, start near best found
            if restart == 0 or global_best_x is None:
                x0 = np.zeros(n)
                # Try a few random points to get a better initial center
                for _ in range(min(5, budget - calls_made[0])):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    x_trial = rng.uniform(-4, 4, n)
                    f_trial = counted_func(x_trial)
                    if f_trial < global_best_f:
                        global_best_f = f_trial
                        global_best_x = x_trial.copy()
                if global_best_x is not None:
                    x0 = global_best_x.copy()
                else:
                    x0 = np.zeros(n)
            else:
                # Start near the global best with some perturbation
                x0 = global_best_x + rng.normal(0, 0.5, n)
                x0 = np.clip(x0, -5, 5)

            # Decreasing sigma for later restarts
            sigma = sigma0 * (0.5 ** restart)
            if sigma < 0.1:
                sigma = 0.1

            # CMA-ES state
            mean = x0.copy()
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)  # We'll use simple step sizes
            sigma_cma = sigma
            stepsize = sigma

            # CMA-ES loop
            max_iter = budget_per_restart
            no_improve = 0
            local_best_f = np.inf
            local_best_x = None

            for it in range(max_iter):
                if calls_made[0] >= budget or time.time() > deadline:
                    break

                # Generate population
                for j in range(lam):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break

                    # Sample from N(mean, stepsize^2 * I) - simplified CMA-ES
                    z = rng.standard_normal(n)
                    x_j = mean + stepsize * z
                    x_j = np.clip(x_j, -5, 5)

                    f_j = counted_func(x_j)

                    if f_j < local_best_f:
                        local_best_f = f_j
                        local_best_x = x_j.copy()

                    if f_j < global_best_f:
                        global_best_f = f_j
                        global_best_x = x_j.copy()

                if local_best_x is None:
                    break

                # Update mean toward best
                mean = local_best_x.copy()

                # Adapt step size: decrease if we're making progress, increase if stuck
                if local_best_f < global_best_f * 0.99 or no_improve > 20:
                    stepsize *= 0.9
                elif no_improve > 50:
                    stepsize *= 1.2
                    stepsize = min(stepsize, 2.0)

                # Check for convergence
                if stepsize < 1e-4:
                    break

                no_improve += 1

            # After CMA-ES phase, do a local polish with Nelder-Mead
            if local_best_x is not None and calls_made[0] < budget and time.time() < deadline:
                x_polished = self._nelder_mead_polish(
                    counted_func, local_best_x, dim, budget, calls_made, deadline, rng
                )
                if x_polished is not None:
                    f_polished = local_best_f  # Already counted
                    if local_best_f < global_best_f:
                        global_best_f = local_best_f
                        global_best_x = x_polished.copy()

        # Final: ensure we have a best
        if global_best_x is not None:
            best_x = global_best_x.copy()
            best_f = global_best_f
        else:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = counted_func(best_x)

        return best_x, best_f

    def _nelder_mead_polish(self, func, x0, dim, budget, calls_made, deadline, rng):
        """Nelder-Mead simplex method for local refinement."""
        # Initial simplex
        alpha = 0.1
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += alpha
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)

        # Evaluate simplex
        fvals = []
        for i in range(dim + 1):
            if calls_made[0] >= budget or time.time() > deadline:
                return x0
            fvals.append(func(simplex[i]))

        # Nelder-Mead parameters
        gamma = 1.0   # reflection
        rho = 2.0     # expansion
        sigma_nm = 0.5  # contraction
        epsilon = 1e-8

        max_iter = 50
        for _ in range(max_iter):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[0] - fvals[-1]) < epsilon:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + gamma * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            if calls_made[0] >= budget or time.time() > deadline:
                break

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + sigma_nm * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        if calls_made[0] >= budget or time.time() > deadline:
                            return simplex[0]
                        fvals[i] = func(simplex[i])

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx]
