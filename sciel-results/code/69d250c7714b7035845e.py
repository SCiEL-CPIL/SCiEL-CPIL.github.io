"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: Coarse LHS sampling followed by Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, sigma, remaining_budget, time_limit, call_counter):
        """
        Nelder-Mead simplex method.
        call_counter: list with one element [current_calls] to track global calls.
        Returns (best_x, best_f).
        """
        dim = self.dim
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += sigma

        fvals = np.empty(dim + 1)
        
        # Evaluate initial simplex
        for i in range(dim + 1):
            if call_counter[0] >= remaining_budget or time.time() > time_limit:
                return simplex[0], fvals[0] if i > 0 else func(simplex[0])
            fvals[i] = func(simplex[i])
            call_counter[0] += 1

        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        # NM parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_shrink = 0.5  # shrink

        while call_counter[0] < remaining_budget and time.time() < time_limit:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            best_x = simplex[0].copy()
            best_f = fvals[0]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            worst = simplex[-1]
            xr = centroid + alpha * (centroid - worst)
            
            if call_counter[0] >= remaining_budget: break
            fr = func(xr)
            call_counter[0] += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                if call_counter[0] >= remaining_budget:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    break
                fe = func(xe)
                call_counter[0] += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Simple reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (xr - centroid)
                    if call_counter[0] >= remaining_budget: break
                    fc = func(xc)
                    call_counter[0] += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if call_counter[0] >= remaining_budget: break
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            call_counter[0] += 1
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (worst - centroid)
                    if call_counter[0] >= remaining_budget: break
                    fc = func(xc)
                    call_counter[0] += 1
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if call_counter[0] >= remaining_budget: break
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            call_counter[0] += 1

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = start_time + 290.0  # safety margin before 300s

        rng = np.random.default_rng(42)
        call_counter = [0]
        
        # Stage 1: Coarse Latin Hypercube-like sampling
        # Use 20 points to bracket the global basin
        n_coarse = 20
        if budget <= n_coarse:
            # If budget is very small, just sample
            for _ in range(budget):
                x = rng.uniform(-4, 4, dim)
                f = func(x)
                call_counter[0] += 1
            return np.zeros(dim), 0.0 # Fallback, shouldn't happen with budget=500

        coarse_points = rng.uniform(-4, 4, (n_coarse, dim))
        best_x = None
        best_f = np.inf
        
        for i in range(n_coarse):
            if call_counter[0] >= budget or time.time() > time_limit:
                break
            x = coarse_points[i]
            f = func(x)
            call_counter[0] += 1
            if f < best_f:
                best_f = f
                best_x = x

        # Stage 2: Refinement with Nelder-Mead
        if best_x is not None and call_counter[0] < budget:
            remaining_budget = budget - call_counter[0]
            # Use a moderate sigma for refinement
            sigma = 1.0
            best_x, best_f = self._nelder_mead(
                func, best_x, sigma, budget, time_limit, call_counter
            )

        return best_x, best_f
# EVOLVE-BLOCK-END
