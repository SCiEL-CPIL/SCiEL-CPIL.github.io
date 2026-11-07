"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol global exploration followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # --- Sobol sequence for global exploration ---
        def sobol_sequence(n, dim, seed=0):
            """Generate n quasi-random points in [0,1]^dim using Sobol-like sequence."""
            # Use a simple Halton-like quasi-random sequence as a fast alternative
            # that gives good space-filling properties
            rng = np.random.default_rng(seed)
            # Use stratified sampling with Latin hypercube for good coverage
            points = np.zeros((n, dim))
            for d in range(dim):
                # Stratified sampling: divide [0,1] into n strata and pick one point per stratum
                strata = np.arange(n) / n
                # Add random offset within each stratum
                offsets = rng.uniform(0, 1/n, n)
                points[:, d] = strata + offsets
            return points

        # Phase 1: Global exploration with Sobol/LHS sampling
        # Use a fraction of budget for global exploration
        n_global = min(50, budget // 4)
        n_global = max(n_global, 10)

        # Generate quasi-random points in [-5, 5]^dim
        sobol_points = sobol_sequence(n_global, dim, seed=42)
        global_candidates = []
        used_budget = 0

        for i in range(n_global):
            if time.time() > deadline:
                break
            x = sobol_points[i] * 10 - 5  # map [0,1] to [-5,5]
            x = np.clip(x, -5, 5)
            fx = func(x)
            used_budget += 1
            global_candidates.append((fx, x))

        # Sort by function value
        global_candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with Nelder-Mead on top candidates
        remaining_budget = budget - used_budget
        if remaining_budget <= 0:
            best_x, best_f = global_candidates[0]
            return best_x, best_f

        # Select top k candidates for local refinement
        k_top = min(3, len(global_candidates))
        # Allocate remaining budget among top candidates
        budget_per_candidate = max(1, remaining_budget // k_top)

        best_overall_x = global_candidates[0][1].copy()
        best_overall_f = global_candidates[0][0]

        for rank in range(k_top):
            if time.time() > deadline:
                break
            if used_budget >= budget:
                break

            start_x, start_f = global_candidates[rank]

            # Nelder-Mead simplex method
            best_x, best_f = self._nelder_mead(
                func, start_x, start_f, dim, budget_per_candidate,
                used_budget, deadline
            )
            used_budget += budget_per_candidate  # approximate

            if best_f < best_overall_f:
                best_overall_f = best_f
                best_overall_x = best_x.copy()

        return best_overall_x, best_overall_f

    def _nelder_mead(self, func, x0, f0, dim, budget, used_budget, deadline):
        """Nelder-Mead simplex method for local refinement."""
        # Initialize simplex
        alpha = 1.0  # expansion
        gamma = 2.0  # reflection
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # Create initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.1 if x0[i - 1] < 4.9 else -0.1

        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        fvals[0] = f0
        local_budget = budget
        local_used = 1  # already evaluated x0

        for i in range(1, dim + 1):
            if time.time() > deadline or local_used >= local_budget:
                break
            xi = np.clip(simplex[i], -5, 5)
            fvals[i] = func(xi)
            local_used += 1

        # Main Nelder-Mead loop
        while local_used < local_budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + gamma * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            local_used += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + alpha * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                local_used += 1
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
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                local_used += 1
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        local_used += 1
                        if local_used >= local_budget or time.time() > deadline:
                            break

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
