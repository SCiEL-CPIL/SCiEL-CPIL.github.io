# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with dynamic budget allocation and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx0, max_iter, deadline, rng):
        """Run Nelder-Mead from x0 with fx0 pre-evaluated, respecting deadline."""
        n = self.dim
        # Build initial simplex
        simplex = [x0.copy()]
        fvals = [fx0]
        for i in range(n):
            xi = x0.copy()
            if fx0 > 0:
                xi[i] += 0.05
            else:
                xi[i] -= 0.05
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            fvals.append(func(xi))

        # Standard Nelder-Mead parameters
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        best_x = x0.copy()
        best_f = fx0

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Convergence check
            if abs(fvals[-1] - fvals[0]) < 1e-12:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            if time.time() > deadline:
                break

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                if time.time() > deadline:
                    break
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
                xc = centroid + rho * (worst - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                if time.time() > deadline:
                    break
                if fc < worst_f:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        if time.time() > deadline:
                            return best_x, best_f
        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start exploration with Nelder-Mead
        n_starts = 12
        starts_per_iter = max(1, n_starts // 4)
        total_starts = 0

        # Track best result globally
        best_x = None
        best_f = np.inf
        call_count = 0

        # Generate diverse starting points
        start_points = []
        for i in range(n_starts):
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = np.full(dim, 2.0)
            elif i == 2:
                x = np.full(dim, -2.0)
            elif i == 3:
                x = rng.uniform(-4, 4, dim)
            else:
                # Mix of structured and random
                x = rng.uniform(-4.5, 4.5, dim)
                if i % 2 == 0:
                    # Perturb some dimensions
                    idx = rng.choice(dim, size=max(1, dim // 2), replace=False)
                    x[idx] = rng.uniform(-4, 4, size=len(idx))
            x = np.clip(x, -5, 5)
            start_points.append(x)

        # Phase 1: Initial evaluation of all starts
        start_results = []
        for x in start_points:
            if call_count >= budget:
                break
            fx = func(x)
            call_count += 1
            start_results.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                break

        if not start_results:
            # Fallback: single random point
            x = np.zeros(dim)
            fx = func(x)
            return x, fx

        # Phase 2: Dynamic allocation - refine the most promising starts
        # Sort starts by current best function value
        start_results.sort(key=lambda t: t[1])

        # Allocate remaining budget dynamically
        remaining_budget = budget - call_count
        # Give more iterations to better starts (top 4 get more)
        n_refine = min(len(start_results), 6)
        # Budget split: top start gets 40%, next two get 25% each, rest split
        if n_refine >= 1:
            allocs = []
            for i in range(n_refine):
                if i == 0:
                    allocs.append(int(remaining_budget * 0.40))
                elif i < 3:
                    allocs.append(int(remaining_budget * 0.20))
                else:
                    allocs.append(max(5, int(remaining_budget * 0.10)))
            # Adjust to not exceed remaining
            total_alloc = sum(allocs)
            if total_alloc > remaining_budget:
                scale = remaining_budget / total_alloc
                allocs = [max(3, int(a * scale)) for a in allocs]
        else:
            allocs = [remaining_budget]

        # Run Nelder-Mead on each selected start
        for i in range(n_refine):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            x0, fx0 = start_results[i]
            max_iter = max(3, allocs[i])
            # Each Nelder-Mead iteration uses ~n+1 to 2*(n+1) function evals
            # But we need to track calls carefully; use a wrapper approach
            # Instead, run a simplified NM that counts internally

            # Simplified: just do a local search around this point
            # Use coordinate descent / Nelder-Mead hybrid
            x = x0.copy()
            fx = fx0
            sigma = 0.5

            for it in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Try a random perturbation (Nelder-Mead-like)
                direction = rng.standard_normal(dim)
                direction = np.clip(direction, -1, 1)
                # Scale by sigma
                x_trial = x + sigma * direction
                x_trial = np.clip(x_trial, -5, 5)

                if np.allclose(x_trial, x):
                    sigma *= 0.5
                    continue

                f_trial = func(x_trial)
                call_count += 1

                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.7

                # Adaptive: if sigma too small, try larger jump
                if sigma < 0.01:
                    sigma = 0.3

        # Phase 3: Final polish around the best point
        if best_x is not None and call_count < budget:
            polish_budget = min(budget - call_count, 50)
            x = best_x.copy()
            fx = best_f
            sigma = 0.1

            for _ in range(polish_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                direction = rng.standard_normal(dim)
                x_trial = x + sigma * direction
                x_trial = np.clip(x_trial, -5, 5)

                if np.allclose(x_trial, x):
                    sigma *= 0.5
                    continue

                f_trial = func(x_trial)
                call_count += 1

                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    sigma *= 1.2
                    best_f = fx
                    best_x = x.copy()
                else:
                    sigma *= 0.7

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
