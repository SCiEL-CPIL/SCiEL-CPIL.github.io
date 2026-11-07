"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with Nelder-Mead polish and final fine ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Phase allocation:
        # Phase 1: Portfolio (1+1)-ES — 70% of budget
        # Phase 2: Nelder-Mead polish — 20% of budget
        # Phase 3: Fine-grained ES polish — 10% of budget
        phase1_calls = int(budget * 0.70)
        phase2_calls = int(budget * 0.20)
        phase3_calls = budget - phase1_calls - phase2_calls

        n_runs = 4
        calls_per_run = phase1_calls // n_runs

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Diverse initial sigmas to cover different scales
        initial_sigmas = [4.0, 2.5, 1.5, 1.0]

        for run_idx in range(n_runs):
            if time.time() >= deadline - 30:
                break
            if total_calls >= phase1_calls:
                break

            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            sigma = initial_sigmas[run_idx]

            for _ in range(calls_per_run - 1):
                if time.time() >= deadline - 30:
                    break
                if total_calls >= phase1_calls:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                sigma = float(np.clip(sigma, 0.005, 8.0))

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        # Phase 2: Nelder-Mead polish
        if total_calls < budget - phase3_calls and time.time() < deadline - 20:
            nm_budget = min(phase2_calls, budget - total_calls - phase3_calls)
            if nm_budget > dim + 5:
                best_x, best_f = self._nelder_mead(func, best_x, best_f, dim, nm_budget, deadline)
                total_calls += nm_budget

        # Phase 3: Fine-grained ES polish
        if total_calls < budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - total_calls

            for _ in range(remaining):
                if time.time() >= deadline - 2:
                    break
                if total_calls >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.15
                else:
                    sigma *= 0.82

                sigma = float(np.clip(sigma, 0.001, 1.0))

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, max_calls, deadline):
        """Nelder-Mead simplex method with proper initialization."""
        # Initialize regular simplex centered at x0
        simplex = []
        fx_list = []

        simplex.append(x0.copy())
        fx_list.append(f0)

        # Create regular simplex vertices
        scale = 0.5  # initial simplex size
        for i in range(dim):
            vi = x0.copy()
            vi[i] += scale
            vi = np.clip(vi, -5, 5)
            simplex.append(vi)
            fx_list.append(func(vi))

        calls_used = 1 + dim

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        while calls_used < max_calls and time.time() < deadline - 5:
            order = np.argsort(fx_list)
            simplex = [simplex[i].copy() for i in order]
            fx_list = [fx_list[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = func(xr)
            calls_used += 1

            if fxr < fx_list[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = func(xe)
                calls_used += 1

                if fxe < fxr:
                    simplex[-1] = xe
                    fx_list[-1] = fxe
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fxr
            elif fxr < fx_list[-2]:
                simplex[-1] = xr
                fx_list[-1] = fxr
            else:
                # Contraction
                if fxr < fx_list[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fxc = func(xc)
                calls_used += 1

                if fxc <= min(fxr, fx_list[-1]):
                    simplex[-1] = xc
                    fx_list[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fx_list[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= max_calls or time.time() >= deadline - 5:
                            break
                    if calls_used >= max_calls or time.time() >= deadline - 5:
                        break

        best_idx = np.argmin(fx_list)
        return simplex[best_idx], fx_list[best_idx]
# EVOLVE-BLOCK-END
