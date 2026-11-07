"""Rewritten optimizer for BBOB f23 (Katsuura) with multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start Nelder-Mead for Katsuura with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        # Wrap func to count calls and check deadline
        call_count = [0]
        best_x = [None]
        best_f = [float('inf')]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > self.budget:
                return best_f[0]  # Return best known to avoid extra calls
            fx = func(np.clip(x, -5.0, 5.0))
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Generate diverse initial points
        # Mix of random uniform, grid-like, and center-biased points
        initial_points = []

        # 1. Random uniform points
        n_random = 8
        for _ in range(n_random):
            x = rng.uniform(-5, 5, self.dim)
            initial_points.append(x)

        # 2. Center-biased points (Gaussian with decreasing sigma)
        for sigma in [2.0, 1.0, 0.5]:
            x = sigma * rng.standard_normal(self.dim)
            x = np.clip(x, -5, 5)
            initial_points.append(x)

        # 3. Some structured points
        # Corner-ish points
        for i in range(4):
            x = rng.choice([-3, -1, 1, 3], size=self.dim)
            initial_points.append(x)

        # Sort initial points by function value to prioritize good starts
        # But we need to evaluate them first... let's just evaluate all and pick top ones
        # Actually, let's evaluate each initial point and keep track

        # Evaluate all initial points
        evaluated_starts = []
        for x in initial_points:
            if call_count[0] >= self.budget:
                break
            fx = counted_func(x)
            evaluated_starts.append((fx, x.copy()))

        # Sort by function value
        evaluated_starts.sort(key=lambda t: t[0])

        # Now do Nelder-Mead from the best starts
        # We'll allocate budget across multiple starts
        # Strategy: try each start with a limited budget, keep best result

        # Determine how many function calls to use per Nelder-Mead run
        # Total remaining budget
        remaining = self.budget - call_count[0]

        # Allocate budget: give more to better starts
        # Use top k starts with proportional budget
        n_starts_to_try = min(len(evaluated_starts), 5)

        # Budget allocation: decreasing for worse starts
        # First start gets 40%, second 25%, third 15%, fourth 10%, fifth 10%
        if n_starts_to_try >= 5:
            fractions = [0.35, 0.25, 0.15, 0.12, 0.13]
        elif n_starts_to_try == 4:
            fractions = [0.40, 0.25, 0.20, 0.15]
        elif n_starts_to_try == 3:
            fractions = [0.45, 0.30, 0.25]
        elif n_starts_to_try == 2:
            fractions = [0.55, 0.45]
        elif n_starts_to_try == 1:
            fractions = [1.0]
        else:
            fractions = []

        for i, (fx_start, x_start) in enumerate(evaluated_starts[:n_starts_to_try]):
            if i >= len(fractions):
                break

            if call_count[0] >= self.budget:
                break

            # Check time
            if time.time() > deadline - 10.0:
                break

            # Budget for this Nelder-Mead run
            nm_budget = int(remaining * fractions[i])
            nm_budget = max(nm_budget, 10)  # Minimum 10 calls

            # Don't exceed remaining budget
            nm_budget = min(nm_budget, self.budget - call_count[0])

            if nm_budget < 5:
                break

            # Run Nelder-Mead
            x_best, f_best = self._nelder_mead(
                counted_func, x_start, nm_budget, deadline
            )

            # Update global best
            if f_best < best_f[0]:
                best_f[0] = f_best
                best_x[0] = x_best.copy()

            remaining = self.budget - call_count[0]

            # If we've used most of the budget, stop
            if remaining < 10:
                break

        return best_x[0], best_f[0]

    def _nelder_mead(self, func, x0, budget, deadline):
        """Nelder-Mead simplex method with adaptive parameters."""
        dim = len(x0)

        # Initialize simplex
        x0 = np.array(x0, dtype=float)
        f0 = func(x0)

        # Create simplex: x0 plus perturbations in each direction
        simplex = [x0.copy()]
        simplex_f = [f0]

        # Initial step size
        step = 0.5
        for i in range(dim):
            xi = x0.copy()
            if xi[i] == 0:
                xi[i] = step
            else:
                xi[i] += step
            xi = np.clip(xi, -5.0, 5.0)
            fi = func(xi)
            simplex.append(xi)
            simplex_f.append(fi)

        # Sort simplex by function value
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        calls_used = 1 + dim  # Initial simplex evaluations
        best_x = simplex[0].copy()
        best_f = simplex_f[0]

        while calls_used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Get centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            calls_used += 1

            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                calls_used += 1

                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)

                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)
                calls_used += 1

                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = func(simplex[i])
                        simplex_f[i] = fi
                        calls_used += 1

                    if calls_used >= budget:
                        break

            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()

            # Check convergence: if simplex is very small, restart with smaller step
            spread = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, dim + 1)])
            if spread < 1e-6:
                # Shrink the simplex and continue
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                    fi = func(simplex[i])
                    simplex_f[i] = fi
                    calls_used += 1

                    if calls_used >= budget:
                        break

                if calls_used >= budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

        return best_x, best_f
# EVOLVE-BLOCK-END
