"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Rastrigin has period 2π ≈ 6.28; box is [-5,5]^5 with width 10.
        # The hidden shift places the global minimum somewhere in [-4,4]^5.
        # We'll use Nelder-Mead with restarts to navigate between basins.

        def nm_simplex(x0, f0, max_iter, rng, deadline, budget_used):
            """Run Nelder-Mead from x0, returning (best_x, best_f, calls_made)."""
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.5 if x0[i] < 4.5 else -0.5

            vals = np.zeros(n + 1)
            vals[0] = f0
            for i in range(n):
                vals[i + 1] = func(simplex[i + 1])
            calls = n + 1

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if budget_used + calls >= budget:
                    break

                # Sort by function value
                order = np.argsort(vals)
                simplex = simplex[order]
                vals = vals[order]

                # Check convergence
                if np.abs(vals[-1] - vals[0]) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                budget_used += 1

                if fr < vals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    budget_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        vals[-1] = fr
                elif fr < vals[-2]:
                    simplex[-1] = xr
                    vals[-1] = fr
                else:
                    # Contraction
                    if fr < vals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls += 1
                        budget_used += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                vals[i] = func(simplex[i])
                            calls += n
                            budget_used += n
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls += 1
                        budget_used += 1
                        if fc < vals[-1]:
                            simplex[-1] = xc
                            vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                vals[i] = func(simplex[i])
                            calls += n
                            budget_used += n

            # Return best point
            best_idx = np.argmin(vals)
            return simplex[best_idx], vals[best_idx], calls, budget_used

        # Multi-start Nelder-Mead: try several initial points
        # Rastrigin has local minima at integer multiples of 2π, so sample around those
        # But the hidden shift means we don't know exact positions. Use diverse starts.

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Generate diverse starting points
        # Combine: origin, random points, and points near multiples of 2π
        starts = []
        # Origin
        starts.append(np.zeros(dim))
        # Random uniform points
        for _ in range(8):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # Points near multiples of 2π (accounting for possible shift)
        for k in [-1, 0, 1]:
            for _ in range(2):
                pt = np.array([k * 2 * np.pi + rng.uniform(-1, 1) for _ in range(dim)])
                pt = np.clip(pt, -4.5, 4.5)
                starts.append(pt)

        # Evaluate all starting points and pick the best few
        start_vals = []
        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fv = func(s_clipped)
            start_vals.append((fv, s_clipped))
            total_calls += 1
            if time.time() > deadline:
                break

        # Sort by value and take top candidates
        start_vals.sort(key=lambda t: t[0])
        top_k = min(5, len(start_vals))

        # Run Nelder-Mead from each top starting point
        for i in range(top_k):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            x0, f0 = start_vals[i]
            remaining_budget = budget - total_calls
            # Allocate remaining budget across restarts
            max_iter = max(1, remaining_budget // (top_k - i))
            max_iter = min(max_iter, 500)

            bx, bf, calls, _ = nm_simplex(x0, f0, max_iter, rng, deadline, total_calls)
            total_calls += calls

            if bf < best_f:
                best_f = bf
                best_x = bx

            # If we're close to a good value, do a local refinement
            if bf < 1.0 and total_calls < budget - 50:
                # Tight local search
                x0_refine = bx.copy()
                f0_refine = bf
                max_iter_refine = min(200, budget - total_calls)
                bx2, bf2, calls2, _ = nm_simplex(x0_refine, f0_refine, max_iter_refine, rng, deadline, total_calls)
                total_calls += calls2
                if bf2 < best_f:
                    best_f = bf2
                    best_x = bx2

        # If best_x is still None (shouldn't happen), use first evaluation
        if best_x is None:
            best_x = start_vals[0][1]
            best_f = start_vals[0][0]

        return best_x, best_f
# EVOLVE-BLOCK-END
