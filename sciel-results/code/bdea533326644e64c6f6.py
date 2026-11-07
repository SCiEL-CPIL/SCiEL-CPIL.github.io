"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s

        # Deterministic starting points spread in [-4, 4]^5
        # Use a Sobol-like or grid-based set for coverage
        # We'll use 10 distinct deterministic points
        points = []
        # Center
        points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                points.append(p)
        # Diagonal points
        for sign in [1, -1]:
            points.append(np.full(dim, sign * 2.0))
        
        # Select 10 points, ensuring diversity
        # Take first 10 from the list above
        start_points = points[:10]
        
        # Budget per run
        runs = len(start_points)
        budget_per_run = budget // runs
        # Ensure we don't exceed total budget
        # We'll track global call count

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Sort start points by distance from center (try center first)
        start_points = sorted(start_points, key=lambda p: np.linalg.norm(p))

        for start in start_points:
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Nelder-Mead simplex
            x0 = start.copy()
            f0 = func(x0)
            total_calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Build simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                total_calls += 1
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(budget_per_run - (dim + 1)):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                total_calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    total_calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    total_calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            total_calls += 1
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            fvals[i] = fi

        return best_x, best_f
# EVOLVE-BLOCK-END
