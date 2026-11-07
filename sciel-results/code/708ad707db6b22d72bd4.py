"""Rewritten optimizer for BBOB f23 (Katsuura) using multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for Katsuura on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            return x, fx

        # Phase 1: Multi-start Nelder-Mead
        # Katsuura is highly multimodal; use multiple starting points
        # Use a quasi-random (Halton-like) set of starts spread across the box
        n_starts = min(30, B // 20)  # at least ~20 calls per start for NM
        remaining_budget = B - n_starts * 5  # reserve some budget

        # Generate spread-out starting points using a simple lattice + jitter
        rng = np.random.default_rng(12345)
        
        # Create a grid of starting points
        n_per_dim = max(2, int(round(n_starts ** (1.0 / d))))
        grid_coords = np.linspace(-4.5, 4.5, n_per_dim)
        
        starts = []
        for idx in range(n_starts):
            # Deterministic spread with some randomness
            if idx < n_per_dim ** d:
                # Use grid point
                coords = np.unravel_index(idx, [n_per_dim] * d)
                x0 = np.array([grid_coords[c] for c in coords], dtype=np.float64)
            else:
                x0 = rng.uniform(-4.5, 4.5, d)
            # Add small jitter
            x0 = x0 + rng.normal(0, 0.1, d)
            x0 = np.clip(x0, -5.0, 5.0)
            starts.append(x0)

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        initial_simplex_size = 0.5

        for start in starts:
            if calls >= B - 50 or time.time() > deadline - 30:
                break

            # Initialize simplex
            x0, f0 = safe_call(start)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Build initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            for i in range(d):
                xi = x0.copy()
                xi[i] += initial_simplex_size
                xi = np.clip(xi, -5.0, 5.0)
                xi, fi = safe_call(xi)
                simplex.append(xi)
                simplex_f.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Nelder-Mead iterations
            max_nm_iters = 100
            for it in range(max_nm_iters):
                if calls >= B - 50 or time.time() > deadline - 30:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                xr, fr = safe_call(xr)

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    xe, fe = safe_call(xe)
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
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    xc, fc = safe_call(xc)

                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            xi, fi = safe_call(simplex[i])
                            simplex[i] = xi
                            simplex_f[i] = fi

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        # Phase 2: Polish with a fine local search around the best point
        if best_x is not None and calls < B - 10 and time.time() < deadline - 10:
            # Coordinate-wise fine tuning
            step = 0.01
            for coord in range(d):
                for _ in range(3):
                    if calls >= B - 5 or time.time() > deadline - 5:
                        break
                    # Try moving in positive direction
                    xp = best_x.copy()
                    xp[coord] += step
                    xp = np.clip(xp, -5.0, 5.0)
                    xp, fp = safe_call(xp)
                    if fp < best_f:
                        best_f = fp
                        best_x = xp
                    else:
                        # Try negative direction
                        xn = best_x.copy()
                        xn[coord] -= step
                        xn = np.clip(xn, -5.0, 5.0)
                        xn, fn = safe_call(xn)
                        if fn < best_f:
                            best_f = fn
                            best_x = xn
                    step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
