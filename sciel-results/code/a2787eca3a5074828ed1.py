"""Optimized black-box continuous optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Sobol sampling + Nelder-Mead refinement for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Generate Sobol sequence points
        # We need enough points for initial exploration
        # Use scipy's Sobol or implement a simple one
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        except ImportError:
            # Fallback: use a simple quasi-random sequence
            sobol = None

        best_x = None
        best_f = np.inf
        calls = 0

        def check_deadline():
            return time.time() > deadline

        def do_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sequence sampling for systematic exploration
        if sobol is not None:
            # Generate 200 Sobol points
            n_sobol = min(200, budget)
            samples = sobol.random(n_sobol)
            # Scale from [0,1]^dim to [-5,5]^dim
            samples = samples * 10 - 5
            for i in range(n_sobol):
                if check_deadline() or calls >= budget:
                    break
                do_call(samples[i])
        else:
            # Fallback: uniform random sampling
            rng = np.random.default_rng(42)
            for i in range(min(200, budget)):
                if check_deadline() or calls >= budget:
                    break
                x = rng.uniform(-5, 5, dim)
                do_call(x)

        # Phase 2: Nelder-Mead simplex method on the best candidate
        if best_x is not None and not check_deadline() and calls < budget:
            # Initialize simplex
            x0 = best_x.copy()
            f0 = best_f

            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 if abs(x0[i]) < 4.5 else 0.05
                if x0[i] > 0:
                    xi[i] -= step
                else:
                    xi[i] += step
                simplex.append(xi)

            # Evaluate initial simplex
            simplex_f = []
            for i in range(dim + 1):
                if check_deadline() or calls >= budget:
                    break
                fi = do_call(simplex[i])
                simplex_f.append(fi)
            else:
                # Full simplex evaluated
                # Nelder-Mead iterations
                alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                max_iter = 500

                for iteration in range(max_iter):
                    if check_deadline() or calls >= budget:
                        break

                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Check convergence
                    if max(simplex_f) - min(simplex_f) < 1e-10:
                        break

                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    fr = do_call(xr)
                    if check_deadline() or calls >= budget:
                        break

                    if fr < simplex_f[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = do_call(xe)
                        if check_deadline() or calls >= budget:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    elif fr < simplex_f[-2]:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    else:
                        # Contraction
                        if fr < simplex_f[-1]:
                            # Outside contraction
                            xc = centroid + rho * (xr - centroid)
                        else:
                            # Inside contraction
                            xc = centroid + rho * (simplex[-1] - centroid)
                        fc = do_call(xc)
                        if check_deadline() or calls >= budget:
                            break
                        if fc < min(fr, simplex_f[-1]):
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if check_deadline() or calls >= budget:
                                    break
                                simplex_f[i] = do_call(simplex[i])
                            else:
                                continue
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
