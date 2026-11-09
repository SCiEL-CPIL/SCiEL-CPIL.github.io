"""
Optimizer for Bent Cigar (f12) using CMA-ES with anisotropy-aware restarts.
CMA-ES adapts its covariance to the narrow valley, outperforming isotropic
methods on this highly anisotropic function.
"""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_x = None
        best_fx = np.inf

        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            calls_used += 1
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Sobol global seeding ---
        n_global = 16
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global) * 10.0 - 5.0
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break

        if best_x is None:
            wrapped_obj(np.zeros(dim))

        # --- Phase 2: CMA-ES with adaptive restarts ---
        lam = 10  # population size
        mu = lam // 2  # number of parents

        # Weights (standard CMA-ES)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = (2.0 - mueff / (dim + 2.0) + 2.0 * w[0]) / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0)**2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        # Step size
        sigma0 = 0.5

        def cma_es(x_center, sigma, max_calls, min_sigma=1e-10):
            nonlocal calls_used, best_x, best_fx
            x_mean = x_center.copy()
            sig = sigma
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chiN = np.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))

            # Initial evaluation of center
            try:
                fc = wrapped_obj(x_mean.copy())
            except Exception:
                return

            stagnation = 0
            max_stagnation = 30  # generations without improvement before restart check

            while calls_used < budget and time.time() < deadline:
                if sig < min_sigma:
                    break

                # Generate population
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 0.0))
                Y = np.random.randn(lam, dim) * sig
                # Transform
                P = x_mean + Y @ (B * D).T

                # Evaluate population
                F = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_used >= budget or time.time() >= deadline:
                        break
                    try:
                        F[i] = wrapped_obj(P[i])
                    except Exception:
                        break
                else:
                    # All evaluated
                    pass
                if calls_used >= budget or time.time() >= deadline:
                    break

                # Sort
                order = np.argsort(F)
                P = P[order]
                F = F[order]

                # Check for improvement
                if F[0] < best_fx:
                    stagnation = 0
                else:
                    stagnation += 1

                # Stop if stagnation is high and sigma is small
                if stagnation > max_stagnation and sig < 0.01:
                    break

                # Weighted mean
                x_new_mean = np.sum(w[:, None] * P[:mu], axis=0)

                # Update C (rank-mu update)
                y = (x_new_mean - x_mean) / sig
                C = (1.0 - c1 - cmu) * C + c1 * np.outer(y, y) + cmu * np.sum(w[:mu, None] * np.outer(P[:mu] - x_mean, P[:mu] - x_mean) / (sig**2), axis=0)
                # Symmetrize
                C = (C + C.T) / 2.0

                # Update pc
                pc = (1.0 - cc) * pc + cc * (1.0 if np.linalg.norm(y) / np.sqrt(1.0 - (1.0 - cc)**2) / sig < (1.4 + cc) * chiN else 0.0) * y

                # Update ps
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (1.0 if np.linalg.norm(y) / np.sqrt(1.0 - (1.0 - cc)**2) / sig < (1.4 + cc) * chiN else 0.0) * y

                # Sigma update
                sig = sig * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1.0))
                sig = np.clip(sig, 1e-12, 10.0)

                # Update mean
                x_mean = x_new_mean

            return x_mean, sig

        # Main CMA-ES loop with restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if calls_used >= budget or time.time() >= deadline:
                break

            x_center = best_x.copy()
            # Initial sigma: based on distance to boundary or fixed
            if restart == 0:
                sigma = 0.5
            else:
                # Restart with smaller sigma, centered on best
                sigma = 0.3 * (0.7 ** restart)

            try:
                result = cma_es(x_center, sigma, budget - calls_used)
            except Exception:
                break

            if time.time() >= deadline or calls_used >= budget:
                break

            # If no improvement in this restart, break
            if best_fx == np.inf:
                break

        return best_x, best_fx
