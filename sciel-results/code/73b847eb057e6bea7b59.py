"""Multi-start Nelder-Mead with Halton sequence initialization and CMA-ES refinement
for Katsuura f23. Combines deterministic global exploration (Halton) with adaptive
covariance-based local search to handle hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        start_time = time.time()
        deadline = start_time + 285  # 45s margin before 330s kill
        best_x = None
        best_f = np.inf
        all_points = []

        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError
            if calls >= budget:
                raise StopIteration
            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                x = x[:dim]
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            all_points.append(x.copy())
            return val

        # Generate Halton sequence points for deterministic diverse coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            # First few primes for bases
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(1, n + 1):
                pt = []
                for d in range(dim):
                    pt.append(halton(i, primes[d]))
                points.append(np.array(pt))
            return points

        # Phase 1: Global exploration with Halton + structured points
        rng = np.random.default_rng(42)
        
        # Generate Halton points mapped to [-5, 5]^dim
        n_halton = 30
        halton_pts = halton_sequence(n_halton, dim)
        halton_pts = np.array(halton_pts) * 10.0 - 5.0  # map [0,1]^dim to [-5,5]^dim
        
        # Add structured points
        start_points = list(halton_pts)
        start_points.append(np.zeros(dim))
        for v in [2.0, -2.0, 3.0, -3.0, 4.0, -4.0]:
            start_points.append(np.full(dim, v))
        for d in range(dim):
            for v in [2.0, -2.0, 4.0, -4.0]:
                x = np.zeros(dim)
                x[d] = v
                start_points.append(x)
        
        # Add some random points for additional diversity
        for _ in range(10):
            start_points.append(rng.uniform(-4.5, 4.5, size=dim))
        
        # Limit to a reasonable number of starts
        max_starts = min(40, len(start_points))
        start_points = start_points[:max_starts]
        
        # Evaluate each starting point with a small Nelder-Mead
        for x0 in start_points:
            if calls >= budget or time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 8, 'xatol': 1e-3, 'fatol': 1e-3}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        # Phase 2: CMA-ES-like adaptive refinement from the best point
        if best_x is not None and calls < budget and time.time() < deadline:
            mean = best_x.copy()
            sigma = 1.0
            C = np.eye(dim)
            eigvals, eigvecs = np.linalg.eigh(C)
            lambda_ = 2 * dim + 1
            no_improve = 0
            max_no_improve = 25
            
            while calls < budget and time.time() < deadline:
                if (time.time() - start_time) > deadline - 5:
                    break
                
                pop = []
                pop_f = []
                
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    z = rng.standard_normal(dim)
                    x_sample = mean + sigma * eigvecs @ (eigvals ** 0.5 * z)
                    x_sample = np.clip(x_sample, -5.0, 5.0)
                    
                    fx = safe_func(x_sample)
                    pop.append(x_sample.copy())
                    pop_f.append(fx)
                
                if len(pop) == 0:
                    break
                
                sorted_idx = np.argsort(pop_f)
                pop_sorted = [pop[i] for i in sorted_idx]
                pop_f_sorted = [pop_f[i] for i in sorted_idx]
                
                improved = False
                if pop_f_sorted[0] < best_f:
                    best_f = pop_f_sorted[0]
                    best_x = pop_sorted[0].copy()
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1
                
                # Update mean with weighted average of top half
                mu = max(1, lambda_ // 2)
                weights = np.zeros(mu)
                w_sum = 0
                for i in range(mu):
                    weights[i] = np.log(mu + 0.5) - np.log(i + 1)
                    w_sum += weights[i]
                weights /= w_sum
                
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * pop_sorted[i]
                mean = new_mean
                
                # Update covariance
                C_new = np.zeros((dim, dim))
                for i in range(mu):
                    diff = pop_sorted[i] - mean
                    C_new += weights[i] * np.outer(diff, diff)
                C = C_new
                
                # Regularize
                C = 0.99 * C + 0.01 * np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                
                # Adapt sigma
                if improved:
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 0.01, 5.0)
                
                if no_improve >= max_no_improve:
                    break

        # Phase 3: Final Nelder-Mead polish from best point
        if best_x is not None and calls < budget and time.time() < deadline:
            try:
                remaining = budget - calls
                est_maxiter = max(1, min(remaining // (dim + 1), 50))
                res = opt.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': est_maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        # Ensure valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
