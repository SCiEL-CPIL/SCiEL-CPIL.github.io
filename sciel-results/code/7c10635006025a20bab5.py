import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0
        evaluated_points = []

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64).reshape(dim)
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_made += 1
            evaluated_points.append((x.copy(), val))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(777)

        # Phase 1: Space-filling exploration with Halton + random
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_phase1 = 40
        phase1_points = []
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in range(1, n_phase1 + 1):
            point = []
            for d in range(dim):
                val = halton(i, primes[d])
                point.append(-5 + 10 * val)
            phase1_points.append(np.array(point))

        # Add some random points
        for i in range(20):
            phase1_points.append(rng.uniform(-5, 5, size=dim))

        # Add points near likely Katsuura minima (x near 1 for polynomial part)
        for i in range(10):
            x = np.ones(dim) + rng.normal(0, 0.5, size=dim)
            phase1_points.append(np.clip(x, -5, 5))

        # Add points near cosine minima (multiples of pi)
        for i in range(10):
            x = rng.uniform(-5, 5, size=dim)
            x = np.round(x / np.pi) * np.pi + rng.normal(0, 0.1, size=dim)
            phase1_points.append(np.clip(x, -5, 5))

        rng.shuffle(phase1_points)

        for x in phase1_points:
            if calls_made >= budget or time.time() > deadline:
                break
            safe_call(x)

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start CMA-ES
        evaluated_points.sort(key=lambda item: item[1])
        top_k = min(6, len(evaluated_points))
        remaining = budget - calls_made

        for i in range(top_k):
            if calls_made >= budget or time.time() > deadline:
                break
            x0 = evaluated_points[i][0].copy()
            runs_remaining = top_k - i
            maxfev = min(80, max(25, remaining // runs_remaining))
            if maxfev < 20:
                break

            # Manual CMA-ES implementation for better control
            self._cma_es(func, x0, dim, maxfev, safe_call, deadline)

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Final local refinement with Nelder-Mead
        remaining = budget - calls_made
        if remaining > 15 and best_x is not None:
            try:
                from scipy.optimize import minimize
                def nm_func(x):
                    if calls_made >= budget or time.time() > deadline:
                        return best_f
                    return safe_call(x)

                minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'maxfev': min(remaining, 40),
                        'xatol': 1e-6,
                        'fatol': 1e-6
                    }
                )
            except Exception:
                pass

        return best_x, best_f

    def _cma_es(self, func, x0, dim, maxfev, safe_call, deadline):
        """Simple CMA-ES implementation."""
        rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
        x = x0.copy().astype(np.float64)
        sigma = 0.5
        best_x = x.copy()
        best_f = safe_call(x)

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        step_size = sigma

        fevals = 1
        while fevals < maxfev and (time.time() < deadline):
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Transform using B and D
                y = B @ (D * z)
                x_offspring = x + step_size * y
                offspring.append(x_offspring)

            # Evaluate offspring
            offspring_f = []
            for x_off in offspring:
                f_off = safe_call(x_off)
                fevals += 1
                offspring_f.append(f_off)
                if f_off < best_f:
                    best_f = f_off
                    best_x = x_off.copy()
                if fevals >= maxfev or time.time() > deadline:
                    break

            if len(offspring_f) < lambda_:
                break

            # Sort offspring
            idx = np.argsort(offspring_f)
            y_selected = np.array([offspring[i] - x for i in idx[:mu]])
            f_selected = np.array([offspring_f[i] for i in idx[:mu]])

            # Update mean
            x = x + step_size * (weights @ y_selected)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B.T @ (D * (weights @ y_selected)))
            h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * fevals))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (D * (weights @ y_selected)))

            # Update covariance matrix
            delta_h = (1 - h_sig) * c_c * (1 - c_c)
            outer = np.outer(pc, pc)
            H = (1 + delta_h - c_c) * np.eye(dim) + delta_h * outer
            C = (1 + c_sigma * (2 - c_sigma) - c_sigma) * C if hasattr(self, 'C') else np.eye(dim)
            # Simplified: just use the rank-one update
            delta_C = c_sigma * (2 - c_sigma)
            C = (1 - delta_C) * C + delta_C * (np.outer(weights @ y_selected, weights @ y_selected) * mu_eff)

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size (CSA)
            step_size *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            step_size = np.clip(step_size, 1e-10, 10.0)

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

        return best_x, best_f
