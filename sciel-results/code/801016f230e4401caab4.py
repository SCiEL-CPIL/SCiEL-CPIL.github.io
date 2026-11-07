"""Derivative-free optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a three-phase approach:
1. Dense Sobol quasi-random sampling for global exploration
2. Multi-start Powell's method for derivative-free local refinement
3. CMA-ES polish for final convergence

Powell's method is particularly effective for the Rosenbrock valley structure
without needing gradient information, making it robust to the Griewank oscillations.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Sobol global exploration + multi-start Powell's + CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_points(self, n_points, n, seed=0):
        """Generate quasi-random Sobol points in [0,1]^n."""
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=n, scramble=True, seed=seed)
            raw = sampler.random(n_points)
        except Exception:
            raw = np.zeros((n_points, n))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for j in range(n):
                p = primes[j % len(primes)]
                for i in range(n_points):
                    f = 1.0
                    num = i + 1
                    while num > 0:
                        num, rem = divmod(num, p)
                        f /= p
                        raw[i, j] += rem * f
                    raw[i, j] = 1.0 - raw[i, j]
        return raw

    def _cmaes(self, x0, f0, func, budget, deadline, sigma0=1.0, pop=None):
        """CMA-ES optimizer."""
        n = self.dim
        if pop is None:
            pop = max(4, int(4 + 3 * np.log(n)))
        mu = max(2, pop // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs
        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        c = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        Csqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        best_x = c.copy()
        best_f = f0
        calls = 0

        it = 0
        while calls < budget and time.time() < deadline:
            if time.time() > deadline:
                break
            Y = np.zeros((pop, n))
            F = np.empty(pop)
            for i in range(pop):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(n)
                y = Csqrt @ z
                x = c + sigma * y
                x = np.clip(x, -5.0, 5.0)
                F[i] = float(func(x))
                Y[i] = y
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
            if calls >= budget:
                break
            idx = np.argsort(F)
            Ys = Y[idx]
            ymean = (w[:, None] * Ys[:mu]).sum(axis=0)
            d = Csqrt @ ymean
            c = c + cs * d
            c = np.clip(c, -5.0, 5.0)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * ymean
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                    (1.4 + 2 / (n + 1)) * chiN:
                p_c = (1 - cs) * p_c + cs * d
                hsig = 1.0
            else:
                hsig = 0.0
            delta_sig = (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** ds)
            C = ((1 - c1 - cmu + c1 * hsig - cmu * hsig) * C
                 + c1 * np.outer(d, d)
                 + cmu * (w[:, None] * (Ys[:mu] @ Csqrt)).sum(axis=0)[:, None]
                 * (Ys[:mu] @ Csqrt))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-4, 10.0))
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                Csqrt = (evecs * np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                Csqrt = np.eye(n)
            it += 1
            if it > 5 and np.allclose(C, np.eye(n)) and sigma < 1e-3:
                break
        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        self._f = func

        best_x = None
        best_f = np.inf
        calls = 0
        evaluated_points = []
        evaluated_values = []

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            evaluated_points.append(x.copy())
            evaluated_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ============================================================
        # PHASE 1: Dense Sobol quasi-random global exploration
        # ============================================================
        sobol_budget = int(self.budget * 0.35)
        sobol_budget = min(sobol_budget, self.budget - 50)

        sobol_raw = self._sobol_points(sobol_budget, n, seed=42)
        sobol_points = sobol_raw * 10.0 - 5.0

        for i in range(sobol_budget):
            if time.time() > deadline - 120:
                break
            if calls >= self.budget:
                break
            evaluate(sobol_points[i])

        # ============================================================
        # PHASE 2: Multi-start Powell's method local refinement
        # ============================================================
        local_budget = self.budget - calls
        if local_budget > 20 and time.time() < deadline - 30:
            # Generate diverse restart points from evaluated points
            if len(evaluated_points) > 0:
                # Sort by function value
                sorted_idx = np.argsort(evaluated_values)
                
                # Take top-K diverse points
                restart_points = []
                used_idx = set()
                
                # First, add the best point
                restart_points.append(evaluated_points[sorted_idx[0]].copy())
                used_idx.add(sorted_idx[0])
                
                # Add diverse points from top performers
                for idx in sorted_idx[1:]:
                    if len(restart_points) >= 15:
                        break
                    point = evaluated_points[idx]
                    # Check diversity
                    is_diverse = True
                    for rp in restart_points:
                        if np.linalg.norm(point - rp) < 1.0:
                            is_diverse = False
                            break
                    if is_diverse:
                        restart_points.append(point.copy())
                        used_idx.add(idx)
                
                # Add some random diverse points
                rng = np.random.default_rng(123)
                for _ in range(8):
                    rp = rng.uniform(-5.0, 5.0, n)
                    restart_points.append(rp)
                
                # Deduplicate
                deduped = []
                for rp in restart_points:
                    rp = np.clip(rp, -5.0, 5.0)
                    is_dup = False
                    for dp in deduped:
                        if np.linalg.norm(rp - dp) < 0.3:
                            is_dup = True
                            break
                    if not is_dup:
                        deduped.append(rp)
                restart_points = deduped
                
                # Sort by proximity to best (closest first)
                if best_x is not None:
                    restart_points.sort(key=lambda p: np.linalg.norm(p - best_x))
                
                # Run Powell's method from each restart point
                n_restarts = min(len(restart_points), 12)
                
                for ri in range(n_restarts):
                    if calls >= self.budget or time.time() > deadline - 20:
                        break
                    
                    x0 = restart_points[ri].copy()
                    x0 = np.clip(x0, -5.0, 5.0)
                    
                    # Budget allocation: earlier restarts get more
                    frac = max(0.2, 1.0 - ri * 0.1)
                    restart_budget = max(15, int(local_budget * frac * 0.25))
                    restart_budget = min(restart_budget, self.budget - calls)
                    
                    if restart_budget < 10:
                        break
                    
                    try:
                        def objective_powell(x):
                            x = np.clip(x, -5.0, 5.0)
                            return float(func(x))
                        
                        res = minimize(
                            objective_powell,
                            x0,
                            method='Powell',
                            bounds=[(-5.0, 5.0)] * n,
                            options={
                                'maxiter': max(10, restart_budget // 3),
                                'maxfev': restart_budget,
                                'xtol': 1e-8,
                                'ftol': 1e-10,
                                'disp': False,
                            }
                        )
                        
                        if res.fun < best_f:
                            best_f = float(res.fun)
                            best_x = np.clip(res.x, -5.0, 5.0).copy()
                        # Track the endpoint
                        rx = np.clip(res.x, -5.0, 5.0)
                        rf = float(res.fun)
                        evaluated_points.append(rx)
                        evaluated_values.append(rf)
                        if rf < best_f:
                            best_f = rf
                            best_x = rx.copy()
                            
                    except Exception:
                        pass

        # ============================================================
        # PHASE 3: CMA-ES polish if budget remains
        # ============================================================
        remaining = self.budget - calls
        if remaining > 30 and time.time() < deadline - 15 and best_x is not None:
            try:
                x3, f3, c3 = self._cmaes(
                    best_x, best_f, func, remaining, deadline,
                    sigma0=0.3, pop=6
                )
                if f3 < best_f:
                    best_f = f3
                    best_x = x3.copy()
            except Exception:
                pass

        # Final safety
        if best_x is None:
            best_x = np.zeros(n)
            best_f = float(func(best_x))

        return best_x, best_f

# EVOLVE-BLOCK-END
