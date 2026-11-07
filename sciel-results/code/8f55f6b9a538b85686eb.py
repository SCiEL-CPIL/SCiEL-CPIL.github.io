"""Combined optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a (1+1)-ES phase for initial exploration, then switches to a GP
surrogate with a simple isotropic kernel and PI-style acquisition.
Includes a wall-clock deadline guard and adaptive phase management.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import exp, sqrt, log
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _init_points(self, rng, n):
        """Generate diverse initial points."""
        pts = [np.zeros(self.dim)]
        for _ in range(n - 1):
            r = rng.random()
            if r < 0.25:
                pts.append(rng.uniform(-5, 5, self.dim))
            elif r < 0.50:
                pts.append(rng.integers(-3, 4, self.dim).astype(float))
            elif r < 0.75:
                pts.append(rng.uniform(-2, 2, self.dim))
            else:
                pts.append(rng.integers(-1, 2, self.dim).astype(float))
        return pts

    def _N_pdf(self, z):
        return exp(-0.5 * z * z) / sqrt(2.0 * np.pi)

    def _N_cdf(self, z):
        """Normal CDF approximation."""
        if z >= 0:
            t = 1.0 / (1.0 + 0.2316419 * z)
            poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
            return 1.0 - poly * exp(-0.5 * z * z) / sqrt(2.0 * np.pi)
        else:
            return 1.0 - self._N_cdf(-z)

    def _fit_gp_simple(self, X, y):
        """Fit a simple GP with isotropic kernel. Returns (l2, alpha, beta)."""
        n, d = X.shape
        if n == 1:
            return 1.0, np.ones(1), float(y[0])
        
        # Estimate length-scale from data spread
        d0 = X - X[0]
        d2_all = (d0 * d0).sum(axis=1)
        d2_pos = d2_all[d2_all > 0]
        if len(d2_pos) > 0:
            l = float(np.clip(np.sqrt(2.0 * np.mean(d2_pos)), 0.1, 10.0))
        else:
            l = 1.0
        l2 = l * l
        
        # Build kernel matrix
        d2 = np.zeros((n, n))
        for i in range(n):
            diff = X[i] - X
            d2[i] = (diff * diff).sum(axis=1)
        K = np.exp(-0.5 * d2 / l2)
        K[np.arange(n), np.arange(n)] += 1e-6
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K[np.arange(n), np.arange(n)] += 1e-4
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K[np.arange(n), np.arange(n)] += 1e-3
                L = np.linalg.cholesky(K)
        
        alpha = np.linalg.solve(L, np.ones(n))
        beta = float(np.mean(y))
        return l2, alpha, beta

    def _acq_pi(self, X, y, l2, alpha, beta, x):
        """Probability-of-improvement acquisition function."""
        n = X.shape[0]
        dx = x - X
        d2 = (dx * dx).sum(axis=1) / l2
        K = np.exp(-0.5 * d2)
        
        mu = beta + (K - 0.5 * d2) @ alpha
        v = (K - 0.5 * d2) * alpha
        var = float(v.sum())
        sd = sqrt(max(var, 1e-12))
        
        if sd < 1e-9:
            return -mu
        
        z = (beta - mu) / sd
        phi = self._N_pdf(z)
        Phi = self._N_cdf(z)
        c = phi * max(z, 0.0) + (beta - mu) * (1.0 - Phi)
        c = max(c, 0.0)
        return -mu - c

    def _acq_pi_grad(self, X, y, l2, alpha, beta, x):
        """PI acquisition with gradient."""
        n = X.shape[0]
        dx = x - X
        d2 = (dx * dx).sum(axis=1) / l2
        K = np.exp(-0.5 * d2)
        
        mu = beta + (K - 0.5 * d2) @ alpha
        v = (K - 0.5 * d2) * alpha
        var = float(v.sum())
        sd = sqrt(max(var, 1e-12))
        
        # Gradient of mu: dmu/dx = -sum_i alpha_i * K_i * dx_i / l2
        g_mu = -(K * dx) / l2
        g_mu = g_mu.sum(axis=0)
        
        if sd < 1e-9:
            val = -mu
            return val, -g_mu
        
        z = (beta - mu) / sd
        phi = self._N_pdf(z)
        Phi = self._N_cdf(z)
        c = phi * max(z, 0.0) + (beta - mu) * (1.0 - Phi)
        c = max(c, 0.0)
        val = -mu - c
        
        # Gradient of acquisition: dval/dx = -g_mu - (c/sd) * g_mu
        g_acq = -g_mu - (c / sd) * g_mu
        return val, g_acq

    def _optimize_acq(self, X, y, l2, alpha, beta, rng, best_x):
        """Optimize acquisition function with multi-start L-BFGS-B."""
        best_val = -np.inf
        best_xq = None
        
        # Multi-start candidates
        cands = [np.zeros(self.dim)]
        cands.append(np.clip(best_x, -5, 5))
        for _ in range(8):
            cands.append(rng.uniform(-5, 5, self.dim))
        
        for x0 in cands:
            x0 = np.clip(x0, -5, 5)
            try:
                res = minimize(
                    self._acq_pi_grad,
                    x0,
                    args=(X, y, l2, alpha, beta),
                    method="L-BFGS-B",
                    bounds=[(-5, 5)] * self.dim,
                    jac=True,
                    options={"maxiter": 20, "maxls": 15},
                )
                xv = np.clip(res.x, -5, 5)
                val, _ = self._acq_pi_grad(X, y, l2, alpha, beta, xv)
                if val > best_val:
                    best_val = val
                    best_xq = xv
            except Exception:
                continue
        
        if best_xq is None:
            best_xq = cands[0]
        return best_xq

    def _nelder_mead(self, x0, func, budget, deadline, rng):
        """Nelder-Mead simplex local refinement."""
        n = self.dim
        # Initialize simplex
        simplex = [np.clip(x0, -5, 5).copy()]
        f_vals = [float(func(np.clip(x0, -5, 5)))]
        
        # Add perturbed vertices
        for i in range(n):
            v = simplex[0].copy()
            delta = 0.5 if v[i] >= 0 else -0.5
            v[i] += delta
            v = np.clip(v, -5, 5)
            simplex.append(v)
            f_vals.append(float(func(v)))
        
        # Sort by function value
        idx = np.argsort(f_vals)
        simplex = [simplex[i] for i in idx]
        f_vals = [f_vals[i] for i in idx]
        
        calls = 1 + n
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 200
        min_improve = 1e-8
        
        for it in range(max_iter):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = float(func(x_r))
            calls += 1
            
            if f_r < f_vals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = float(func(x_e))
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = float(func(x_c))
                calls += 1
                
                if f_c < f_vals[-1]:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = float(func(simplex[i]))
                        calls += 1
                        if calls >= budget or time.time() >= deadline:
                            break
                    if calls >= budget or time.time() >= deadline:
                        break
            
            # Check convergence
            if it > 0 and abs(f_vals[0] - f_vals[-1]) < min_improve:
                break
            
            # Re-sort
            idx = np.argsort(f_vals)
            simplex = [simplex[i] for i in idx]
            f_vals = [f_vals[i] for i in idx]
        
        # Return best point
        best_idx = np.argmin(f_vals)
        return simplex[best_idx], f_vals[best_idx], calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(0)
        
        # Phase 1: Diverse initial sampling
        init_pts = self._init_points(rng, 10)
        xs, ys = [], []
        best = (None, np.inf)
        
        def sample(x):
            nonlocal best
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            xs.append(x)
            ys.append(f)
            if f < best[1]:
                best = (x.copy(), f)
            return f
        
        for x in init_pts:
            if time.time() >= deadline:
                break
            sample(x)
        
        # Phase 2: Nelder-Mead local refinement from best point
        nm_budget = 100
        nm_deadline = deadline - 10.0
        try:
            nm_x, nm_f, nm_calls = self._nelder_mead(
                best[0], func, nm_budget, nm_deadline, rng
            )
            # Record all the NM evaluations
            # Note: NM already called func internally, so we just update best
            if nm_f < best[1]:
                best = (nm_x.copy(), nm_f)
        except Exception:
            pass
        
        # Phase 3: GP model-and-optimize with remaining budget
        n_min_gp = 10
        remaining = self.budget - len(xs) - nm_calls if 'nm_calls' in locals() else self.budget - len(xs)
        
        for _ in range(max(0, remaining)):
            if time.time() >= deadline:
                break
            
            X = np.array(xs)
            y = np.array(ys, dtype=float)
            n = len(xs)
            
            if n < n_min_gp:
                # Not enough points: explore around best
                xq = np.clip(best[0] + 1.0 * rng.standard_normal(self.dim), -5, 5)
            else:
                # Try GP
                try:
                    l2, alpha, beta = self._fit_gp_simple(X, y)
                    xq = self._optimize_acq(X, y, l2, alpha, beta, rng, best[0])
                except Exception:
                    # Fall back to local exploration
                    xq = np.clip(best[0] + 0.8 * rng.standard_normal(self.dim), -5, 5)
            
            sample(xq)
        
        return best
# EVOLVE-BLOCK-END
