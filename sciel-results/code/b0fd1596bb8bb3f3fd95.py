"""Combined surrogate-model + CMA-ES optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: surrogate-guided global search + CMA-ES local refinement.
    
    Phase 1: Quadratic surrogate model with L-BFGS acquisition for basin-finding.
    Phase 2: CMA-ES from the best point found for local refinement.
    Phase 3: Additional CMA-ES runs from other promising points if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # History of samples for the local quadratic model
        history = []
        MAX_HISTORY = 40

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            history.append((x.copy(), fx))
            if len(history) > MAX_HISTORY:
                history.pop(0)
            return fx

        # ---------- Phase 0: Multi-start to find both Lunacek basins ----------
        init_points = [
            np.zeros(n),
            rng.uniform(-1, 1, n),
            rng.uniform(-2, 2, n),
            np.ones(n) * 2.5,
            np.ones(n) * -2.5,
            rng.uniform(-4, 4, n),
            rng.uniform(-4, 4, n),
            np.ones(n) * 1.5,
            np.ones(n) * -1.5,
        ]
        for xp in init_points:
            if calls_made >= total_budget:
                break
            safe_func(xp)

        # ---------- Quadratic local model ----------
        def _features(x):
            """Build feature vector for quadratic model: [1, x, x_i*x_j (i<=j)]"""
            x2 = np.zeros(n * (n + 1) // 2)
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    x2[idx] = x[i] * x[j]
                    idx += 1
            return np.concatenate(([1.0], x, x2))

        def _fit_quadratic():
            """Fit quadratic model from history. Returns (c, g, H) or None."""
            if len(history) < n + 2:
                return None
            pts = [h[0] for h in history]
            fs = np.array([h[1] for h in history])
            A = np.array([_features(p) for p in pts])
            try:
                coeff, residuals, rank, sv = np.linalg.lstsq(A, fs, rcond=None)
            except np.linalg.LinAlgError:
                return None
            c = coeff[0]
            g = coeff[1:1 + n].copy()
            qflat = coeff[1 + n:].copy()
            H = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    H[i, j] = qflat[idx]
                    if i != j:
                        H[j, i] = qflat[idx]
                    idx += 1
            H = 0.5 * (H + H.T)
            return c, g, H

        def _quadratic_min(c, g, H):
            """Find minimizer of c + g^T x + 0.5 x^T H x, regularized."""
            lam = 1e-4
            H_reg = H + lam * np.eye(n)
            try:
                z = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                z = np.linalg.pinv(H_reg) @ (-g)
            return np.clip(z, -5, 5)

        # ---------- Phase 1: Surrogate-model guided search ----------
        x_current = best_x.copy()
        fx_current = best_f
        sigma = 0.8
        stagnation_count = 0
        max_stagnation = 25
        window = []
        window_size = 10

        # Track points that look promising for later CMA-ES restarts
        promising_points = [(best_x.copy(), best_f)]

        while calls_made < total_budget and time.time() < deadline - 15:
            # Fit a local quadratic model
            model = _fit_quadratic()

            # Build candidate set
            candidates = []

            # Candidate 1: surrogate minimum
            if model is not None:
                c, g, H = model
                try:
                    def model_f(xv):
                        xv = np.asarray(xv)
                        return c + g @ xv + 0.5 * (xv @ (H @ xv))

                    def model_grad(xv):
                        xv = np.asarray(xv)
                        return g + H @ xv

                    res = minimize(model_f, x_current.copy(), jac=model_grad,
                                   method='L-BFGS-B',
                                   bounds=[(-5.0, 5.0)] * n,
                                   options={'maxiter': 20, 'gtol': 1e-4})
                    x_prop = np.clip(res.x, -5, 5)
                    f_prop = float(res.fun)
                    if f_prop < fx_current - 1e-9 or np.linalg.norm(x_prop - x_current) > 1e-6:
                        candidates.append((x_prop, f_prop))
                except Exception:
                    x_opt = _quadratic_min(c, g, H)
                    if x_opt is not None:
                        candidates.append((x_opt, model(c, g, H, x_opt)))

            # Candidate 2: small random perturbation
            x_rand = x_current + sigma * rng.standard_normal(n)
            x_rand = np.clip(x_rand, -5, 5)
            candidates.append((x_rand, None))

            # Candidate 3: medium random perturbation
            x_rand2 = x_current + (sigma * 2.0) * rng.standard_normal(n)
            x_rand2 = np.clip(x_rand2, -5, 5)
            candidates.append((x_rand2, None))

            # Candidate 4: large jump to explore other basins
            if rng.random() < 0.1:
                x_jump = rng.uniform(-4, 4, n)
                candidates.append((x_jump, None))

            # Pick candidate
            scored = [c for c in candidates if c[1] is not None]
            if scored:
                scored.sort(key=lambda t: t[1])
                if rng.random() < 0.7:
                    x_next = scored[0][0]
                else:
                    x_next = candidates[rng.integers(len(candidates))][0]
            else:
                x_next = candidates[rng.integers(len(candidates))][0]

            # Evaluate
            f_next = safe_func(x_next)

            # Adapt step size
            if f_next < fx_current - 1e-12:
                x_current = x_next
                fx_current = f_next
                window.append(1)
                stagnation_count = 0
                # Track promising points
                if len(promising_points) < 10 or f_next < promising_points[-1][1]:
                    promising_points.append((x_next.copy(), f_next))
            else:
                window.append(0)
                stagnation_count += 1

            if len(window) >= window_size:
                success_rate = np.mean(window)
                if success_rate > 0.5:
                    sigma *= 0.7  # shrink step
                elif success_rate < 0.2:
                    sigma *= 1.5  # grow step
                window = window[-window_size:]

            sigma = np.clip(sigma, 0.05, 4.0)

            # Restart if stuck too long
            if stagnation_count >= max_stagnation:
                # Try a fresh random start
                x_restart = rng.uniform(-4, 4, n)
                f_restart = safe_func(x_restart)
                if f_restart < fx_current:
                    x_current = x_restart
                    fx_current = f_restart
                else:
                    x_current = x_restart
                    fx_current = f_restart
                sigma = 1.0
                stagnation_count = 0
                window = []

        # ---------- Phase 2: CMA-ES from best point ----------
        if best_x is not None and calls_made < total_budget and time.time() < deadline - 10:
            best_x, best_f = self._cmaes_run(func, best_x.copy(), 0.5, calls_made, total_budget, deadline, rng)

        # ---------- Phase 3: CMA-ES from other promising points ----------
        if calls_made < total_budget and time.time() < deadline - 10:
            # Sort promising points by fitness, try the best ones
            promising_points.sort(key=lambda t: t[1])
            tried = 0
            for px, pf in promising_points:
                if calls_made >= total_budget or time.time() >= deadline - 5:
                    break
                if tried >= 3:
                    break
                # Only try if this point is different from current best
                if np.linalg.norm(px - best_x) > 0.5:
                    _, _ = self._cmaes_run(func, px.copy(), 0.3, calls_made, total_budget, deadline, rng)
                    tried += 1

        return best_x, best_f

    def _cmaes_run(self, func, center, sigma, calls_made, total_budget, deadline, rng):
        """Run CMA-ES from given center and sigma. Returns (best_x, best_f)."""
        n = self.dim
        
        # Population size
        lam = int(4 + 3 * np.log(n))
        mu = lam // 2
        
        # CMA-ES parameters
        cc = (4 + n / (n + 4)) / ((n + 1.3) ** 2 + 4)
        cs = (n + 1.7) / (n + 9 + mu)
        c1 = 2 / ((n + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1 / mu) / ((n + 2) ** 2 + mu))
        
        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Initialize
        x_mean = center.copy()
        sigma_local = sigma
        C = np.eye(n)
        B = np.eye(n)
        d = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Track best
        run_best_x = x_mean.copy()
        run_best_f = np.inf
        
        # Evaluate initial point
        x_eval = np.clip(x_mean, -5, 5)
        f0 = float(func(x_eval))
        calls_made += 1
        if f0 < run_best_f:
            run_best_f = f0
            run_best_x = x_eval.copy()
        
        min_sigma = 1e-8
        
        while calls_made < total_budget and time.time() < deadline - 5:
            if sigma_local < min_sigma:
                break
            
            # Generate population
            population = []
            for i in range(lam):
                z = rng.standard_normal(n)
                x_new = x_mean + sigma_local * (B @ (d * z))
                x_new = np.clip(x_new, -5, 5)
                population.append(x_new)
            
            # Evaluate population
            fvals = []
            for x in population:
                if calls_made >= total_budget or time.time() >= deadline - 5:
                    break
                f = float(func(x))
                calls_made += 1
                fvals.append(f)
                if f < run_best_f:
                    run_best_f = f
                    run_best_x = x.copy()
            
            if len(fvals) < lam:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals)
            x_sorted = [population[i] for i in indices]
            f_sorted = [fvals[i] for i in indices]
            
            # Check convergence
            if np.std(f_sorted[:mu]) < 1e-10 * (1 + abs(f_sorted[0])):
                break
            
            # Weighted mean
            x_new_mean = np.sum([weights[i] * x_sorted[i] for i in range(mu)], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma_local * (B @ (d ** -1) @ (x_new_mean - x_mean))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            
            sigma_local *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / chi_n - 1))
            sigma_local = max(sigma_local, min_sigma)
            
            # Update covariance matrix
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma_local
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * np.outer(pc, pc) + \
                cmu * np.sum([weights[i] * np.outer(x_sorted[i] - x_mean, x_sorted[i] - x_mean) for i in range(mu)], axis=0)
            
            # Update eigen decomposition periodically
            if calls_made % (2 * n) == 0 or np.max(np.abs(C - C.T)) > 1e-10:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    B = eigvecs
                    d = np.sqrt(eigvals)
                except:
                    pass
            
            # Update mean
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
        
        return run_best_x, run_best_f
# EVOLVE-BLOCK-END
