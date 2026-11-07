import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.dim = 5
        self.budget = 500

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30

        # Phase 1: Broader random sampling with more points for better initial coverage
        n_global = 40
        for i in range(n_global):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)

        if self.best_x is None:
            return np.zeros(self.dim), 1e30

        # Phase 2: CMA-ES with adaptive population and restart strategy
        remaining = self.budget - self.call_count
        if remaining > 20:
            try:
                self._cmaes_restart(wrapped_func, self.best_x.copy(), remaining)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: Local polish with Nelder-Mead
        remaining = self.budget - self.call_count
        if remaining > 5 and time.time() < self.deadline - 1:
            try:
                self._nelder_mead_polish(wrapped_func, self.best_x.copy(), min(remaining, 60))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _cmaes_restart(self, func, x_start, max_calls):
        """CMA-ES with restarts based on stagnation detection."""
        n = self.dim
        max_restarts = 3
        calls_per_restart = max_calls // max_restarts
        
        for restart in range(max_restarts):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            
            current_max = min(calls_per_restart, self.budget - self.call_count)
            if current_max < 10:
                break
            
            # For subsequent restarts, start from the best found so far
            if restart > 0:
                x_start = self.best_x.copy()
            
            improved = self._cmaes_single(func, x_start, current_max)
            
            # If no improvement, try a different starting point
            if not improved and restart < max_restarts - 1:
                # Sample a new random point near the best
                x_start = self.best_x + np.random.randn(n) * 0.5
                x_start = np.clip(x_start, -self.bound, self.bound)

    def _cmaes_single(self, func, x_start, max_calls):
        """Single CMA-ES run. Returns True if improvement was made."""
        n = self.dim
        x = x_start.copy()
        sigma = 1.5
        
        # Population size
        lam = 20
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial state
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_cholesky = np.eye(n)
        
        calls_used = 0
        best_f_local = self.best_f
        best_x_local = x_start.copy()
        prev_best_f = best_f_local
        no_improve_count = 0
        max_no_improve = 8
        initial_best = self.best_f
        
        for iteration in range(200):
            if calls_used >= max_calls:
                break
            if time.time() > self.deadline:
                break
            if self.call_count >= self.budget:
                break
            
            # Sample candidates
            candidates = []
            fvals = []
            
            for j in range(lam):
                if calls_used + (lam - j) > max_calls:
                    break
                if time.time() > self.deadline:
                    break
                if self.call_count >= self.budget:
                    break
                
                z = np.random.randn(n)
                x_candidate = x + sigma * (C_cholesky @ z)
                x_candidate = np.clip(x_candidate, -self.bound, self.bound)
                candidates.append(x_candidate)
                
                fx = func(x_candidate)
                fvals.append(fx)
                calls_used += 1
            
            if len(candidates) < mu:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals)
            sorted_x = [candidates[i] for i in indices]
            sorted_f = [fvals[i] for i in indices]
            
            # Update best
            if sorted_f[0] < best_f_local:
                best_f_local = sorted_f[0]
                best_x_local = candidates[indices[0]].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean with weighted recombination
            w = weights[:mu]
            x_new = sum(w[i] * sorted_x[i] for i in range(mu))
            
            # Update p_sigma
            y_sigma = (x_new - x) / sigma if sigma > 1e-30 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * y_sigma
            
            # Update covariance
            rank_one = np.outer(p_c, p_c)
            y_vecs = np.array([(sorted_x[i] - x) / (sigma + 1e-30) for i in range(mu)])
            rank_mu = np.sum([w[i] * np.outer(y_vecs[i], y_vecs[i]) for i in range(mu)], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Symmetrize and factorize
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C + 1e-20 * np.eye(n))
                C_cholesky = L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C_cholesky = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x = x_new
            
            # Early stopping
            if sigma < 1e-10:
                break
            if no_improve_count >= max_no_improve:
                break
            if np.linalg.norm(x_new - x_start) < 1e-10 and np.linalg.norm(y_sigma) < 1e-8:
                break
        
        improved = best_f_local < initial_best - 1e-12
        return improved

    def _nelder_mead_polish(self, func, x_start, max_calls):
        """Nelder-Mead simplex method for local refinement."""
        n = self.dim
        x = x_start.copy()
        f0 = func(x)
        calls = 1
        
        # Build initial simplex
        simplex = [x.copy()]
        for i in range(n):
            xi = x.copy()
            step = 0.1 if abs(x[i]) > 0.1 else 0.05
            xi[i] += step
            xi = np.clip(xi, -self.bound, self.bound)
            simplex.append(xi)
            fi = func(xi)
            calls += 1
        
        # Sort simplex
        simplex.sort(key=lambda x: func(x) if False else 0)  # We'll track fvals separately
        fvals = [f0] + [func(x) for x in simplex[1:]]
        
        # Actually, let's redo this properly
        simplex = [x.copy()]
        fvals = [f0]
        for i in range(n):
            xi = x.copy()
            step = 0.1 if abs(x[i]) > 0.1 else 0.05
            xi[i] += step
            xi = np.clip(xi, -self.bound, self.bound)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
        
        # Nelder-Mead iterations
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_nm = 0.5  # shrink
        
        for iteration in range(max_calls - n - 1):
            if calls >= max_calls or time.time() > self.deadline:
                break
            if self.call_count >= self.budget:
                break
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -self.bound, self.bound)
            f_r = func(x_r)
            calls += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -self.bound, self.bound)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -self.bound, self.bound)
                f_c = func(x_c)
                calls += 1
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -self.bound, self.bound)
                        fvals[i] = func(simplex[i])
                        calls += 1
                    if calls >= max_calls or time.time() > self.deadline or self.call_count >= self.budget:
                        break
            
            # Check convergence
            f_range = max(fvals) - min(fvals)
            if f_range < 1e-12:
                break

    def _fine_polish(self, func, x_start, max_calls):
        """Very fine local search with tiny step sizes."""
        n = self.dim
        x = x_start.copy()
        f_best = func(x)
        calls = 1
        step = 0.03
        for it in range(max_calls - 1):
            if calls >= max_calls or time.time() > self.deadline:
                break
            if self.call_count >= self.budget:
                break
            improved = False
            for i in range(n):
                for sign in [1, -1]:
                    if calls >= max_calls or time.time() > self.deadline:
                        break
                    if self.call_count >= self.budget:
                        break
                    x_try = x.copy()
                    x_try[i] += sign * step
                    x_try = np.clip(x_try, -self.bound, self.bound)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_best - 1e-12:
                        x = x_try
                        f_best = f_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
