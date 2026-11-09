"""CMA-ES based optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer with LHS initialization and deadline guard.
    Uses covariance matrix adaptation to handle rotated landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        total_budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        def check_deadline():
            return time.time() - start_time > 325.0  # Safety margin

        # --- Phase 1: LHS Initialization ---
        n_lhs = min(10, total_budget // 10)
        if n_lhs < 5:
            n_lhs = 5
            
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5.0 + 10.0 * (perm + rng.random(n_lhs)) / n_lhs
        
        # Add center and a few random points
        candidates = list(lhs_points)
        candidates.append(np.zeros(dim))
        for _ in range(3):
            candidates.append(rng.uniform(-5, 5, dim))

        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for c in candidates:
            if check_deadline() or calls_made >= total_budget:
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None or calls_made >= total_budget:
            return best_x, best_f

        # --- Phase 2: CMA-ES ---
        # Parameters
        lambda_ = 4 * int(1 + 3 * np.log(dim))  # Population size
        if lambda_ < 4:
            lambda_ = 4
            
        # Initial sigma based on spread of initial points
        if len(candidates) > 1:
            # Estimate initial sigma from LHS spread
            std_est = np.std([np.linalg.norm(c - best_x) for c in candidates if c is not None])
            sigma = max(0.5, min(2.0, std_est))
        else:
            sigma = 1.0

        # CMA-ES state
        mean = best_x.copy()
        C = np.eye(dim)
        step_size = sigma
        
        # Update rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Precompute Cholesky of C
        L = np.linalg.cholesky(C)
        
        # Restart logic
        restart_threshold = 100
        no_improve = 0
        stagnation_counter = 0

        while calls_made < total_budget and not check_deadline():
            # Generate population
            pops = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Transform to correlated normal
                x_i = mean + step_size * L @ z
                pops[i] = x_i
            
            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls_made >= total_budget or check_deadline():
                    break
                fvals[i] = func(pops[i])
                calls_made += 1
            
            if calls_made >= total_budget or check_deadline():
                break
                
            # Sort by fitness
            order = np.argsort(fvals)
            pops = pops[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pops[0].copy()
                no_improve = 0
            else:
                no_improve += 1
                
            # Restart if stuck
            if no_improve > restart_threshold:
                # Perturb mean
                mean = best_x + 0.5 * rng.standard_normal(dim)
                mean = np.clip(mean, -5, 5)
                C = np.eye(dim)
                L = np.linalg.cholesky(C)
                step_size = max(0.5, step_size * 0.5)
                no_improve = 0
                restart_threshold = max(50, restart_threshold // 2)
                # Re-evaluate to ensure valid state? No, just continue loop.
                # We need to make sure we don't exceed budget.
                continue

            # Update mean
            y_w = np.sum(weights[:, np.newaxis] * pops[:mu], axis=0)
            mean = mean + y_w
            
            # Update covariance
            # p_sigma and p_c
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(L) * (L.T @ y_w)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1))) < (1 + 2 / (dim + 1)) * np.trace(C) / dim else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / np.linalg.norm(L) * (L.T @ y_w)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu / np.linalg.norm(L)**2 * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pops[:mu] - mean, pops[:mu] - mean), axis=0)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                L = np.linalg.cholesky(C + 1e-9 * np.eye(dim))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(np.eye(dim) + 1e-9 * C)
                C = np.eye(dim)

            # Update step size
            step_size *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1))) - np.trace(C) / dim))
            step_size = np.clip(step_size, 1e-6, 10.0)

            # Check for convergence/stagnation
            if np.linalg.norm(p_c) < np.sqrt(1 - (1 - c_c)**(2 * (calls_made + 1))) * (1 + 2 / (dim + 1)) * np.trace(C) / dim:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            if stagnation_counter > 200:
                break

        # --- Phase 3: Local Polish (Nelder-Mead) ---
        if best_x is not None and not check_deadline() and calls_made < total_budget:
            nm_budget = min(total_budget - calls_made, 50)
            if nm_budget > dim + 1:
                x_nm, f_nm, calls_nm = self._nm_polish(func, best_x, dim, nm_budget, rng)
                calls_made += calls_nm
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm

        return best_x, best_f

    def _nm_polish(self, func, x0, dim, budget, rng, lo=-5.0, hi=5.0):
        """Nelder-Mead with box projection; returns (best_x, best_f, calls_used)."""
        x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
        f0 = func(x0)
        calls = 1
        # Initial simplex
        xk = [x0.copy()]
        for j in range(dim):
            xi = x0.copy()
            step = 0.1 if x0[j] < 0 else -0.1
            xi[j] += step
            xi = np.clip(xi, lo, hi)
            xk.append(xi)
        fx = [f0] + [func(xi) for xi in xk[1:]]
        calls += dim

        for _ in range(budget - calls):
            if calls >= budget:
                break
            order = np.argsort(fx)
            xk = [xk[i] for i in order]
            fx = [fx[i] for i in order]
            if calls >= budget:
                break

            # centroid of all but worst
            centroid = np.mean(xk[:-1], axis=0)
            worst = xk[-1]
            # reflection
            xr = np.clip(centroid + 1.0 * (centroid - worst), lo, hi)
            fr = func(xr)
            calls += 1
            if fr < fx[-2]:
                # expansion
                xe = np.clip(centroid + 2.0 * (xr - centroid), lo, hi)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    xk[-1], fx[-1] = xe, fe
                else:
                    xk[-1], fx[-1] = xr, fr
            elif fr <= fx[0]:
                xk[-1], fx[-1] = xr, fr
            else:
                # contraction
                xc = np.clip(centroid + 0.5 * (worst - centroid), lo, hi)
                fc = func(xc)
                calls += 1
                if fc < fx[-1]:
                    xk[-1], fx[-1] = xc, fc
                else:
                    # shrink toward best
                    for i in range(1, dim + 1):
                        xk[i] = np.clip(xk[0] + 0.5 * (xk[i] - xk[0]), lo, hi)
                        fx[i] = func(xk[i])
                        calls += 1
                        if calls >= budget:
                            break
            if calls >= budget:
                break

        order = np.argsort(fx)
        best_i = order[0]
        return xk[best_i], fx[best_i], calls
# EVOLVE-BLOCK-END
