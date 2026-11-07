import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 325s.
        deadline = start_time + 325.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Storage for GP
        X_obs = []
        y_obs = []
        
        # GP Hyperparameters (will be tuned periodically)
        # Initial guesses: signal variance 10, length scale 1, noise 1e-5
        gp_sigma_f = 10.0
        gp_length_scale = 1.0
        gp_noise = 1e-5
        
        def eval_and_track(x):
            nonlocal calls, best_f, best_x, X_obs, y_obs
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            X_obs.append(x)
            y_obs.append(val)
            return val

        # Phase 1: Initial Scan with Halton Sequence
        # This ensures early progress in AOCC and provides data for GP
        def halton_sequence(index, dim):
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point[i] = r
            return point

        n_scan = min(50, budget // 10) # 50 points for scan
        for i in range(1, n_scan + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_sequence(i, dim) * 10 - 5
            x = np.clip(x, -5, 5)
            eval_and_track(x)

        # Phase 2: Bayesian Optimization with GP
        # We will loop until budget or time is exhausted
        
        def compute_lml(params, X, y):
            """Compute Log Marginal Likelihood for GP hyperparameters."""
            sigma_f, l, sigma_n = params
            if sigma_f <= 0 or l <= 0 or sigma_n <= 0:
                return -1e10
            
            n = len(X)
            # Pairwise squared distances
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=-1)
            
            # RBF Kernel
            K = (sigma_f**2) * np.exp(-0.5 * sq_dists / (l**2))
            K[np.diag_indices_from(K)] += sigma_n**2
            
            try:
                L, _ = cho_factor(K)
                alpha = cho_solve((L, _), y)
                lml = -0.5 * y @ alpha - 0.5 * np.sum(np.log(np.diag(L))**2)
                # Add a small penalty to prevent extreme values
                lml -= 1e-6 * (np.abs(sigma_f) + np.abs(l) + np.abs(sigma_n))
                return lml
            except Exception:
                return -1e10

        def update_gp_params():
            """Periodically update GP hyperparameters using Nelder-Mead."""
            nonlocal gp_sigma_f, gp_length_scale, gp_noise
            if len(X_obs) < 5:
                return
                
            X = np.array(X_obs)
            y = np.array(y_obs)
            
            # We only use the last 100 points for fitting to keep it fast and relevant to local structure
            if len(X) > 100:
                X = X[-100:]
                y = y[-100:]
                
            # Initial guess from current values
            init_params = [gp_sigma_f, gp_length_scale, gp_noise]
            
            try:
                res = minimize(
                    lambda p: -compute_lml(p, X, y), 
                    init_params, 
                    method='Nelder-Mead',
                    options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 50}
                )
                if res.success and res.fun > -1e9:
                    gp_sigma_f = max(np.abs(res.x[0]), 1e-3)
                    gp_length_scale = max(np.abs(res.x[1]), 0.1)
                    gp_noise = max(np.abs(res.x[2]), 1e-8)
            except Exception:
                pass

        def gp_predict(X_new):
            """Predict mean and variance for X_new."""
            if len(X_obs) == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new))
                
            X = np.array(X_obs)
            y = np.array(y_obs)
            X_new = np.array(X_new)
            
            diff_obs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dists_obs = np.sum(diff_obs**2, axis=-1)
            K = (gp_sigma_f**2) * np.exp(-0.5 * sq_dists_obs / (gp_length_scale**2))
            K[np.diag_indices_from(K)] += gp_noise**2
            
            diff_new = X[:, np.newaxis, :] - X_new[np.newaxis, :, :]
            sq_dists_new = np.sum(diff_new**2, axis=-1)
            K_x = (gp_sigma_f**2) * np.exp(-0.5 * sq_dists_new / (gp_length_scale**2))
            
            try:
                L, _ = cho_factor(K)
                v = cho_solve((L, _), K_x.T) # (N, M)
            except Exception:
                # Fallback: return prior mean 0, prior variance sigma_f^2
                return np.zeros(len(X_new)), np.ones(len(X_new)) * (gp_sigma_f**2)

            mu = v.T @ y
            var = gp_sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, var

        def acquisition(x):
            """Expected Improvement for minimization."""
            x = np.array(x).reshape(1, -1)
            mu, var = gp_predict(x)
            mu = mu[0]
            sigma = np.sqrt(var[0])
            
            if sigma < 1e-8:
                return 0.0
            
            z = (best_f - mu) / sigma
            ei = (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei # Negate for minimization

        last_fit_call = calls
        fit_interval = 10
        
        while calls < budget and time.time() < deadline:
            # 1. Update GP parameters periodically
            if calls - last_fit_call >= fit_interval:
                update_gp_params()
                last_fit_call = calls
            
            # 2. Find next point by maximizing EI
            # Use multiple random starts to avoid local maxima in EI
            n_starts = 10
            best_ei_val = float('inf')
            best_ei_x = None
            
            # Generate starts: random in box, plus perturbations of best_x
            rng = np.random.default_rng(calls)
            starts = [rng.uniform(-5, 5, size=dim) for _ in range(n_starts)]
            # Add some starts near the current best to exploit
            for _ in range(5):
                starts.append(best_x + rng.normal(0, 0.5, size=dim))
            
            for s in starts:
                s = np.clip(s, -5, 5)
                try:
                    # Optimize EI locally from this start
                    res_ei = minimize(
                        acquisition, 
                        s, 
                        method='Nelder-Mead', 
                        options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 50}
                    )
                    if res_ei.fun < best_ei_val:
                        best_ei_val = res_ei.fun
                        best_ei_x = res_ei.x
                except Exception:
                    continue
            
            if best_ei_x is None:
                # Fallback: random point
                best_ei_x = rng.uniform(-5, 5, size=dim)
            
            best_ei_x = np.clip(best_ei_x, -5, 5)
            
            # 3. Evaluate the selected point
            # We can also do a tiny local refinement here, but EI selection usually lands well.
            # Given the budget constraint and GP overhead, we just evaluate directly.
            eval_and_track(best_ei_x)
            
            # Optional: If we have plenty of budget left and time, do a quick local polish
            # But for now, the GP loop itself handles the exploration/exploitation balance.
            
        return best_x, best_f
