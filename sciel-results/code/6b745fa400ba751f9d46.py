"""Model-and-optimize black-box optimizer for BBOB f18 (Schaffers F7).

Strategy:
1. Robust CMA-ES with Rejection Sampling to handle the box constraints properly.
2. Adaptive Stagnation Restart: Resets covariance but keeps the best mean.
3. Nelder-Mead Polishing: A final local search phase to refine the solution
   in the ill-conditioned basin.
"""
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with rejection sampling and Nelder-Mead polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 320.0  # Safety margin before 330s kill
        calls = [0]
        n = self.dim
        
        def time_ok():
            return (time.time() - t0) < WALL

        def safe_eval(x):
            """One recorded, exception-safe, budget/time-guarded evaluation."""
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x = np.zeros(n)
        best_f = np.inf

        # --- Phase 1: CMA-ES ---
        # Hyperparameters
        lam = 20
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / float(w @ w)
        
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c1 = 2.0 / (n ** 1.6 + 18.0)
        cmu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff)
        chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        def run_cma_es(start_x, start_sigma, max_generations):
            nonlocal best_x, best_f
            
            m = start_x.copy()
            sigma = start_sigma
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            g = 0
            g_since_improve = 0
            local_best_f = best_f
            local_best_x = best_x.copy()

            while calls[0] < self.budget and time_ok() and g < max_generations:
                if self.budget - calls[0] < lam:
                    break

                # Generate population with Rejection Sampling
                X = []
                Ys = []
                attempts = 0
                max_attempts = lam * 10 # Prevent infinite loop if box is tight
                
                while len(X) < lam and attempts < max_attempts:
                    z = rng.standard_normal(n)
                    y = (z @ B.T) * D
                    x = m + sigma * y
                    if np.all(x >= -5.0) and np.all(x <= 5.0):
                        X.append(x)
                        Ys.append(y)
                    attempts += 1
                if len(X) < lam:
                    # If we can't fill the population, pad with best_x (degenerate case)
                    while len(X) < lam:
                        X.append(local_best_x.copy())
                        Ys.append(np.zeros(n))
                
                X = np.array(X)
                Ys = np.array(Ys)
                
                # Evaluate
                fs = np.array([safe_eval(x) for x in X])
                fs[fs is None] = 1e18 # Handle None if budget/time hit mid-loop

                # Update best
                fi = int(np.argmin(fs))
                if fs[fi] < local_best_f:
                    local_best_f = float(fs[fi])
                    local_best_x = X[fi].copy()
                    g_since_improve = 0
                else:
                    g_since_improve += 1

                # CMA-ES Update
                idx = np.argsort(fs)
                top_idx = idx[:mu]
                Ys_top = Ys[top_idx]
                Xs_top = X[top_idx]
                
                # Weighted mean
                m_new = w @ Xs_top
                m_new = np.clip(m_new, -5.0, 5.0) # Clip mean to box for stability
                
                # Step size adaptation
                ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (w @ Ys_top)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (g + 1)))) / chiN < \
                       1.4 + 2.0 / (n + 1)
                pc = (1.0 - c_c) * pc + hsig * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * (w @ (Xs_top - m[None, :])) / sigma

                C = (1.0 - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) + (hsig - 1.0 + c1) * C)
                C += cmu * sum(wi * np.outer(y, y) for wi, y in zip(w, Ys_top))

                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1.0))
                sigma = float(np.clip(sigma, 1e-10, 10.0))

                m = m_new

                try:
                    eig, B = np.linalg.eigh(C)
                    eig = np.maximum(eig, 1e-12)
                    D = np.sqrt(eig)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    sigma *= 0.5 # Penalize failure

                g += 1

                # Stagnation Check: Restart logic
                # If no improvement for 15 generations, or sigma is tiny
                if g_since_improve > 15 or (g > 5 and sigma < 1e-5):
                    # Update global best
                    best_f = local_best_f
                    best_x = local_best_x.copy()
                    
                    # Soft Restart:
                    # Keep the best point as the new mean.
                    # Reset covariance to identity.
                    # Increase sigma slightly to explore around the best point.
                    if g > 5: # Don't restart immediately
                        m = local_best_x.copy()
                        sigma = max(sigma * 2.0, 0.5) # Ensure we explore a bit
                        C = np.eye(n)
                        B = np.eye(n)
                        D = np.ones(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                        g = 0
                        g_since_improve = 0
                        continue # Continue loop with reset state
                    else:
                        break

            # Update global best at end of CMA-ES phase
            best_f = local_best_f
            best_x = local_best_x.copy()

        # Initialize CMA-ES
        # Start with a point in the center, large sigma to cover the box
        init_x = rng.uniform(-2.0, 2.0, n)
        run_cma_es(init_x, 2.0, 20) # Allow up to 20 restarts/generations logic inside

        # --- Phase 2: Nelder-Mead Polishing ---
        # Use the best point from CMA-ES as the starting point for local search.
        # Nelder-Mead is efficient for low-dim smooth-ish functions.
        if calls[0] < self.budget and time_ok():
            x0 = best_x.copy()
            
            # Wrap func for scipy
            def scipy_func(x):
                if calls[0] >= self.budget or not time_ok():
                    return 1e18
                return safe_eval(x)

            # Try Nelder-Mead
            try:
                res = minimize(
                    scipy_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.success or res.fun < best_f:
                    if np.isfinite(res.fun):
                        best_f = float(res.fun)
                        best_x = res.x
            except Exception:
                pass

        # --- Phase 3: Final (1+1)-ES Polish ---
        # If Nelder-Mead didn't use all budget, do a fine-grained local search.
        if calls[0] < self.budget and time_ok():
            m_tail = best_x.copy()
            sigma_tail = 0.1
            while calls[0] < self.budget and time_ok():
                y = m_tail + sigma_tail * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_eval(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    m_tail = best_x
                    sigma_tail *= 1.1
                else:
                    sigma_tail *= 0.9
                    sigma_tail = float(np.clip(sigma_tail, 1e-10, 10.0))

        return best_x, float(best_f)
