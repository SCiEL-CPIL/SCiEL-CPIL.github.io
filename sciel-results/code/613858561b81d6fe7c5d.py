import numpy as np
import time

class Optimizer:
    """
    Surrogate-assisted optimizer (Expected Improvement) with a local polish phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Data storage
        X_data = []
        F_data = []

        def elapsed():
            return time.time() - t0

        def check_deadline():
            return elapsed() > deadline

        def evaluate(x):
            nonlocal used, best_f, best_x
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            used += 1
            X_data.append(x_clip.copy())
            F_data.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        # --- Phase 1: Initial Exploration ---
        # Sample a diverse set of points to get a good initial model.
        # Use a combination of center, axis-aligned, and random points.
        rng = np.random.default_rng(42)
        
        init_points = []
        init_points.append(np.zeros(dim))
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 4.0
                init_points.append(p)
        
        # Add random points to fill up initial budget (approx 20% of total, min 15)
        init_budget = max(15, total_budget // 5)
        num_random = init_budget - len(init_points)
        for _ in range(max(0, num_random)):
            p = rng.uniform(-4.5, 4.5, dim)
            init_points.append(p)
            
        for p in init_points:
            if check_deadline() or used >= total_budget:
                break
            evaluate(p)

        # --- Phase 2: Surrogate-Assisted Search (Expected Improvement) ---
        # Fit a simple Gaussian Process (or similar surrogate) and use EI to select next point.
        # To keep it fast and dependency-free (standard library + numpy), we implement a 
        # simplified kernel-based surrogate or use scipy if available? 
        # Actually, implementing a robust GP from scratch in numpy for 5D is possible but tricky.
        # Alternatively, we can use a simple local search (like Nelder-Mead or CMA-ES-like)
        # but the directive suggests surrogate-assisted. 
        # Let's try a simple "Kriging" with a linear kernel or RBF kernel.
        
        # Given the time constraints and complexity, a full GP implementation might be risky.
        # Let's try a simpler approach: Multivariate Normal model? No, that's just mean/cov.
        # Let's try a local surrogate: Fit a polynomial or just use a local search from the best points?
        # The directive specifically asks for "surrogate-assisted strategy that fits a sparse Gaussian Process or Kriging model".
        # Let's implement a basic GP with RBF kernel.
        
        def rbf_kernel(X1, X2, l):
            # X1: (N, D), X2: (M, D)
            # Returns (N, M)
            sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            np.maximum(sqdist, 0, out=sqdist)
            return np.exp(-0.5 * sqdist / (l * l))

        # Hyperparameters for GP
        sigma_f = 1.0  # Signal standard deviation
        l = 1.0        # Length scale
        sigma_n = 1e-6 # Noise
        
        # We will update the GP model periodically or after each step?
        # Fitting a GP is O(N^3). With N up to 500, this is fine (500^3 = 125M, might be slow in Python).
        # 500x500 matrix inversion is fast in numpy.
        
        # Let's use a smaller budget for GP steps, or fit once and then use local search?
        # Or fit GP, pick best EI point, evaluate, repeat.
        
        # To ensure progress, we will interleave GP steps with local polish steps.
        
        # Let's define the GP step function
        def gp_ei_step():
            nonlocal used, best_f, best_x
            if check_deadline() or used >= total_budget:
                return
            
            X = np.array(X_data)
            F = np.array(F_data)
            
            # Standardize F for stability
            F_mean = np.mean(F)
            F_std = np.std(F)
            if F_std < 1e-10:
                F_std = 1.0
            F_norm = (F - F_mean) / F_std
            
            # Compute kernel matrix K
            K = rbf_kernel(X, X, l)
            K += sigma_n * np.eye(len(X))
            
            # Regularize to ensure positive definiteness
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return np.zeros(dim)
            
            # Predict at a candidate set
            # We need to choose candidate points. Use a grid or random samples.
            # Since dim=5, a fine grid is expensive. Let's use random candidates + local neighborhood.
            n_candidates = 50
            candidates = []
            # Random points in box
            for _ in range(20):
                candidates.append(rng.uniform(-5, 5, dim))
            # Points around current best
            for _ in range(30):
                candidates.append(best_x + rng.normal(0, 0.5, dim))
            candidates = np.array(candidates)
            
            # Compute K(x_c, X)
            K_c = rbf_kernel(candidates, X, l)
            
            # Mean and Variance
            # mu = K_c @ L^-T @ (L^-1 @ F_norm)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, F_norm))
            mu = K_c @ alpha
            v = np.linalg.solve(L, K_c.T)
            var = sigma_f**2 - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var)
            
            # Expected Improvement
            # EI(x) = (f_best_norm - mu) * Phi(z) + std * phi(z)
            # where z = (f_best_norm - mu) / std
            f_best_norm = (best_f - F_mean) / F_std
            
            z = (f_best_norm - mu) / std
            # Use standard normal CDF and PDF
            # For large negative z, Phi(z) ~ 0. For large positive, Phi(z) ~ 1.
            from scipy.stats import norm
            ei = (f_best_norm - mu) * norm.cdf(z) + std * norm.pdf(z)
            
            # Select best EI
            idx = np.argmax(ei)
            return candidates[idx]

        # Let's also implement a simple local polish (Nelder-Mead or (1+1)-ES) to run after some GP steps.
        # Strategy:
        # 1. Do ~100 GP steps.
        # 2. Do local polish from best point with remaining budget.
        
        gp_steps = 0
        max_gp_steps = 150  # Limit GP steps to avoid too much model fitting overhead
        
        while gp_steps < max_gp_steps and not check_deadline() and used < total_budget:
            x_next = gp_ei_step()
            if x_next is None:
                break
            evaluate(x_next)
            gp_steps += 1
            
            # Every 10 steps, try a small local polish to exploit
            if gp_steps % 10 == 0 and used < total_budget - 50:
                # Simple (1+1)-ES for 20 steps
                x_cur = best_x.copy()
                f_cur = best_f
                sigma = 0.5
                for _ in range(20):
                    if check_deadline() or used >= total_budget:
                        break
                    y = x_cur + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    f_y = func(y)
                    used += 1
                    X_data.append(y.copy())
                    F_data.append(f_y)
                    if f_y < best_f:
                        best_f = f_y
                        best_x = y.copy()
                    if f_y < f_cur:
                        x_cur = y
                        f_cur = f_y
                        sigma *= 1.05
                    else:
                        sigma *= 0.9
                        if sigma < 0.01:
                            break

        # --- Phase 3: Final Polish ---
        # Use remaining budget for a dedicated local search (Nelder-Mead or CMA-ES style)
        if not check_deadline() and used < total_budget:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma = 0.2
            
            # (1+1)-ES
            while used < total_budget and not check_deadline():
                y = x_polish + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = func(y)
                used += 1
                X_data.append(y.copy())
                F_data.append(f_y)
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
                if f_y < f_polish:
                    x_polish = y
                    f_polish = f_y
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 1e-5:
                        break

        return best_x, best_f
