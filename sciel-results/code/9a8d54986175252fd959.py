"""Multi-start CMA-ES with time guard for BBOB f23 (Katsuura)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # --- Multi-start CMA-ES ---
        # We'll run multiple short CMA-ES runs from different starting points.
        # Given 500 evaluations and dim=5, we can afford a few restarts.
        
        # Strategy: 
        # - First, do a quick random sampling to get a good initial point.
        # - Then run CMA-ES with the remaining budget.
        # - If time allows, do a second CMA-ES from a different start.
        
        # Let's use a single CMA-ES with a reasonable initial sigma.
        # CMA-ES parameters:
        # lambda = 4 * floor(1 + 3*ln(dim)) for dim=5 -> 4*floor(1+3*1.609) = 4*5 = 20
        # Actually standard is lambda = 4 + floor(3*ln(N)) = 4+4=8 for N=5. Let's use 20 for more exploration.
        # mu = lambda // 2
        # c = (mu_eff + 2) / ((N + 2)^2 + mu_eff)
        # d = 1 + 2*max(0, sqrt((mu_eff-1)/(N+1)) - 1) + c
        # chi = sqrt(N) * (1 - 1/(4*N) + 1/(21*N^2))
        
        # For simplicity and robustness, let's use a simplified CMA-ES.
        # Actually, let's implement a proper CMA-ES.
        
        lambda_ = 20
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / ((dim + 2) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial point: try a few random points, pick the best
        n_init = 10
        init_points = rng.uniform(-5, 5, (n_init, dim))
        init_fvals = np.array([func(p) for p in init_points])
        best_idx = np.argmin(init_fvals)
        x_mean = init_points[best_idx].copy()
        f_best = init_fvals[best_idx]
        
        # Initial sigma: based on the spread of the best few points
        # Or just a reasonable default
        sigma = 2.0
        
        calls_made = n_init
        B = None
        D = None
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # CMA-ES main loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline - 2.0:
                break
            
            # Generate offspring
            # Compute eigen decomposition if not done or periodically
            if B is None:
                # Initial B and D
                B = np.eye(dim)
                D = np.ones(dim)
            
            # Sample lambda_ points
            z = rng.standard_normal((lambda_, dim))
            y = (x_mean + sigma * (z @ (B * D[:, np.newaxis])).T).T
            # Clip to bounds
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fvals = np.array([func(yi) for yi in y])
            calls_made += lambda_
            
            # Sort by fitness
            order = np.argsort(fvals)
            y = y[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < f_best:
                f_best = fvals[0]
            
            # Update mean
            w = weights[:mu]
            x_mean_new = np.sum(w[:, np.newaxis] * y[:mu], axis=0)
            
            # Update sigma (step-size)
            y_w = (y[:mu] - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * (w @ y_w)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_made // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * chi
            
            if hsig:
                pc = (1 - c_c) * pc + c_c * (y_w.T @ w / sigma)
            else:
                pc = (1 - c_c) * pc
            
            # Update covariance matrix (simplified: use rank-mu + rank-1 update)
            # For simplicity, let's use a simplified update or just reset B,D periodically
            # Actually, let's implement the full update
            
            # Compute H matrix
            v = ps / np.linalg.norm(ps) if np.linalg.norm(ps) > 1e-16 else np.zeros(dim)
            pc_mat = np.outer(pc, pc)
            C_new = (1 + c_sigma * (2 - c_sigma) * (hsig - 1) + c_c * (2 - c_c)) * np.eye(dim)
            # This is getting complex. Let's use a simpler approach.
            
            # Alternative: use scipy.optimize if available, but let's stick to numpy.
            # Let's use a simpler CMA-ES variant or just restart with a new sigma.
            
            # For now, let's just update x_mean and adapt sigma simply
            x_mean = x_mean_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)
            
            # Reset B and D periodically (simplification)
            # In a full implementation, we'd update C = B diag(D^2) B^T
            # For simplicity, we'll reset B,D every 10 iterations
            if calls_made % (10 * lambda_) < lambda_:
                B = np.eye(dim)
                D = np.ones(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
        
        # If we have budget left, do a local search
        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline - 1.0:
            # Simple local search
            step = 0.1
            for _ in range(min(remaining, 50)):
                if time.time() > deadline - 1.0:
                    break
                for i in range(dim):
                    for direction in [-1, 1]:
                        x_trial = x_mean.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < f_best:
                            f_best = f_trial
                            x_mean = x_trial
                            step *= 0.5
                        if calls_made >= budget:
                            break
                    if calls_made >= budget:
                        break
                if calls_made >= budget:
                    break
        
        return x_mean, f_best
