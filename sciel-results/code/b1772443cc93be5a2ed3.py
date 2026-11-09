import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Coarse exploration with fewer points to save budget for CMA-ES
        n_explore = min(30, self.budget // 10)
        for i in range(n_explore):
            if time.time() > deadline or call_count >= self.budget:
                break
            x = rng.uniform(-4.0, 4.0, size=self.dim)
            try:
                wrapped_func(x)
            except (TimeoutError, StopIteration):
                break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)
        
        # Phase 2: Multi-restart CMA-ES
        remaining_budget = self.budget - call_count
        if remaining_budget > 10:
            new_best_x, new_best_f = self._run_multi_restart_cmaes(
                wrapped_func, best_x, best_f, remaining_budget, deadline, rng
            )
            if new_best_f < best_f:
                best_f = new_best_f
                best_x = new_best_x
        
        return best_x, best_f

    def _run_multi_restart_cmaes(self, wrapped_func, x0, f0, budget, deadline, rng):
        """Multi-restart CMA-ES with stagnation detection."""
        dim = self.dim
        global_best_x = x0.copy()
        global_best_f = f0
        calls_made = 0
        
        # CMA-ES parameters - smaller population for speed
        lambda_ = max(6, 4 + int(2 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = (2 - mu_eff / (dim**2 + 1)) / ((dim + 2)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        c_sigma = cs * (2 + 2 / (mu_eff + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        max_restarts = 5
        calls_per_restart = max(20, budget // (max_restarts + 1))
        
        for restart in range(max_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # For first restart, start from best point; for others, random restart
            if restart == 0:
                mean = global_best_x.copy()
                sigma = 1.5
            else:
                # Random restart biased toward best region
                if rng.random() < 0.5:
                    mean = global_best_x.copy() + 2.0 * rng.standard_normal(dim)
                else:
                    mean = rng.uniform(-4.0, 4.0, size=dim)
                mean = np.clip(mean, -5.0, 5.0)
                sigma = 1.5
            
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_f = np.inf
            local_best_x = None
            
            # Run CMA-ES for this restart
            max_iter = calls_per_restart // lambda_
            for iteration in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Compute C^{1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                # Generate and evaluate samples
                samples = []
                for i in range(lambda_):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    samples.append(y)
                
                if len(samples) < 2:
                    break
                
                fvals = []
                for y in samples:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    try:
                        fy = wrapped_func(y)
                        fvals.append((fy, y))
                        calls_made += 1
                    except (TimeoutError, StopIteration):
                        break
                
                if len(fvals) < 2:
                    break
                
                # Sort by fitness
                fvals.sort(key=lambda t: t[0])
                
                # Update global best
                if fvals[0][0] < global_best_f:
                    global_best_f = fvals[0][0]
                    global_best_x = fvals[0][1].copy()
                
                if fvals[0][0] < local_best_f:
                    local_best_f = fvals[0][0]
                    local_best_x = fvals[0][1].copy()
                
                # Compute weighted mean of top mu
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * fvals[i][1]
                
                # Update covariance matrix
                C_old = C.copy()
                diff = new_mean - mean
                rank_one = np.outer(diff, diff)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    diff_i = fvals[i][1] - mean
                    rank_mu += weights[i] * np.outer(diff_i, diff_i)
                
                C = (1 - c1 - cmu) * C_old + c1 * (rank_one + (1 - mu_eff) * C_old) + cmu * rank_mu
                
                # Update evolution paths
                ps = (1 - cs) * ps + cs / np.sqrt(mu_eff) * (C_old @ (diff / sigma))
                pc = (1 - c1) * pc + c1 * (np.sign(ps) - 1)
                
                # Adapt sigma
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 5.0)
                
                mean = new_mean
                
                # Stagnation: if sigma collapsed, break out to restart
                if sigma < 1e-5:
                    break
        
        return global_best_x, global_best_f
