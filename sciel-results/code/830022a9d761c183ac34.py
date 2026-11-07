"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        
        best_x = None
        best_f = np.inf
        call_count = 0

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def run_cmaes(x0, sigma0):
            nonlocal best_x, best_f, call_count, deadline
            
            n = self.dim
            
            # CMA-ES Parameters
            # For n=5, lambda=9 is standard (4 + int(3*log(5)) = 4+4=8, usually rounded up or kept at 9 for robustness)
            lam = 9 
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (n + 1.3) / (n + 3 + 2 * (n / (n + 1)) ** 2)
            c_c = (4 + n / n) / (n + 4 + 2 * (n / (n + 1)) ** 2)
            c_cov = (2 + 2.0 / (mu_eff + 1)) / (n + 3 + 2.0 / (mu_eff + 1))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_cov
            
            # State
            x = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            
            # Initial evaluation
            if call_count >= self.budget:
                return
            fx = make_call(x)
            if fx is None:
                return
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()

            while call_count < self.budget and (time.time() < deadline - 2.0):
                # 1. Sample population
                # Compute Cholesky of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite (rare in practice with standard CMA-ES, but possible with bad steps)
                    C = np.eye(n)
                    L = np.eye(n)
                
                ys = np.array([L @ np.random.standard_normal(n) for _ in range(lam)])
                xs = np.array([x + sigma * y for y in ys])
                
                # 2. Evaluate
                fvals = []
                valid_indices = []
                for i in range(lam):
                    if call_count >= self.budget or time.time() > deadline - 2.0:
                        break
                    xi = np.clip(xs[i], -5, 5)
                    fi = make_call(xi)
                    if fi is None:
                        return
                    fvals.append(fi)
                    valid_indices.append(i)
                    if np.isfinite(fi) and fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                if len(fvals) < lam:
                    break
                    
                # 3. Sort
                fvals_arr = np.array(fvals)
                sorted_idx = np.argsort(fvals_arr)
                xs_sorted = xs[valid_indices][sorted_idx]
                ys_sorted = ys[sorted_idx]
                fvals_sorted = fvals_arr[sorted_idx]
                
                # 4. Update mean
                x_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # 5. Update path variables
                # ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * C^-1/2 * (x_new - x)/sigma
                # We need C^-1/2. Since L = C^1/2, C^-1/2 = L^-T.
                # For n=5, solving linear systems is cheap.
                try:
                    C_inv_sqrt = np.linalg.inv(L.T)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)
                    
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x) / sigma)
                
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count + 1))) < (1.4 + 2 / (n + 1)) * chi_n)
                
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                
                # 6. Update covariance
                # C = (1 + c_cov(2 - c_cov) + hsig * c_c(2 - c_c - c_cov))C 
                #    + c_cov (ps ps^T - (1 - c_c)C) + (1 - c_cov) * sum(weights_i (y_i - y_mean)(y_i - y_mean)^T)
                
                # Note: Standard CMA-ES updates C based on the rank-mu update of the *selected* solutions.
                # The term (y_i - y_mean) usually uses the weighted mean of the selected ys.
                # Let's stick to the standard formula:
                # C = (1 + c_cov(2 - c_cov) + hsig * c_c(2 - c_c - c_cov))C 
                #     + c_cov (ps ps^T - (1 - c_c)C) 
                #     + (1 - c_cov) * (sum_{i=1}^mu weights_i (y_i - y_mean)(y_i - y_mean)^T)
                
                # Wait, the standard rank-mu update uses the *normalized* steps y_i.
                # y_mean is the weighted mean of the selected y_i's.
                y_mean = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
                
                # Compute rank-mu update part
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    diff = ys_sorted[i] - y_mean
                    rank_mu += weights[i] * np.outer(diff, diff)
                    
                # Combine
                c1 = c_cov
                c1_mu = 1 - c_cov
                
                # The factor for the "rank-1" part (ps ps^T) and the "rank-mu" part
                # Standard:
                # C = (1 - c1) C + c1 (ps ps^T + (1 - c_c) C) + c1_mu * rank_mu
                # Let's expand:
                # C = (1 - c1) C + c1 ps ps^T + c1(1 - c_c) C + c1_mu rank_mu
                # C = (1 - c1 + c1(1 - c_c)) C + c1 ps ps^T + c1_mu rank_mu
                # C = (1 - c1 c_c) C + c1 ps ps^T + c1_mu rank_mu
                
                # Also need to account for the hsig term? 
                # The hsig term is usually embedded in the update of ps and pc, 
                # but some formulations add it to C directly if hsig is false? 
                # Actually, the standard Hansen formulation:
                # C = (1 + c_cov(2 - c_cov) + hsig * c_c(2 - c_c - c_cov)) C
                #     + c_cov (ps ps^T - (1 - c_c) C)
                #     + (1 - c_cov) * (rank_mu)
                
                # Let's use this exact form to be safe.
                factor_c = 1 + c_cov * (2 - c_cov) + (hsig * c_c * (2 - c_c - c_cov) if hsig else 0)
                # If hsig is False, the term is usually just (1 + c_cov(2-c_cov))C + c_cov(ps ps^T - (1-c_c)C) + ...
                # Wait, if hsig is 0, the pc update is (1-c_c)pc, so the "memory" of pc is reduced.
                # The C update formula often includes the hsig indicator in the coefficient of C.
                
                # Let's simplify to the most common robust implementation:
                # C = (1 - c1) C + c1 (ps ps^T + (1 - c_c) C) + c1_mu * rank_mu
                # This is equivalent to:
                # C = (1 - c1 c_c) C + c1 ps ps^T + c1_mu rank_mu
                
                C = (1 - c_cov * c_c) * C + c_cov * np.outer(ps, ps) + c1_mu * rank_mu
                
                # 7. Update sigma (CSA - Cumulative Step Adaptation)
                # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / chi_n - 1))
                # We need d_sigma.
                d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_cov
                sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
                
                # Ensure sigma doesn't explode or vanish
                if sigma > 1e6: sigma = 1e6
                if sigma < 1e-12: sigma = 1e-12
                
                x = x_new
                
                # Check for convergence
                # If sigma is very small or C is nearly singular
                if sigma < 1e-8:
                    break
                # Check condition number of C
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if eigvals[0] < 1e-12:
                        break
                    cond = eigvals[-1] / max(eigvals[0], 1e-12)
                    if cond > 1e12:
                        break
                except:
                    pass

        # Strategy:
        # 1. Evaluate origin.
        # 2. Run CMA-ES from origin with a reasonable initial sigma.
        # 3. If budget remains and CMA-ES converged or stopped early, try a restart from the best point found 
        #    with a larger sigma to escape local minima (though CMA-ES is global, it can get stuck in flat regions).
        
        rng = np.random.default_rng(42)
        
        # Initial probe
        x0 = np.zeros(self.dim)
        f0 = make_call(x0)
        if f0 is not None:
            if np.isfinite(f0) and f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # Run CMA-ES from origin
        # Initial sigma: The box is [-5, 5]. A good start is 1.0 or 2.0.
        # Since the minimum is in [-4, 4], 1.0 is safe.
        if call_count < self.budget:
            run_cmaes(x0, 1.0)
            
        # If we have significant budget left (e.g., > 100 calls), try a restart.
        # CMA-ES is stochastic, so a restart from the current best with a perturbed sigma might help.
        if best_x is not None and call_count < self.budget - 50:
            # Restart from best_x with a slightly different sigma or random perturbation
            # To avoid re-evaluating the exact same point, we can just let CMA-ES run again.
            # But CMA-ES uses its own RNG. We should reset the RNG or use a different seed?
            # The current implementation uses np.random.standard_normal which uses the global RNG.
            # Let's create a local RNG for the second run to ensure diversity?
            # Actually, re-running CMA-ES from the best point found is a standard "restart" strategy.
            
            # Perturb the start point slightly to avoid identical trajectory
            x_restart = best_x.copy()
            # Add small noise
            x_restart += np.random.normal(0, 0.1, self.dim)
            x_restart = np.clip(x_restart, -5, 5)
            
            run_cmaes(x_restart, 0.5) # Smaller sigma for local refinement
            
        if best_x is None:
            # Fallback
            x = np.zeros(self.dim)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
