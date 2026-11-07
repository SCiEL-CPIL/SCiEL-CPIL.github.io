"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # Time limit guard
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        
        # Initial mean
        # Start near center as minimizer is in [-4, 4]^5
        m = np.zeros(dim)
        
        # Initial step size
        # Weierstrass functions have features at various scales.
        # A standard initial sigma is 1.0. 
        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 10.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Learning rates
        # For small dim, these are standard
        mu = lam // 2  # Half the population contributes to update
        
        # Weights (proportional to rank)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation (CSA)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        
        # Covariance matrix adaptation (PC-ES)
        c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Path accumulators
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Damping for CSA
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Best solution found
        best_x = m.copy()
        best_f = func(best_x)
        evals_used = 1
        
        # Pre-allocate population storage if needed, but generating on the fly is fine
        while evals_used < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            # 1. Sample population
            # Generate standard normal vectors
            ys = np.random.randn(lam, dim)
            
            # Whitening transformation (Cholesky or Eigen)
            # For dim=5, Cholesky is fast
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-PD due to numerical issues
                C = np.eye(dim)
                L = np.eye(dim)
                
            # Sample z ~ N(0, 1)
            zs = ys @ L.T  # This results in N(0, C) if C is PD
            # Actually, standard CMA-ES samples z ~ N(0, I), then x = m + sigma * z @ L.T
            # Wait, if we want x ~ N(m, sigma^2 C), we need z ~ N(0, I)
            # Let z = standard normal. x = m + sigma * z @ L.T where L is Cholesky of C.
            
            # Correction: 
            # We want x = m + sigma * (L @ z)
            # If z is (lam, dim) rows, then x = m + sigma * (z @ L.T)
            
            xs = m[None, :] + sigma * (zs)
            
            # Clip to domain
            xs = np.clip(xs, -5, 5)
            
            # 2. Evaluate
            fs = np.array([func(x) for x in xs])
            evals_used += lam
            
            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = xs[idx_best]
                
            # 3. Update CMA-ES state
            # Sort by fitness
            idx_sorted = np.argsort(fs)
            y_sorted = ys[idx_sorted]
            x_sorted = xs[idx_sorted]
            f_sorted = fs[idx_sorted]
            
            # Weighted mean
            # Use only top mu individuals
            w = weights
            new_m = np.dot(w, x_sorted[:mu])
            
            # Update step size (CSA)
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^(-1/2) * (new_m - m)/sigma
            # C^(-1/2) is L^(-T)
            # We have L such that C = L L.T. So C^(-1/2) = L^(-T).
            # (new_m - m)/sigma is the mean shift in standardized space? No, it's the raw shift divided by sigma.
            # The formula is: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) / sigma * C^{-1/2} (new_m - m)
            # Let's compute C^{-1/2} (new_m - m)
            # We can solve L.T v = (new_m - m)/sigma for v?
            # C^{-1/2} x = L^{-T} x.
            # Let's just use scipy.linalg or numpy solve.
            # dim is 5, so this is cheap.
            
            diff = (new_m - m) / sigma
            # Solve L.T * v = diff
            # L is lower triangular
            v = np.linalg.solve(L.T, diff)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
            
            # hsigma condition
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (budget - evals_used + lam))) / chiN
            # Note: The exponent in the denominator uses the number of remaining steps? 
            # Standard formula: hsig = ||ps|| / (chiN * sqrt(1 - (1-cs)^2 * t)) where t is step number?
            # Actually, it's simpler: hsig = ||ps|| / (chiN * sqrt(1 - (1-cs)^(2 * lam))) is not quite right for online updates.
            # Common approximation: hsig = 1.0 if ||ps|| < (1.4 + 2/(dim+1)) * chiN else 0.0
            # Let's use the simple condition for stability.
            hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0
            
            # Update sigma
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
            
            # Update covariance matrix
            # pc = (1-c1)*pc + hsig*sqrt(c1*(2-c1)*mueff)*(new_m-m)/sigma
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff
            
            # Rank-1 update
            rank1_update = np.outer(pc, pc)
            
            # Rank-mu update
            # v = (x_sorted[:mu] - m) / sigma
            v_mu = (x_sorted[:mu] - m) / sigma
            rankmu_update = np.dot((w[:, None] * v_mu).T, v_mu)
            
            # Update C
            C = (1 + c1 * hsig * (2 - hsig) - c1 - cmu) * C + c1 * rank1_update + cmu * rankmu_update
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Ensure positive definiteness?
            # For dim=5, usually not needed if parameters are tuned well.
            # But let's check eigenvalues occasionally? No, too slow.
            # If Cholesky fails, we reset C in the next iteration.
            
            # Update mean
            m = new_m
            
            # Check if we have enough budget for next full population
            if budget - evals_used < lam:
                # Do a few more single steps if budget allows?
                # Or just stop.
                # Since we clip evals, we might have leftover budget.
                # Let's use remaining budget for a few extra steps of size 1?
                # Actually, the loop condition `evals_used < budget` will handle this if we change step size.
                # But for simplicity and to avoid complex partial updates, we can break if we can't do a full pop.
                # However, 500 is not divisible by 8. 500 = 62 * 8 + 4.
                # We will do 62 full generations. Then we have 4 evals left.
                # We can use those 4 evals to refine the current best with small steps.
                break

        # Use remaining budget for local refinement around best_x
        remaining = budget - evals_used
        if remaining > 0 and time.time() - start_time < time_limit - 5:
            # Use scipy Nelder-Mead for high-precision polish
            # Reserve some budget for the local optimizer
            local_budget = min(remaining, 200)
            
            # Define bounds for scipy
            bounds = [(-5, 5)] * dim
            
            try:
                # Use Nelder-Mead with tight tolerances
                result = scipy_minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={
                        'maxiter': local_budget // 2,
                        'maxfev': local_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun
                    evals_used += result.nfev
            except Exception:
                # Fallback to simple local search if scipy fails
                local_sigma = max(sigma * 0.1, 1e-4)
                for _ in range(remaining):
                    if time.time() - start_time > time_limit:
                        break
                    y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    if fy < best_f:
                        best_x = y
                        best_f = fy
                        local_sigma *= 1.1
                    else:
                        local_sigma *= 0.9
                    evals_used += 1
                    if evals_used >= budget:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
