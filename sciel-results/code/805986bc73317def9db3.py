"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid (1+1)-ES followed by CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 320.0 # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        
        # Phase 1: (1+1)-ES for fast initial convergence
        # Use a significant portion of budget if it's small, otherwise a fixed number
        # 300 budget is decent. Let's use 50 evaluations for (1+1)-ES.
        # This ensures we have a good starting point for CMA-ES.
        
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        
        sigma = 1.0
        calls_made = 1
        
        # (1+1)-ES phase
        es_budget = min(50, budget // 6) # Use ~1/6th for initial exploration
        for i in range(es_budget):
            if time.time() - start_time > timeout:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.85
            # Adapt sigma based on success rate roughly? 1/5 rule is standard but simple multiplicative is okay for short phase.
            
        # Phase 2: CMA-ES
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x, fx

        # CMA-ES Parameters
        # Population size N
        N = max(4, int(4 + 3 * np.log(dim)))
        # Ensure we have enough budget for at least 2 generations
        if remaining_budget < 2 * N:
             # Fall back to (1+1)-ES for the rest
             for i in range(remaining_budget):
                 if time.time() - start_time > timeout:
                     break
                 y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                 fy = func(y)
                 calls_made += 1
                 if fy < fx:
                     x, fx = y, fy
                     sigma *= 1.5
                 else:
                     sigma *= 0.85
             return x, fx

        # Initialize CMA-ES state
        # Mean
        m = x.copy()
        # Covariance matrix
        C = np.eye(dim)
        # Step size
        sigma = max(0.1, sigma) # Keep some momentum from ES, but cap it
        
        # Weights
        mu = N // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (w[1] - 1) / ((dim + 2)**2 - mueff))
        
        # Path accumulators
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Pre-compute constants for sigma update
        # Norm of expected normal vector
        # For standard normal, norm is approx sqrt(dim)
        
        best_x = x.copy()
        best_f = fx
        
        while calls_made < budget:
            if time.time() - start_time > timeout:
                break
            
            # Generate population
            # Use Cholesky decomposition for sampling from N(0, C)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular (shouldn't happen often with regularization)
                L = np.eye(dim)
                
            # Sample N points
            # z ~ N(0, I)
            Z = rng.standard_normal((N, dim))
            # y = m + sigma * L @ z.T
            Y = m[None, :] + sigma * (L @ Z.T).T
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(yi) for yi in Y])
            calls_made += N
            
            # Sort by function value
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # Weighted mean of the top mu
            yw = np.sum(w[:mu, None] * Y_sorted[:mu], axis=0)
            
            # Update path accumulators
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^(-1/2) * (yw - m)/sigma
            # We need C^(-1/2). We can use L^(-T) * L^(-1) or just solve.
            # Actually, standard CMA-ES uses C^(-1/2) * (yw-m)/sigma.
            # Let's compute the transformed step
            step = (yw - m) / sigma
            # Apply C^(-1/2)
            # Since C = L @ L.T, C^(-1/2) is not simply L^(-1).
            # However, we can update ps in the "white noise" space if we transform the whole system, 
            # but the standard implementation updates ps in the original space using the inverse square root.
            # A simpler approximation often used in simplified implementations:
            # Just use the step directly if C is close to identity, or compute the transformation.
            # For robustness and speed, let's compute the transformation via solve.
            # We want v such that C^(-1/2) v = step.
            # This is equivalent to solving C v = C^(-1/2) step? No.
            # Let's stick to the standard formula:
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (L^(-T) * (yw-m)/sigma) ? 
            # Actually, the standard update is:
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^(-1/2) * (yw - m) / sigma
            
            # Computing C^(-1/2) is expensive. 
            # Alternative: Update ps in the "z-space" (before applying L).
            # Let's restart with a simpler, more robust CMA-ES variant that avoids explicit C^(-1/2) 
            # by updating the mean and covariance directly from the samples, 
            # and adapting sigma based on the norm of the step in the transformed space.
            
            # To keep it simple and fast:
            # 1. Update mean m = yw
            # 2. Update C = (1 - c1 - cmu)*C + c1*outer(ps,ps) + cmu*sum(wi*outer(zi,zi))
            #    where zi = (yi - m_old)/sigma
            # 3. Update sigma based on norm of ps.
            
            # Let's compute zi for the top mu
            Z_sorted = Z[order]
            Zi = Z_sorted[:mu] # These are the standard normal samples that produced the top solutions
            
            # Update ps
            # The term C^(-1/2) * (yw - m) / sigma is effectively the step in the "white noise" space
            # which is approximated by the mean of the Z vectors? No.
            # Let's use the standard update but approximate C^(-1/2) by using the Cholesky factor L.
            # If C = L L^T, then C^(-1/2) is not L^(-1).
            # However, for dim=5, we can just compute the matrix inverse square root.
            # Eigen decomposition is stable.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small eigenvalues
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            step_white = C_inv_sqrt @ step
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * step_white
            
            # Update pc
            # Check if ps is too large (delta_hsig)
            # norm_ps = np.linalg.norm(ps)
            # threshold = (1.4 + 2/(dim+1)) * np.sqrt(dim) # Approximation
            
            # For simplicity and robustness in a short run, we might skip the hsig check or simplify it.
            # Let's include a basic check.
            # The expected norm of ps is sqrt(1 - (1-cs)^(2*t))
            # We don't track t exactly, so we use a heuristic.
            
            pc = (1 - cc) * pc + cc * step_white
            
            # Update mean
            m = yw
            
            # Update C
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(ps, ps)
            for i in range(mu):
                C += cmu * w[i] * np.outer(Zi[i], Zi[i])
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp( cs / dsigma * ( norm(ps) / sqrt(1-(1-cs)^(2*t)) - 1 ) )
            # Approximate t by calls_made / N
            t = calls_made / N
            expected_norm = np.sqrt(1 - (1 - cs)**(2 * t))
            if expected_norm > 1e-8:
                norm_ps = np.linalg.norm(ps)
                factor = (norm_ps / expected_norm - 1)
                # dsigma is usually around 1 to 5. Let's use 1 for aggressive adaptation or 5 for stable.
                # In BBOB, stable is better.
                sigma *= np.exp(cs * factor / 5.0)
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
