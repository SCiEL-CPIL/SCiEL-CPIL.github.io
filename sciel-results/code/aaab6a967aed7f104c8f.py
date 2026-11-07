"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        time_limit = 290.0  # 300s limit, keep 10s margin

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam > budget:
            lam = budget
        # Number of weighted recombination points
        mu = lam // 2
        # Weights
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.dot(weights, weights)
        
        # Strategy parameters
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 2.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        dsigma = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (mueff + 1.0)) - 1.0) + cs

        # Initial state
        xmean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sqrtc1 = np.sqrt(c1)
        sqrtcmu = np.sqrt(cmu)
        
        # Cholesky factor of C
        try:
            C12 = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C12 = np.eye(dim)

        # Evaluation loop
        evals = 0
        best_x = xmean.copy()
        best_f = np.inf
        
        # Generate initial population
        while evals < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            # Generate offspring
            # Use standard normal and transform
            z = np.random.standard_normal((lam, dim))
            # Apply Cholesky factor
            y = z @ C12.T
            pop = xmean + sigma * y
            
            # Evaluate
            fvals = np.array([func(xi) for xi in pop])
            evals += lam
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()

            if evals >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update mean
            xnew = np.dot(weights, pop[:mu])
            
            # Update step size control variable ps
            # Inverse Cholesky factor
            try:
                Cinv12 = np.linalg.inv(C12)
            except np.linalg.LinAlgError:
                Cinv12 = np.eye(dim)
                
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv12 @ (xnew - xmean) / sigma)
            
            # Update evolution paths
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals + 1) / (dim * cs)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xnew - xmean) / sigma
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(weights * np.outer(pop[:mu] - xmean, pop[:mu] - xmean) / sigma**2, axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update Cholesky factor
            try:
                C12 = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, reset to identity (rare)
                C = np.eye(dim)
                C12 = np.eye(dim)
            
            # Update step size
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update mean
            xmean = xnew
            
            # Clip xmean to box? No, CMA-ES handles this via sampling, 
            # but func clips internally. However, keeping mean in bounds is good practice.
            # The problem states func clips, so we don't strictly need to clip xmean,
            # but it helps stability.
            xmean = np.clip(xmean, -5, 5)

        return best_x, best_f

# EVOLVE-BLOCK-END
