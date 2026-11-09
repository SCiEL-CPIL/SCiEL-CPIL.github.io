import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Number of parents
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff)))
        
        # Damping for sigma update
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Initial state
        rng = np.random.default_rng(0)
        # Start near center, but maybe slightly off to avoid symmetry issues if any? 
        # For BBOB, center is usually safe for start, but global search handles the rest.
        m = np.zeros(dim) 
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = m.copy()
        best_f = float('inf')
        
        eval_count = 0
        
        # Pre-compute inverse square root of C for efficiency? 
        # For dim=5, just doing eigh every step is fine.
        
        while eval_count < budget:
            if time.time() > deadline:
                break
                
            # Sample population
            # Calculate sqrt of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Ensure positive
            sqrt_eigvals = np.sqrt(eigvals)
            
            # Generate standard normal samples
            Z = rng.standard_normal((lam, dim))
            
            # Transform to Y
            # Y = eigvecs @ (sqrt_eigvals * Z.T).T
            Y = (eigvecs * sqrt_eigvals) @ Z.T
            
            # X = m + sigma * Y
            X = m + sigma * Y.T
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # Fill remaining with worst value? Or break?
                    # If we break mid-evaluation, we can't update properly.
                    # Better to just stop the loop if we hit deadline during eval.
                    # But we need to finish the current batch to update? 
                    # Actually, if we run out of time, we should just return best.
                    # Let's check time before evaluating each point.
                    pass
                F[i] = func(X[i])
                eval_count += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                    
            if time.time() > deadline:
                break
                
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            m_new = np.dot(weights, X_sorted[:mu])
            
            # Update path states
            inv_sqrt_C = (eigvecs / sqrt_eigvals).T # This is (C^-1/2)
            # Note: Y = C^-1/2 @ (X - m) / sigma is not exactly right because we clipped X.
            # CMA-ES assumes Gaussian sampling. Clipping distorts the distribution.
            # However, for box constraints, clipping is a common heuristic.
            # A more rigorous approach would be to project or use a different distribution, 
            # but clipping is standard for simple implementations.
            
            # To update pc and ps, we need the "natural" coordinates.
            # Ideally, we should use the unclipped Y for the state updates if possible, 
            # but the fitness is of the clipped X.
            # Let's use the unclipped Y for the updates to maintain the CMA-ES dynamics,
            # assuming the clipping is a minor perturbation or that the optimum is interior.
            # If the optimum is on the boundary, this might be slightly suboptimal, but robust.
            
            Y_sorted = Y[order] # Unclipped Y
            
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ (m_new - m) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (eval_count // lam)))) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs)**(2 * (eval_count // lam)))
            
            # pc update
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - m) / sigma
            
            # Update C
            # Rank-one update
            C = (1 + c1 * hsig * (cc - 1) - c1) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            # Y_mu = (X_sorted[:mu] - m) / sigma
            # We should use the actual X-m, but we need to map it back to the eigenbasis?
            # The standard update uses the actual step vectors.
            # Let's use the unclipped Y for consistency with the sampling distribution.
            Y_mu = Y_sorted[:mu]
            C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * Y_mu[:, np.newaxis, :] * Y_mu[:, np.newaxis, :].T, axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            # chiN = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            
            # Update mean
            m = m_new
            
            # Check convergence
            if sigma < 1e-8:
                break
            if np.linalg.norm(m - m_new) < 1e-8 * (1 + np.abs(m)):
                break

        return best_x, best_f
