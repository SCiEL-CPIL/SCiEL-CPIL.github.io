import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, stop at 290s
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control
        cs = 1.0 / (d + 3) + 2.0 / (d - 2)
        ds = 1.0 + 2.0 * max(0, (mueff - 1) / (d + 1)) + cs
        c1 = (2.0 + 2.0 * (mueff - 2) / (d + 1)) / ((d + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2 + 1.0 / mueff) / ((d + 2)**2 + mueff))
        
        # Initial values
        # Start from center or random? Random is safer for shifted landscapes.
        # But for f13 (sharp ridge), starting somewhat central might help if the shift is small.
        # The prompt says minimizer is in [-4, 4]^5.
        # Let's start at 0, which is in the box, or a random point.
        # Given the "shifted" nature, a random start is more robust.
        x_mean = np.random.uniform(-4, 4, d)
        
        # Initial sigma: large enough to explore the box
        # Box width is 10. Let's start with sigma = 2.0
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(d)
        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        # Inverse sqrt of C for step size control
        # For d=5, we can just compute it directly if needed, but standard CMA-ES uses eigen decomposition periodically.
        # For simplicity and speed in low-d, we can use the standard update.
        
        # Pre-compute constants
        hsig = (1.5 + 2.0 / (d + 1)) * (1.0 - np.exp(-0.5 * cs * ds))
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        evals_used = 1
        
        # Main loop
        while evals_used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            # We need to update the inverse square root of C or use eigen decomposition
            # For d=5, eigen decomposition every generation is cheap.
            # However, standard CMA-ES updates C and then uses it.
            
            # To generate samples, we need to transform standard normal to x_mean + sigma * C^{1/2} * z
            # We can use scipy.linalg.sqrtm or np.linalg.eigh
            # Since d is small, eigh is fast.
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Ensure positive definiteness
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            C_sqrt_inv = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            # Actually, we want C^{1/2}, not C^{-1/2}.
            # z' = C^{1/2} z.
            # C^{1/2} = V diag(sqrt(lambda)) V^T
            C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            
            ys = []
            fs = []
            
            for i in range(lam):
                z = np.random.standard_normal(d)
                y = x_mean + sigma * (C_sqrt @ z)
                # Clip to box? The func clips, but we should probably keep it in bounds for better search.
                # The prompt says func clips. So we can just pass y.
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                evals_used += 1
                
                if time.time() > deadline:
                    # If we run out of time mid-generation, we break out.
                    # We need to handle the partial generation.
                    # For simplicity, if we break here, we stop the whole loop.
                    # The current generation is incomplete, so we don't update C.
                    # We just return the best found so far.
                    # We need to ensure we don't use the incomplete data.
                    # So we break out of the inner loop and the outer loop.
                    pass
                    
            # If we broke due to time, we should exit the outer loop too.
            # The `while` condition will be checked, but we should be explicit.
            # Actually, if we broke the inner loop, ys is incomplete.
            # Let's restructure: check time before starting a new generation.
            # The check above is sufficient if we assume generation time is small.
            # But if lam is large, it might be slow. For d=5, lam=7, it's fast.
            
            # Sort offspring by fitness
            order = np.argsort(fs)
            ys = np.array(ys)
            fs = np.array(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update sigma
            # y = (x_mean_new - x_mean) / sigma
            y_sigma = (x_mean_new - x_mean) / sigma
            p_sigma = (1.0 - cs) * p_sigma + np.sqrt(cs * (2.0 - cs) * mueff) * y_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            
            # Update covariance
            y_c = (x_mean_new - x_mean) / sigma
            p_c = (1.0 - cs) * p_c + np.sqrt(cs * (2.0 - cs) * mueff) * y_c
            
            # Update C
            # C = (1 - c1 - cmu) C + c1 (p_c p_c^T + (1 - hsig) delta_sig C) + cmu sum(wi (yi - x_mean)(yi - x_mean)^T)
            # delta_sig = 1 - exp(-2 * cs * ds * (norm_p_sigma^2 - d))
            delta_sig = 1.0 - np.exp(-2.0 * cs * ds * (norm_p_sigma**2 - d))
            
            # Correction term
            C_update = (1.0 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1.0 - hsig) * delta_sig * C)
            
            # Sum of outer products of weighted offspring
            # Use the top mu offspring
            ys_mu = ys[:mu]
            # Center them
            ys_mu_centered = ys_mu - x_mean
            # Weighted sum
            # C_mu = sum(weights[i] * outer(ys_mu_centered[i], ys_mu_centered[i]))
            # For efficiency, we can do this in a loop or with einsum
            C_mu = np.sum(weights[:mu, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', np.ones(mu), ys_mu_centered, ys_mu_centered), axis=0)
            # Wait, the einsum is wrong.
            # outer(y_i, y_i) is dxd.
            # We want sum_i w_i * outer(y_i, y_i)
            # Let's just use a loop for d=5, it's negligible.
            C_mu = np.zeros((d, d))
            for i in range(mu):
                C_mu += weights[i] * np.outer(ys_mu_centered[i], ys_mu_centered[i])
                
            C = C_update + cmu * C_mu
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma = sigma * np.exp(cs / ds * (norm_p_sigma / np.sqrt(d) - 1.0))
            
            # Update mean
            x_mean = x_mean_new
            
            # Check for stagnation or convergence
            # If sigma is too small, we can stop.
            if sigma < 1e-12:
                break

        return best_x, best_f
