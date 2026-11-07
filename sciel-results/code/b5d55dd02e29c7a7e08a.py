import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        # CMA-ES typically uses lambda = 4 + floor(3 * log(n))
        # For n=4, lambda = 4 + 3*1.38 = 8. Let's use 10 for better exploration.
        lambda_pop = 10
        
        # Initialize CMA-ES parameters
        # Start at the center of the box
        mean = (lo + hi) / 2.0
        # Step size: fraction of the box size
        sigma = 0.5 * np.mean(hi - lo)
        # Covariance matrix: Identity
        C = np.eye(dim)
        # Inverse square root of C (initially I)
        C_inv_sqrt = np.eye(dim)
        
        # Parameters for CMA-ES update
        # Standard CMA-ES parameters for small dim
        mu = lambda_pop // 2  # Number of parents
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        X = mean + sigma * (C_inv_sqrt @ np.random.randn(lambda_pop, dim).T).T
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        evals = lambda_pop
        
        # Archive of all non-dominated points found so far
        # We will return the best set. Since the evaluator records EVERY returned objective vector,
        # we need to be careful. The contract says: "Every returned objective vector is recorded".
        # This implies that if we return a set of points, all of them are added to the archive.
        # If we return the same point multiple times, it's likely deduplicated or just added.
        # To maximize hypervolume, we want the final returned set to be a good approximation of the Pareto front.
        # However, the evaluator calls __call__ ONCE. So we just return the final population or a subset.
        # The current program returns X (the final population).
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Select non-dominated points from the current population
            # For efficiency, we can use a simple non-dominated sort
            # Since lambda is small (10), O(N^2) is fine
            nd_indices = self._non_dominated_indices(F)
            
            if len(nd_indices) == 0:
                # Fallback: use all
                nd_indices = list(range(lambda_pop))
            
            # Sort non-dominated points by distance to the ideal point (or similar) for selection?
            # In standard CMA-ES, we select the best mu individuals.
            # For MO, we can select the non-dominated ones. If there are more than mu, we need to choose.
            # Let's just use the non-dominated ones as the "parents" for the mean update.
            
            # Update Mean
            # Use the centroid of the non-dominated points
            if len(nd_indices) > 0:
                new_mean = np.mean(X[nd_indices], axis=0)
            else:
                new_mean = mean
            
            # Update Step Size
            # Calculate the normalized step
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            # Note: The formula usually uses the weighted sum of the selected parents.
            # For MO, this is tricky. Let's simplify:
            # Use the direction from old mean to new mean.
            delta_mean = new_mean - mean
            # Normalize by sigma
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta_mean) / sigma
            
            # Check for step size adaptation
            # If ||p_sigma|| is large, increase sigma, otherwise decrease
            # chi_n is the expected norm of N(0, I)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update Covariance
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ delta_mean / sigma
            
            # Rank-one and rank-mu update
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(evals/lambda)}) < (1.4 + 2/(dim+1)) * chi_n
            # This is getting complex and prone to bugs with small budgets.
            
            # Alternative: Simpler Covariance Update
            # Just update C to match the covariance of the non-dominated points.
            # This is a "Covariance Matrix Adaptation" in the most direct sense.
            if len(nd_indices) > 1:
                X_nd = X[nd_indices]
                # Center
                X_centered = X_nd - new_mean
                # Covariance
                C_new = (X_centered.T @ X_centered) / (len(nd_indices) - 1)
                # Add regularization to keep it positive definite and not too small
                C_new += 1e-6 * np.eye(dim)
                # Blend with old C to avoid instability
                C = 0.5 * C + 0.5 * C_new
                # Update C_inv_sqrt
                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8) # Ensure positive
                C_inv_sqrt = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals))) @ eigvecs.T
            else:
                # If only one point, keep C
                pass
            
            # Update Mean
            mean = new_mean
            
            # Generate new population
            X = mean + sigma * (C_inv_sqrt @ np.random.randn(lambda_pop, dim).T).T
            X = np.clip(X, lo, hi)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            evals += lambda_pop
            
        # Final Selection: Return the non-dominated points from the last population?
        # Or maintain an archive?
        # The contract says "Every returned objective vector is recorded".
        # If I return the final population X, only those points are recorded.
        # To get a better front, I should return the best non-dominated points found so far.
        # But I only have the current F. I didn't keep an archive.
        # Let's keep an archive of all non-dominated points.
        
        # Wait, the loop above overwrites X and F. I need to maintain an archive.
        # Let's refactor to maintain an archive.
        
        # RE-IMPLEMENTATION WITH ARCHIVE
        
        # Reset state
        t_start = time.time()
        deadline = t_start + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        lambda_pop = 10
        
        mean = (lo + hi) / 2.0
        sigma = 0.5 * np.mean(hi - lo)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        mu_eff = 5.0 # Fixed for simplicity
        
        # Archive: list of (x, f)
        archive = []
        
        # Initial population
        X = mean + sigma * (C_inv_sqrt @ np.random.randn(lambda_pop, dim).T).T
        X = np.clip(X, lo, hi)
        F = np.array([problem(x) for x in X])
        evals = lambda_pop
        
        # Update archive
        for i in range(len(X)):
            self._update_archive(archive, X[i], F[i])
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated points from current population
            nd_indices = self._non_dominated_indices(F)
            
            if len(nd_indices) == 0:
                nd_indices = list(range(lambda_pop))
            
            new_mean = np.mean(X[nd_indices], axis=0)
            delta_mean = new_mean - mean
            
            # Update p_sigma
            p_sigma = (1 - 0.5) * p_sigma + np.sqrt(0.5 * 1.5 * mu_eff) * (C_inv_sqrt @ delta_mean) / sigma
            chi_n = np.sqrt(dim)
            sigma *= np.exp(0.5 / 1.0 * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update Covariance based on non-dominated points
            if len(nd_indices) > 1:
                X_nd = X[nd_indices]
                X_centered = X_nd - new_mean
                C_new = (X_centered.T @ X_centered) / (len(nd_indices) - 1)
                C_new += 1e-6 * np.eye(dim)
                C = 0.7 * C + 0.3 * C_new
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                C_inv_sqrt = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals))) @ eigvecs.T
            else:
                # Keep C
                pass
            
            mean = new_mean
            
            # Generate new population
            X = mean + sigma * (C_inv_sqrt @ np.random.randn(lambda_pop, dim).T).T
            X = np.clip(X, lo, hi)
            
            F = np.array([problem(x) for x in X])
            evals += lambda_pop
            
            # Update archive
            for i in range(len(X)):
                self._update_archive(archive, X[i], F[i])
        
        # Return the archive points
        if archive:
            return np.array([p[0] for p in archive])
        else:
            return X

    def _non_dominated_indices(self, F):
        """Returns indices of non-dominated points in F."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _update_archive(self, archive, x, f):
        """Adds (x, f) to archive if it is not dominated by any existing point."""
        # Check if f is dominated by any point in archive
        for ax, af in archive:
            if np.all(af <= f) and np.any(af < f):
                return
        
        # Check if f dominates any point in archive, and remove them
        new_archive = []
        for ax, af in archive:
            if np.all(f <= af) and np.any(f < af):
                continue
            new_archive.append((ax, af))
        
        new_archive.append((x, f))
        archive[:] = new_archive
