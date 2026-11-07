"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class Optimizer:
    """
    Multi-objective optimizer using Model-Based Evolutionary Algorithm (MBEA).
    
    Strategy:
    1. Initialize with a small set of random points to fit GP surrogates.
    2. Iteratively:
       a. Fit GP models for each objective.
       b. Generate a dense candidate grid.
       c. Predict mean and variance for each objective at candidates.
       d. Compute Expected Hypervolume Improvement (EHVI) for each candidate.
       e. Select the candidate with the highest EHVI.
       f. Evaluate the selected point and add to the archive.
    3. Return the archive of evaluated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume calculation
        # The task says reference point is [1.1, 1.3, 1.1]
        ref_point = np.array([1.1, 1.3, 1.1])

        # Initial population: 10 random points
        n_init = 10
        rng = np.random.default_rng(0)
        
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        # Ensure we have valid points (not dominated by ref, etc., though GP handles it)
        # Archive: list of (x, f) pairs
        archive_X = X_init.copy()
        archive_F = F_init.copy()
        
        # Filter out points that are worse than reference point in all objectives?
        # No, the task says "Only points strictly better than the reference point in every objective count."
        # But for GP fitting, we should probably keep all evaluated points to model the landscape.
        # However, if a point is terrible, it might skew the GP.
        # Let's keep all points for now.

        # Dense grid for acquisition function
        # In 4D, a full grid is expensive. Use Latin Hypercube Sampling for candidates.
        n_candidates = 2000
        
        # We have budget - n_init evaluations left.
        remaining_budget = self.budget - n_init
        
        # GP setup
        # Use a simple RBF kernel with auto-optimized length scales
        # To speed up, we might fix length scales or use a simpler kernel if time is tight.
        # But let's try auto for accuracy.
        
        # Pre-allocate storage for GP models
        gps = []
        
        for i in range(remaining_budget):
            if time.time() - start_time > max_time:
                break
                
            # 1. Fit GPs
            # Fit one GP per objective
            current_gps = []
            for k in range(self.n_obj):
                y = archive_F[:, k]
                # Normalize y to [0, 1] for GP stability? Or just use raw.
                # Raw is fine if scale is reasonable. RE37 objectives are usually in [0, 1] or similar.
                # Let's normalize to help GP convergence.
                y_min = y.min()
                y_max = y.max()
                if y_max > y_min:
                    y_norm = (y - y_min) / (y_max - y_min)
                else:
                    y_norm = y
                    y_min = 0
                    y_max = 1
                
                kernel = ConstantKernel(1.0) * RBF(length_scale=0.5)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
                gp.fit(archive_X, y_norm)
                current_gps.append((gp, y_min, y_max))
            
            # 2. Generate Candidates
            # Use LHS for diversity
            candidates = lo + (hi - lo) * rng.random((n_candidates, self.dim))
            
            # 3. Predict Mean and Variance
            # We need to predict for all objectives at all candidates.
            # This can be slow if done one by one.
            # GP.predict returns (mean, std).
            
            means = np.zeros((n_candidates, self.n_obj))
            stds = np.zeros((n_candidates, self.n_obj))
            
            for k in range(self.n_obj):
                gp, y_min, y_max = current_gps[k]
                mean_norm, std_norm = gp.predict(candidates, return_std=True)
                # Denormalize
                means[:, k] = mean_norm * (y_max - y_min) + y_min
                stds[:, k] = std_norm * (y_max - y_min)
            
            # 4. Compute EHVI
            # EHVI is complex to compute exactly. We can approximate it using a simplified metric.
            # A common approximation: 
            # For each candidate, estimate the hypervolume contribution if it were added to the archive.
            # This is O(N^2) per candidate, which is expensive.
            # Alternative: Use a scalarizing acquisition function like Expected Improvement (EI) on a weighted sum,
            # or a simple "uncertainty + improvement" metric.
            # Given the time constraints and complexity, let's use a simpler acquisition:
            # Maximize the minimum of the predicted improvements over objectives, weighted by uncertainty.
            # Or, better: Use the "Expected Hypervolume Improvement" approximation via sampling.
            
            # Let's try a simpler approach: 
            # For each candidate, compute the predicted hypervolume of the archive + this candidate.
            # This is still expensive.
            
            # Let's use a proxy: 
            # For each candidate, compute the weighted sum of objectives using random weights,
            # and compute the Expected Improvement (EI) for that scalar function.
            # This is similar to the current program but with GP-based EI.
            
            # Actually, let's stick to a robust, fast heuristic:
            # Select the candidate that maximizes the minimum predicted improvement over the current worst point in the archive?
            # No, that's not right.
            
            # Let's implement a fast EHVI approximation:
            # 1. Identify the non-dominated points in the archive.
            # 2. For each candidate, compute its hypervolume contribution to the archive.
            # 3. Add the uncertainty term.
            
            # To keep it fast, let's use a grid of weight vectors for scalarization.
            # This is a compromise.
            
            # Let's try a different, faster acquisition: 
            # "Maximize the minimum distance to the current Pareto front in objective space, penalized by uncertainty".
            
            # Actually, the simplest effective MBOA acquisition is often just maximizing the predicted improvement of a weighted sum.
            # Let's use 10 random weight vectors and compute EI for each, then take the max.
            
            n_weights = 10
            weights = rng.random((n_weights, self.n_obj))
            weights /= weights.sum(axis=1, keepdims=True)
            
            # Current best scalar values for each weight
            # This is O(N * W)
            F_scalar = archive_F @ weights.T  # (N, W)
            best_f_scalar = F_scalar.min(axis=0)  # (W,)
            
            # Predicted scalar values for candidates
            # C_scalar = means @ weights.T (N_cand, W)
            # Std_scalar = sqrt(sum((stds * weights)^2)) (N_cand, W)
            
            C_scalar = means @ weights.T
            # Approximate std of weighted sum
            std_scalar = np.sqrt(np.sum((stds * weights.T) ** 2, axis=1))
            
            # Compute EI for each weight and candidate
            # EI = (mu_best - mu_pred) * Phi((mu_best - mu_pred)/sigma) + sigma * phi((mu_best - mu_pred)/sigma)
            # Note: We are minimizing, so mu_best is the minimum.
            # If mu_pred < mu_best, improvement is mu_best - mu_pred.
            
            from scipy.stats import norm
            
            Z = (best_f_scalar[None, :] - C_scalar) / (std_scalar[:, None] + 1e-8)
            EI = (best_f_scalar[None, :] - C_scalar) * norm.cdf(Z) + std_scalar[:, None] * norm.pdf(Z)
            
            # Max EI over weights
            max_ei = EI.max(axis=1)
            
            # Select best candidate
            best_idx = np.argmax(max_ei)
            x_new = candidates[best_idx]
            
            # 5. Evaluate
            f_new = problem(x_new)
            
            # 6. Update Archive
            archive_X = np.vstack([archive_X, x_new])
            archive_F = np.vstack([archive_F, f_new])
            
            # Optional: Remove dominated points from archive to keep it small?
            # For GP fitting, more data is better, but dominated points might be noise.
            # Let's keep all points for now to ensure robust GP fitting.
            # If the archive gets too large (e.g., > 50), we might subsample.
            if len(archive_X) > 50:
                # Subsample to 50 points, ensuring non-dominated points are kept.
                # Simple heuristic: keep the best 50 points by sum of objectives.
                sums = archive_F.sum(axis=1)
                top_indices = np.argsort(sums)[:50]
                archive_X = archive_X[top_indices]
                archive_F = archive_F[top_indices]

        return archive_X

# EVOLVE-BLOCK-END
