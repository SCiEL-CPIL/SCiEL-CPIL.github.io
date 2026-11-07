"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Multi-objective optimizer using Gaussian Process surrogate modeling (MOBO)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Normalization of bounds to [0, 1] for GP stability
        x_min = lo
        x_max = hi
        x_range = x_max - x_min
        
        # Initial sampling: Latin Hypercube Sampling for good coverage
        n_init = min(20, self.budget)
        rng = np.random.default_rng(42)
        X_init = rng.random((n_init, dim))
        
        # Evaluate initial points
        X_eval = []
        F_eval = []
        
        for i in range(n_init):
            x = x_min + X_init[i] * x_range
            f = problem(x)
            X_eval.append(x)
            F_eval.append(f)
            
            if time.time() > deadline:
                break
                
        X_eval = np.array(X_eval)
        F_eval = np.array(F_eval)
        
        if len(X_eval) == 0:
            # Fallback if initial evals fail or timeout immediately
            x = (x_min + x_max) / 2
            f = problem(x)
            return [x]
            
        # Normalize objectives for GP (mean 0, std 1)
        F_mean = np.mean(F_eval, axis=0)
        F_std = np.std(F_eval, axis=0)
        F_std[F_std < 1e-8] = 1.0
        F_norm = (F_eval - F_mean) / F_std
        
        # Gaussian Process Kernel (Matern 5/2)
        def kernel(X1, X2, length_scale):
            dist = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))
            sqrt5_dist = np.sqrt(5) * dist / length_scale
            k = (1 + sqrt5_dist) * np.exp(-sqrt5_dist)
            return k

        # Hyperparameters: length_scale, signal_variance, noise_variance
        # We will optimize these roughly or use fixed reasonable values for speed
        # Given the small budget (200), we can't do heavy hyperparameter optimization.
        # We'll use a fixed set of hyperparameters tuned for general smoothness.
        # For RE34, objectives are likely smooth.
        length_scale = 0.5
        signal_var = 1.0
        noise_var = 0.01
        
        # Build GP model
        X_gp = X_eval # Keep original scale? No, normalize input to [0,1] for GP
        X_norm = (X_eval - x_min) / x_range
        
        K = kernel(X_norm, X_norm, length_scale) + noise_var * np.eye(len(X_norm))
        L = np.linalg.cholesky(K)
        
        # Precompute inverse
        K_inv = np.linalg.inv(K)
        
        # Acquire function: Expected Improvement on the Pareto Front (approximated)
        # Since we have 3 objectives, we use a simple strategy:
        # 1. Find non-dominated points in current set.
        # 2. For each new point, estimate the probability of being non-dominated or improving HV.
        # A simpler proxy for MOBO with small budgets is to maximize the minimum distance
        # to existing points (exploration) weighted by the uncertainty (exploitation).
        # However, a better approach for HV is to sample from the GP and check non-domination.
        
        # Let's use a simple acquisition: 
        # For a grid of candidate points, compute the GP mean and variance.
        # We want points that are likely to be non-dominated.
        # A common heuristic is to maximize the sum of "improvement" over the current worst point in each objective, 
        # but that's complex. 
        # Let's try: Sample 100 candidates from a grid + random.
        # For each candidate, predict F. 
        # Compute the "hypervolume contribution" of this point to the current front.
        # This is expensive. 
        # Alternative: Use a simple "max-min" diversity metric combined with uncertainty.
        
        # Let's implement a simpler, robust acquisition:
        # 1. Generate a set of candidate points (e.g., 50 random + 50 local perturbations of best points).
        # 2. For each candidate, predict the objective vector using the GP.
        # 3. Calculate the "expected hypervolume improvement" (EHVI) is too slow.
        # 4. Instead, use a proxy: The candidate is good if its predicted values are lower than the current non-dominated front in at least one objective, and not worse in all.
        # 5. Even simpler: Select the point that maximizes the minimum distance to the existing Pareto front in objective space, weighted by uncertainty.
        
        # Let's stick to a simpler, effective strategy for small budgets:
        # "Greedy Hypervolume Maximization" with GP guidance.
        # 1. Identify current Pareto Front (PF).
        # 2. Generate candidates.
        # 3. For each candidate, predict F.
        # 4. Add predicted F to PF. Calculate new HV.
        # 5. Select candidate with max HV gain.
        
        # To make this fast, we limit the candidate set size.
        
        def get_pareto_front(F):
            mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        def calculate_hv(F, ref_point):
            # Simple HV calculation for 3D
            # Filter points better than ref
            F = F[np.all(F < ref_point, axis=1)]
            if len(F) == 0:
                return 0.0
            # Sort by first objective
            idx = np.argsort(F[:, 0])
            F = F[idx]
            # For each point, find the volume it contributes
            # This is an O(N^2) or O(N^3) algorithm. For small N (<100), it's fine.
            # We'll use a simple inclusion-exclusion or just sum of boxes for small N approximation?
            # No, exact HV is required for scoring, but for acquisition we can approximate.
            # For 3D, we can sort by x, then for each x, find the max y and z.
            # Actually, let's just use a simple volume estimation for the acquisition function speed.
            # Or, since N is small, do the exact calculation.
            
            # Exact 3D HV:
            # Sort by x
            # Maintain a structure of (y, z) points that are non-dominated in the remaining space.
            # This is complex to implement correctly in a few lines.
            # Let's use a simpler proxy for acquisition: 
            # Score = - (sum of predicted objectives) + alpha * (sum of predicted variances)
            # This encourages exploration of uncertain regions that might be better.
            
            # Let's go with the proxy score for speed.
            return 0.0 # Placeholder, we won't use exact HV in the loop for speed

        # Revised Strategy:
        # Use a simpler acquisition: 
        # 1. Predict F and variance for a set of candidates.
        # 2. Score = - (weighted sum of predicted F) + beta * (weighted sum of predicted variance).
        # This is similar to UCB (Upper Confidence Bound).
        
        beta = 2.0 # Exploration/exploitation trade-off
        
        # Main Loop
        while len(X_eval) < self.budget:
            if time.time() > deadline:
                break
                
            # Generate candidates
            n_candidates = 20
            # Mix of random and local search around best points
            # Best points = non-dominated
            pf_mask = get_pareto_front(F_eval)
            pf_indices = np.where(pf_mask)[0]
            
            candidates = []
            # 10 random
            X_rand = rng.random((10, dim))
            candidates.append(X_rand)
            
            # 10 local perturbations of PF points
            if len(pf_indices) > 0:
                pf_X = X_eval[pf_indices]
                # Randomly select 10 parents
                parents = pf_X[rng.integers(0, len(pf_X), size=10)]
                # Perturb
                perturb = rng.standard_normal((10, dim)) * 0.1
                X_local = parents + perturb
                X_local = np.clip(X_local, 0, 1)
                candidates.append(X_local)
            
            candidates = np.vstack(candidates)
            
            # Predict F and Var for candidates
            X_cand_norm = candidates # Already in [0,1]
            
            K_c = kernel(X_cand_norm, X_norm, length_scale)
            K_c_var = K_c
            
            # Mean and Variance
            # Mean = K_c @ K_inv @ F_norm
            mean_pred = K_c @ (K_inv @ F_norm)
            
            # Variance = signal_var - K_c @ K_inv @ K_c.T
            # K_c is (n_cand, n_data), K_inv is (n_data, n_data)
            # K_c @ K_inv is (n_cand, n_data)
            # (K_c @ K_inv) @ K_c.T is (n_cand, n_cand)
            # We only need the diagonal (variance for each candidate)
            K_inv_K_c = K_inv @ K_c.T # (n_data, n_cand)
            var_pred = signal_var - np.sum(K_c * K_inv_K_c.T, axis=1)
            var_pred = np.clip(var_pred, 0, None)
            std_pred = np.sqrt(var_pred)
            
            # Denormalize predictions
            F_pred = mean_pred * F_std + F_mean
            std_pred = std_pred * F_std # Scale uncertainty
            
            # Acquisition Score
            # We want to minimize F. So lower F_pred is better.
            # Higher std_pred is better (uncertainty).
            # Score = - sum(F_pred) + beta * sum(std_pred)
            # Normalize F_pred and std_pred to comparable scales? 
            # F_mean and F_std are already used.
            # Let's just use the raw values.
            
            scores = -np.sum(F_pred, axis=1) + beta * np.sum(std_pred, axis=1)
            
            # Select best candidate
            best_idx = np.argmax(scores)
            x_new_norm = candidates[best_idx]
            x_new = x_min + x_new_norm * x_range
            
            # Evaluate
            f_new = problem(x_new)
            
            # Update Data
            X_eval = np.vstack([X_eval, [x_new]])
            F_eval = np.vstack([F_eval, [f_new]])
            
            # Update GP Model
            # This requires rebuilding K and L.
            # For speed, we can just rebuild it. N is small (<200).
            X_norm = (X_eval - x_min) / x_range
            K = kernel(X_norm, X_norm, length_scale) + noise_var * np.eye(len(X_norm))
            try:
                L = np.linalg.cholesky(K)
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                # If singular, add small jitter
                K += 1e-6 * np.eye(len(X_norm))
                try:
                    L = np.linalg.cholesky(K)
                    K_inv = np.linalg.inv(K)
                except np.linalg.LinAlgError:
                    break # Can't update, stop
            
            # Update F_norm
            F_mean = np.mean(F_eval, axis=0)
            F_std = np.std(F_eval, axis=0)
            F_std[F_std < 1e-8] = 1.0
            F_norm = (F_eval - F_mean) / F_std
            
        return X_eval

# EVOLVE-BLOCK-END
