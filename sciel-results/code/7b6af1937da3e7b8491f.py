"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-Local Hybrid with RBF Surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325  # Safety margin before 330s kill
        n = self.dim
        B = self.budget
        rng = np.random.default_rng(42)

        # Data storage
        X = []
        F = []
        best_x = None
        best_f = np.inf
        calls = 0

        def add_point(x, f):
            nonlocal best_x, best_f, calls
            X.append(x)
            F.append(f)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Global Exploration with Sobol-like or Latin Hypercube
        # For n=5, we can do a quick LHS to cover the space
        # Let's use a stratified sampling approach
        n_init = 20
        if B >= n_init:
            # Create LHS points
            # Stratify each dimension into n_init intervals
            # For simplicity, use random permutation per dimension
            X_lhs = np.zeros((n_init, n))
            for d in range(n):
                perm = rng.permutation(n_init)
                X_lhs[:, d] = (perm + 0.5) / n_init * 10.0 - 5.0
            
            for x in X_lhs:
                if calls >= B:
                    break
                fx = func(x)
                add_point(x, fx)

        # Phase 2: Surrogate-guided Search
        # We will use an RBF surrogate to guide the search.
        # Parameters:
        # - Use all points so far for fitting
        # - Acquisition: Expected Improvement (EI)
        # - Candidate generation: Mix of local perturbations around best and global random
        
        # RBF Kernel: Gaussian
        # K(x, x') = exp(-gamma * ||x - x'||^2)
        # We need to choose gamma. A common heuristic is based on the average distance.
        # Or we can fix it. Given the box is [-5, 5]^5, typical distances are ~sqrt(5)*10/sqrt(3) ~ 12.
        # Let's try a fixed gamma first, or adapt it.
        
        def build_rbf(X_train, F_train, gamma=None):
            """Build RBF model and return prediction function."""
            X_train = np.array(X_train)
            F_train = np.array(F_train)
            n_pts = len(X_train)
            
            if n_pts == 1:
                return lambda x: F_train[0]
            
            # Compute kernel matrix
            # For efficiency, we compute distances on the fly during prediction
            # But for fitting, we need the matrix
            diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            if gamma is None:
                # Heuristic gamma
                avg_dist_sq = np.mean(dist_sq[dist_sq > 1e-8]) if np.any(dist_sq > 1e-8) else 1.0
                gamma = 1.0 / (2.0 * avg_dist_sq * 0.1) # Heuristic: wider kernel
                # Cap gamma to avoid numerical issues
                gamma = min(gamma, 10.0)
                gamma = max(gamma, 0.01)
            
            K = np.exp(-gamma * dist_sq)
            
            # Add ridge for stability
            K += 1e-8 * np.eye(n_pts)
            
            # Solve for weights
            # K @ w = F_train
            try:
                w = np.linalg.solve(K, F_train)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K, F_train, rcond=None)[0]
            
            def predict(X_pred):
                """Predict mean and variance (approx)."""
                X_pred = np.array(X_pred)
                if X_pred.ndim == 1:
                    X_pred = X_pred[np.newaxis, :]
                
                diff = X_pred[:, np.newaxis, :] - X_train[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=2)
                K_pred = np.exp(-gamma * dist_sq)
                
                mean = K_pred @ w
                # Variance approximation: 1 - K_pred @ K^{-1} @ K_pred.T
                # This is expensive. We can approximate variance by distance to nearest point or kernel value.
                # For EI, we need std.
                # Let's use a simple heuristic for std: 
                # std = sqrt(1 - max(K_pred)) * (std(F_train) + 1)
                # Or just use a fixed exploration term that decays?
                # Better: Use the diagonal of the covariance.
                # var_i = 1 - sum_j K_ij * (K^{-1} K_ij)
                # This requires solving K^{-1} K_pred.T
                # Let's precompute K_inv or use Cholesky.
                
                # For now, let's just return mean and a rough variance estimate
                # Rough variance: based on distance to nearest training point
                min_dist_sq = np.min(dist_sq, axis=1)
                # If we are close to a point, variance is low. If far, variance is high.
                # Map min_dist_sq to variance
                # var = sigma_f^2 * (1 - exp(-gamma * min_dist_sq))
                sigma_f = np.std(F_train) + 1e-3
                var = sigma_f**2 * (1 - np.exp(-gamma * min_dist_sq))
                # Also, if we are far from all points, variance should be high.
                # The above formula gives var -> sigma_f^2 as dist -> inf.
                
                return mean, var

            return predict, gamma

        # Acquisition Function: Expected Improvement
        def expected_improvement(mean, var, f_best, xi=0.01):
            """
            EI = (f_best - mean - xi) * Phi(z) + std * phi(z)
            where z = (f_best - mean - xi) / std
            """
            std = np.sqrt(np.maximum(var, 1e-8))
            z = (f_best - mean - xi) / std
            # Phi and phi
            # Using scipy.special if available, otherwise manual
            try:
                from scipy.special import erf, ndtr
                # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
                # phi(z) = 1/sqrt(2*pi) * exp(-z^2/2)
                phi = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * z**2)
                Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
            except ImportError:
                # Fallback
                Phi = ndtr(z) if 'ndtr' in dir() else 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0))) # Approx
                phi = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * z**2)
            
            ei = (f_best - mean - xi) * Phi + std * phi
            return ei

        # Main Loop
        while calls < B:
            if time.time() - start_time > time_limit:
                break
            
            # Update surrogate
            if len(X) < 2:
                # Need at least 2 points for RBF
                x_new = np.random.uniform(-5, 5, n)
                f_new = func(x_new)
                add_point(x_new, f_new)
                continue
            
            try:
                predict, gamma = build_rbf(X, F)
            except Exception:
                # Fallback to random if RBF fails
                x_new = np.random.uniform(-5, 5, n)
                f_new = func(x_new)
                add_point(x_new, f_new)
                continue
            
            # Generate candidates
            n_candidates = 200
            candidates = []
            
            # 1. Local perturbations around best point
            for _ in range(n_candidates // 2):
                noise = rng.standard_normal(n) * 0.5
                x_cand = best_x + noise
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            # 2. Random global points
            for _ in range(n_candidates // 2):
                x_cand = rng.uniform(-5, 5, n)
                candidates.append(x_cand)
            
            # Evaluate candidates with surrogate
            candidates = np.array(candidates)
            means, vars_ = predict(candidates)
            
            # Calculate EI
            ei_values = expected_improvement(means, vars_, best_f)
            
            # Select best candidate
            idx_best = np.argmax(ei_values)
            x_new = candidates[idx_best]
            
            # Evaluate true function
            f_new = func(x_new)
            add_point(x_new, f_new)
            
            # Optional: If we have many points, we can reduce the number of candidates
            # or adjust the balance between local and global.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
