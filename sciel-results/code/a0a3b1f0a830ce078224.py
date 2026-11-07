"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Gaussian Process Surrogate with UCB Acquisition for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        
        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            if call_count >= budget:
                return np.inf
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Gaussian Process Implementation ---
        
        def kernel(x1, x2, length_scales, variance):
            # RBF Kernel
            # x1: (N, D), x2: (M, D)
            # Returns (N, M)
            # Normalize lengths for stability
            ls = length_scales
            diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2 / ls**2, axis=2)
            return variance * np.exp(-0.5 * sq_dist)

        def predict_gp(X_train, Y_train, X_pred, length_scales, variance, noise):
            """
            Predict mean and variance for GP.
            X_train: (N, D)
            Y_train: (N,)
            X_pred: (M, D)
            """
            K = kernel(X_train, X_train, length_scales, variance)
            K += noise * np.eye(len(X_train))
            
            # Solve K * alpha = Y
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                # Fallback to regularized solve if Cholesky fails
                K_reg = K + 1e-6 * np.eye(len(X_train))
                alpha = np.linalg.solve(K_reg, Y_train)

            K_s = kernel(X_train, X_pred, length_scales, variance)
            # K_s is (N, M)
            
            mu = K_s.T @ alpha
            
            # Variance calculation
            # v = L^-1 K_s
            try:
                v = np.linalg.solve(L, K_s)
                var = variance - np.sum(v**2, axis=0)
            except np.linalg.LinAlgError:
                # If Cholesky failed, we can't easily get variance from L.
                # Approximate variance or use a simplified version.
                # For robustness, let's just return a constant variance if Cholesky fails repeatedly
                # Or recompute with regularization
                K_reg = K + 1e-6 * np.eye(len(X_train))
                # This is getting complex. Let's stick to the Cholesky path and assume it works for small N.
                # If it fails, we might have numerical issues.
                # Let's add a small jitter to K in the main loop to prevent this.
                var = np.full(len(X_pred), variance) # Fallback: assume max uncertainty

            var = np.maximum(var, 1e-10) # Numerical stability
            return mu, var

        # Hyperparameters
        # Initial length scales: assume features are independent and similar scale
        length_scales = np.ones(dim) * 1.0
        variance = 1.0
        noise = 1e-6
        
        # Initialize with a few points to establish a baseline
        # Use a small Sobol-like or random set for initialization
        # To ensure we cover the space, let's use a deterministic grid for the first 5-10 points?
        # Or just random? Random is safer for unknown shifts.
        
        # Strategy:
        # 1. Initial random sampling (e.g., 10 points) to build initial GP.
        # 2. Iteratively select point maximizing UCB = mu + beta * sigma.
        # 3. Update GP.
        # 4. Repeat until budget exhausted.
        
        # Initial Points
        n_init = 10
        np.random.seed(42)
        
        X_data = []
        Y_data = []
        
        for _ in range(n_init):
            if time.time() > deadline or call_count >= budget:
                break
            x_new = np.random.uniform(-5, 5, dim)
            f_new = evaluate(x_new)
            X_data.append(x_new)
            Y_data.append(f_new)
            
        if not X_data:
            return best_x, best_f

        X_train = np.array(X_data)
        Y_train = np.array(Y_data)
        
        # Adaptive Beta for UCB
        # Start with high exploration, decay over time
        # beta = log(t) / (2 * log(2 * dim * t)) ? Or simpler: beta = 1.0 initially, decay.
        # Let's use a simple schedule: beta = 1.0 / (1 + 0.1 * t) or similar.
        # Actually, standard UCB uses beta_t.
        # Let's try a fixed beta first, or a decaying one.
        # For 500 budget, beta=1.0 is a good start.
        
        current_iter = n_init
        
        while call_count < budget and time.time() < deadline:
            # Generate a candidate pool for the next point
            # Instead of a single point, generate a batch of candidates and pick the best one
            # This is more robust than picking a single random point and hoping it's good.
            # Generate 50-100 random candidates
            n_candidates = 100
            candidates = np.random.uniform(-5, 5, (n_candidates, dim))
            
            # Predict mean and variance for candidates
            try:
                mu, var = predict_gp(X_train, Y_train, candidates, length_scales, variance, noise)
                sigma = np.sqrt(var)
            except Exception:
                # If GP prediction fails, fall back to random selection
                idx = np.random.choice(n_candidates)
                x_new = candidates[idx]
            else:
                # UCB Acquisition
                # beta controls exploration vs exploitation
                # Let's decay beta slightly to focus on exploitation later
                # beta = 1.0 / (1 + 0.01 * current_iter)
                beta = 1.0
                acq = mu + beta * sigma
                idx = np.argmax(acq)
                x_new = candidates[idx]
            
            # Evaluate the selected point
            f_new = evaluate(x_new)
            
            # Update GP data
            X_train = np.vstack([X_train, x_new])
            Y_train = np.append(Y_train, f_new)
            
            # Optional: Adapt Hyperparameters
            # For simplicity and speed, we might keep hyperparameters fixed or update them slowly.
            # Updating length scales via MLE is expensive.
            # Given the 300s limit and 500 calls, a simple fixed kernel is often sufficient 
            # if the scale is right. The box is [-5,5], so length scale 1.0 is reasonable.
            # If the function is very smooth, larger LS. If jagged, smaller.
            # Gallagher 21 peaks is multimodal.
            
            current_iter += 1
            
            # Early stopping if we found a very good value?
            # No, keep going to maximize AOCC (early progress counts).
            
            # Check for convergence of the GP?
            # If the max UCB is close to the best found value, we might be stuck.
            # But for AOCC, we just need to keep finding better values if possible.
            
            # To prevent infinite loops if GP breaks:
            if not np.isfinite(f_new):
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
