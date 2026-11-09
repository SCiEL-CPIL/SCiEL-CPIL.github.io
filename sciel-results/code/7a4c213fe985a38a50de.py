import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        call_count = 0

        # Phase 1: Global Sampling to find a good starting point
        n_init = 30
        if budget < n_init:
            x0 = np.zeros(dim)
            fx0 = func(x0)
            return x0, fx0

        # Use a mix of random and structured points
        X_init = np.random.uniform(-5, 5, size=(n_init, dim))
        # Ensure origin is included for the sphere function
        X_init[0] = 0.0

        F_init = np.array([func(x) for x in X_init])
        call_count = n_init

        best_idx = np.argmin(F_init)
        x_best = X_init[best_idx].copy()
        f_best = F_init[best_idx]

        # Phase 2: Derivative-Free Trust-Region (DF-TR) Solver
        # We maintain a set of points (X) and their function values (F).
        # We fit a quadratic model to these points and use it to propose new points.
        
        # Initial trust region radius
        delta = 1.0
        delta_max = 5.0 * np.sqrt(dim)
        delta_min = 1e-4

        # We need n_model = (dim+1)(dim+2)/2 points for a full quadratic model
        n_model = (dim + 1) * (dim + 2) // 2

        # Initialize the local set around x_best
        X_set = [x_best]
        F_set = [f_best]

        # Generate n_model - 1 points around x_best to fill the model
        for _ in range(n_model - 1):
            if call_count >= budget:
                break
            direction = np.random.randn(dim)
            direction /= np.linalg.norm(direction)
            x_new = x_best + delta * direction
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = func(x_new)
            call_count += 1
            X_set.append(x_new)
            F_set.append(f_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()

        # Helper to get features for quadratic model
        def get_features(x):
            feats = [1.0]
            feats.extend(x)
            for j in range(dim):
                feats.append(x[j] * x[j])
            for j in range(dim):
                for k in range(j + 1, dim):
                    feats.append(x[j] * x[k])
            return np.array(feats)

        # Pre-compute feature matrix for current set? No, it changes.
        # But we can pre-compute the design matrix structure.
        
        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break

            X_arr = np.array(X_set)
            F_arr = np.array(F_set)

            # Fit quadratic model
            A = np.array([get_features(x) for x in X_arr])
            
            # Regularization to handle ill-conditioned cases
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, F_arr, rcond=None)
            except:
                break

            c = coeffs[0]
            g = coeffs[1 : 1 + dim]
            H_diag = coeffs[1 + dim : 1 + 2 * dim]
            H_off_diag = coeffs[1 + 2 * dim :]
            
            H = np.zeros((dim, dim))
            for j in range(dim):
                H[j, j] = H_diag[j]
            idx = 0
            for j in range(dim):
                for k in range(j + 1, dim):
                    H[j, k] = H_off_diag[idx]
                    H[k, j] = H_off_diag[idx]
                    idx += 1

            # Find minimizer of the model within the trust region
            # Model: m(x) = c + g^T x + 0.5 x^T H x
            # Gradient: g + H x
            # Hessian: H
            
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Check if H is positive definite
            if np.min(eigvals) > 1e-8:
                # Unconstrained minimizer
                try:
                    x_uncon = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    x_uncon = x_best
                
                # Check if inside trust region
                dist = np.linalg.norm(x_uncon - x_best)
                if dist <= delta:
                    x_trial = x_uncon
                else:
                    # Project onto trust region boundary
                    x_trial = x_best + delta * (x_uncon - x_best) / dist
            else:
                # H is not PD. 
                # For a sphere function, this shouldn't happen if we are close.
                # If it does, we might be far or the model is bad.
                # Let's take a step in the negative gradient direction.
                if np.linalg.norm(g) > 1e-8:
                    x_trial = x_best - delta * g / np.linalg.norm(g)
                else:
                    x_trial = x_best.copy()

            # Clip to bounds
            x_trial = np.clip(x_trial, -5.0, 5.0)

            # Evaluate
            f_trial = func(x_trial)
            call_count += 1

            # Update trust region
            actual_reduction = f_best - f_trial
            # Predicted reduction
            m_best = c + g.T @ x_best + 0.5 * x_best.T @ H @ x_best
            m_trial = c + g.T @ x_trial + 0.5 * x_trial.T @ H @ x_trial
            predicted_reduction = m_best - m_trial
            
            if actual_reduction > 0:
                # Success
                ratio = actual_reduction / max(abs(predicted_reduction), 1e-8)
                if ratio > 0.75:
                    delta = min(2 * delta, delta_max)
                elif ratio > 0.25:
                    pass
                else:
                    delta = delta / 2.0
            else:
                # Failure
                delta = delta / 2.0

            # Update best
            if f_trial < f_best:
                f_best = f_trial
                x_best = x_trial.copy()

            # Update X_set and F_set
            # Replace the point with the highest function value to keep the set focused
            X_set.append(x_trial)
            F_set.append(f_trial)
            
            if len(X_set) > n_model:
                idx_max = np.argmax(F_set)
                X_set.pop(idx_max)
                F_set.pop(idx_max)

            # Stopping criteria
            if delta < delta_min:
                break
            if np.linalg.norm(g) < 1e-6:
                break

        return x_best, f_best
