import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # RLS State
        self.P = [np.eye(dim) / 1.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive Forgetting
        self.base_lambda = 0.98
        self.min_lambda = 0.90
        self.max_lambda = 0.999
        self.error_window = 20
        self.recent_errors = [np.zeros(self.error_window) for _ in range(n_arms)]
        self.error_idx = [0] * n_arms
        
        # Deadline Guard
        self.start_time = time.time()
        self.deadline = 140.0
        
        # Random generator for Thompson Sampling
        self.rng = np.random.default_rng()

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _get_lambda(self, arm):
        if self.t < 5:
            return self.base_lambda
            
        n_recent = min(self.t, self.error_window)
        idx = self.error_idx[arm]
        recent = self.recent_errors[arm]
        
        # Gather recent errors
        if n_recent <= self.error_window:
            # Handle circular buffer logic simply for small n
            # Since we just want the last n_recent, and error_idx points to next write
            # The errors are stored at (idx - 1), (idx - 2), ...
            # Actually, let's just keep a list for simplicity if dim is small, 
            # or stick to circular buffer.
            # Circular buffer retrieval:
            errors = []
            for i in range(n_recent):
                pos = (idx - 1 - i) % self.error_window
                errors.append(recent[pos])
            errors = np.array(errors)
        else:
            errors = recent[:] # If window is full and n_recent > window, just use window
            
        if len(errors) == 0:
            return self.base_lambda
            
        mse = np.mean(errors**2)
        
        # Map MSE to lambda
        # Low MSE -> High lambda (stable)
        # High MSE -> Low lambda (adaptive)
        # Thresholds
        if mse < 0.05:
            return self.max_lambda
        elif mse > 0.5:
            return self.min_lambda
        else:
            # Linear interpolation
            # 0.05 -> 0.999, 0.5 -> 0.90
            t_norm = (mse - 0.05) / (0.5 - 0.05)
            return self.max_lambda - t_norm * (self.max_lambda - self.min_lambda)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        if self._check_deadline():
            # Greedy fallback
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                score = self.theta[a] @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # Thompson Sampling
        best_arm = 0
        best_sampled_reward = -np.inf
        
        for a in range(self.n):
            # Sample theta from N(theta, P)
            # P is symmetric positive definite
            # Use Cholesky or SVD for stability if needed, but for small dim, rng.multivariate_normal is fine
            # To be faster, we can use: theta_sample = theta + P @ rng.standard_normal(dim)
            # Wait, variance is P. So cov = P.
            # rng.multivariate_normal(mean, cov) is O(d^3).
            # Manual sampling: z ~ N(0, I). sample = mean + L @ z where L L^T = P.
            # Cholesky decomposition of P.
            try:
                L = np.linalg.cholesky(self.P[a])
                z = self.rng.standard_normal(self.d)
                theta_sample = self.theta[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback if P is not positive definite
                theta_sample = self.theta[a]
                
            reward_sample = theta_sample @ context
            
            if reward_sample > best_sampled_reward:
                best_sampled_reward = reward_sample
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        if self._check_deadline():
            return

        # Compute prediction error using current mean theta
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # Store error
        self.recent_errors[arm][self.error_idx[arm]] = error
        self.error_idx[arm] = (self.error_idx[arm] + 1) % self.error_window
        
        # Adaptive lambda
        lam = self._get_lambda(arm)
        
        # RLS Update
        P = self.P[arm]
        theta = self.theta[arm]
        
        Px = P @ context
        denom = lam + context @ Px
        if denom < 1e-10:
            denom = 1e-10
            
        gain = Px / denom
        theta_new = theta + gain * error
        P_new = (P - np.outer(gain, context)) / lam
        
        # Regularization/Reset if P becomes ill-conditioned
        if np.trace(P_new) > 1e4:
            P_new = np.eye(self.d) / 1.0
            theta_new = np.zeros(self.d)
            
        self.P[arm] = P_new
        self.theta[arm] = theta_new
