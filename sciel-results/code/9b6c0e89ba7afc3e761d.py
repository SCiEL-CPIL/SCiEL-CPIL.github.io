"""Discounted Least Squares with Adaptive Forgetting and Contextual-Free Baseline."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Per-arm Discounted Least Squares (DLS) for linear model estimation.
    - Adaptive forgetting factor based on prediction error.
    - Contextual-free baseline (EMA of rewards) per arm.
    - Blended score: combination of linear prediction and baseline.
    - Thompson Sampling on the linear component for exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = self.t0 + 110.0

        # DLS State: H (Hessian), b (gradient)
        # Initialize H with identity to encourage exploration initially
        self.H = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.H_inv = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        
        # Baseline: EMA of rewards
        self.reward_ema = [0.0] * n_arms
        self.reward_counts = [0] * n_arms
        
        # Adaptive Forgetting
        # Base forgetting factor (close to 1 is slow forgetting)
        self.base_lambda = 0.98
        # Error-based adjustment
        self.err_ema = [0.0] * n_arms
        
        # Exploration parameters
        self.use_ts = True
        self.blend_weight = 0.2 # Weight for baseline in score
        
        self._deadline_reached = False

    def _check_deadline(self):
        if self._deadline_reached:
            return False
        if (time.time() - self.t0) >= self.deadline:
            self._deadline_reached = True
            return False
        return True

    def _get_lambda(self, arm):
        """
        Compute adaptive forgetting factor.
        If error is high, lambda decreases (faster forgetting).
        """
        if self.reward_counts[arm] < 5:
            return 0.95
        
        # Normalize error
        # Assuming reward scale is around 1
        err = self.err_ema[arm]
        # Map error to lambda
        # High error -> low lambda (e.g., 0.9)
        # Low error -> high lambda (e.g., 0.99)
        # Use sigmoid-like mapping
        # lambda = base_lambda - adjustment * err
        # Let's use: lambda = max(0.9, min(0.99, self.base_lambda - 0.01 * err))
        # Or more aggressive:
        adj = min(0.05, 0.01 * err)
        lam = self.base_lambda - adj
        return max(0.90, min(0.99, lam))

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                pred = self.theta[a] @ ctx
                base = self.reward_ema[a]
                score = (1 - self.blend_weight) * pred + self.blend_weight * base
                scores.append(score)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Predicted reward from linear model
            pred = self.theta[a] @ ctx
            
            # Baseline reward
            base = self.reward_ema[a]
            
            # Blend
            score = (1 - self.blend_weight) * pred + self.blend_weight * base
            
            if self.use_ts:
                # Thompson Sampling on theta
                # Covariance is proportional to H_inv
                # Scale H_inv to match expected variance
                # If H is large, H_inv is small (high confidence)
                try:
                    # Add small jitter for numerical stability
                    cov = self.H_inv[a] + 1e-8 * np.eye(self.d)
                    L = np.linalg.cholesky(cov)
                    z = np.random.randn(self.d)
                    theta_sample = self.theta[a] + L @ z
                    # Re-compute score with sampled theta
                    # Note: We only sample the linear part, not the baseline
                    score = (1 - self.blend_weight) * (theta_sample @ ctx) + self.blend_weight * base
                except np.linalg.LinAlgError:
                    pass # Keep original score
            
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        self.reward_counts[arm] += 1
        
        # Update Baseline EMA
        alpha_base = 0.1
        self.reward_ema[arm] = (1 - alpha_base) * self.reward_ema[arm] + alpha_base * reward
        
        # Update DLS
        lam = self._get_lambda(arm)
        
        # Discount previous H and b
        self.H[arm] = lam * self.H[arm]
        self.b[arm] = lam * self.b[arm]
        
        # Add new observation
        self.H[arm] += np.outer(ctx, ctx)
        self.b[arm] += reward * ctx
        
        # Update H_inv using Sherman-Morrison-Woodbury or just invert
        # Inverting d x d matrix every step is O(d^3).
        # For small d (e.g., < 10), this is fine.
        # For larger d, use SMW.
        # Let's use SMW for efficiency and stability.
        # H_new = H_old + x x^T
        # H_new_inv = H_old_inv - (H_old_inv x x^T H_old_inv) / (1 + x^T H_old_inv x)
        
        H_inv = self.H_inv[arm]
        Hx = H_inv @ ctx
        denom = 1.0 + ctx @ Hx
        
        if denom > 1e-9:
            self.H_inv[arm] = H_inv - np.outer(Hx, Hx) / denom
        else:
            # Fallback to direct inversion if SMW fails (numerical issues)
            try:
                self.H_inv[arm] = np.linalg.inv(self.H[arm])
            except np.linalg.LinAlgError:
                self.H_inv[arm] = np.eye(self.d) * 1e-6
        
        # Update theta
        self.theta[arm] = self.H_inv[arm] @ self.b[arm]
        
        # Update error EMA for adaptive lambda
        pred = self.theta[arm] @ ctx
        err = (reward - pred) ** 2
        alpha_err = 0.1
        self.err_ema[arm] = (1 - alpha_err) * self.err_ema[arm] + alpha_err * err

# EVOLVE-BLOCK-END
