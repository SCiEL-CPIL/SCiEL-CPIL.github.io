"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary Linear Contextual Bandit with Adaptive Forgetting.
    
    This policy uses LinUCB with a dynamically adjusted forgetting factor (lambda)
    for each arm. The forgetting factor is adapted based on the recent prediction 
    errors (residuals) of that arm. Large or increasing residuals indicate 
    non-stationarity (drift), causing the policy to forget older data faster.
    
    Key features:
    1. Per-arm adaptive lambda based on residual variance and magnitude.
    2. Global drift detection to trigger aggressive forgetting across all arms.
    3. Constant exploration coefficient (alpha) to maintain sensitivity to changes.
    4. Strict time limit guard to ensure termination within evaluator constraints.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize linear regression matrices
        # Small regularization to avoid singularity initially
        self.reg = 0.1
        self.A = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.lambdas = [0.95 for _ in range(n_arms)]
        self.lambda_min = 0.80
        self.lambda_max = 0.99
        self.lambda_base = 0.95
        self.lambda_adapt_rate = 0.1
        
        # Residual tracking for each arm
        self.residuals = [deque(maxlen=10) for _ in range(n_arms)]
        
        # Global drift tracking
        self.recent_rewards = deque(maxlen=20)
        self.global_reward_avg = 0.0
        self.global_reward_var = 1.0
        
        # Exploration coefficient
        # In non-stationary settings, we keep alpha constant or slowly decaying.
        # A value around 1.0-1.5 is typical for dim < 10.
        self.alpha = 1.2
        self.alpha_init = 1.2
        self.alpha_min = 0.6
        self.alpha_decay_rate = 0.0003  # Slow decay per total observation
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # Safety margin: 150s limit, stop at 145s

    def _solve(self, A, b):
        """Solve A x = b with fallback for singular matrices."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _update_lambda(self, arm, residual_sq):
        """
        Update the forgetting factor for a specific arm based on recent errors.
        High or volatile errors -> lower lambda (faster forgetting).
        Low and stable errors -> higher lambda (slower forgetting, more memory).
        """
        self.residuals[arm].append(residual_sq)
        
        if len(self.residuals[arm]) < 3:
            return

        recent_res = list(self.residuals[arm])
        mean_res = np.mean(recent_res)
        std_res = np.std(recent_res) if len(recent_res) > 2 else 0.0
        
        # Heuristic: 
        # If mean error is high, or variance is high (instability), decrease lambda.
        # We want lambda to be lower when error is large.
        # Target lambda ~ 0.95 for low error, ~0.85 for high error.
        
        # Scale error metric
        error_metric = mean_res + std_res
        
        # Map error_metric to lambda adjustment
        # If error_metric is small, keep lambda high.
        # If error_metric is large, lower lambda.
        
        # Simple linear mapping with clamping
        # sensitivity factor
        sens = 0.05
        adjustment = -sens * error_metric
        
        new_lambda = self.lambdas[arm] + self.lambda_adapt_rate * adjustment
        new_lambda = max(self.lambda_min, min(self.lambda_max, new_lambda))
        
        self.lambdas[arm] = new_lambda

    def _check_global_drift(self, reward):
        """
        Detect global drift by comparing current reward to recent history.
        Returns True if a significant drop or change is detected.
        Uses smoothed statistics for more robust detection.
        """
        self.recent_rewards.append(reward)
        
        if len(self.recent_rewards) < 5:
            return False
            
        current_mean = np.mean(self.recent_rewards)
        # Simple variance estimation
        current_var = np.var(self.recent_rewards)
        
        # Update running stats slowly with adaptive rate
        # Use a slightly faster adaptation for variance to catch sudden changes
        self.global_reward_avg = 0.9 * self.global_reward_avg + 0.1 * current_mean
        self.global_reward_var = 0.85 * self.global_reward_var + 0.15 * current_var
        
        # Drift if current mean deviates significantly from global average
        # or if variance spikes
        # Normalize threshold by the scale of rewards
        scale = max(1.0, np.sqrt(self.global_reward_var + 1e-6))
        drift_threshold = 0.6 * scale
        
        if abs(current_mean - self.global_reward_avg) > drift_threshold:
            return True
            
        return False

    def _select_arm(self, context):
        """Select arm using LinUCB with adaptive parameters."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Adaptively decay alpha based on total observations
        # This reduces over-exploration in later rounds while maintaining
        # sufficient exploration early
        total_obs = sum(self.counts)
        self.alpha = max(self.alpha_min, self.alpha_init - self.alpha_decay_rate * total_obs)

        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            # Solve for theta
            theta = self._solve(A, b)
            mu = theta @ context
            
            # Compute variance term: context^T A^{-1} context
            try:
                A_inv_context = np.linalg.solve(A, context)
                v = context @ A_inv_context
            except np.linalg.LinAlgError:
                v = 1.0
            
            # Exploration bonus
            # We can slightly boost exploration for arms with very few pulls
            # to ensure initial coverage, but LinUCB naturally handles this via uncertainty.
            bonus = self.alpha * np.sqrt(max(v, 0.0))
            
            scores.append(mu + bonus)

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: return arm with highest historical average reward if available
            # Since we don't track averages explicitly here, we can just return 0
            # or a random arm. To be safe and deterministic, return 0.
            # A better fallback would be tracking moving averages, but for simplicity and speed:
            return 0

        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute prediction error
        A = self.A[arm]
        b = self.b[arm]
        theta = self._solve(A, b)
        mu_pred = theta @ context
        residual = reward - mu_pred
        residual_sq = residual ** 2

        # 2. Update adaptive lambda for this arm
        self._update_lambda(arm, residual_sq)

        # 3. Check for global drift
        is_global_drift = self._check_global_drift(reward)
        if is_global_drift:
            # If global drift is detected, aggressively forget for all arms
            # by lowering all lambdas
            for a in range(self.n):
                self.lambdas[a] = max(self.lambda_min, self.lambdas[a] * 0.95)

        # 4. Update linear model for the pulled arm
        lam = self.lambdas[arm]
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # 5. Update count
        self.counts[arm] += 1


# EVOLVE-BLOCK-END
