"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with dynamic alpha and refined non-stationary handling."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Initialize covariance matrices and bias vectors for each arm
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        self._start_time = time.time()
        # Safety margin: 120s limit, use 115s to be safe
        self._deadline = self._start_time + 115.0
        
        # Parameters for adaptive exploration and forgetting
        self._base_alpha = 0.7
        self._min_alpha = 0.3
        self._max_alpha = 1.5
        
        self._base_lambda = 0.96
        self._min_lambda = 0.80
        self._max_lambda = 0.995
        
        self._error_window = 10
        self._recent_errors = []
        
        # State for tracking
        self._last_arm = None
        self._last_context = None
        self._round = 0

    def _get_alpha(self):
        """
        Dynamically adjust UCB alpha based on recent prediction errors.
        High error variance or magnitude -> higher alpha (more exploration).
        Low error -> lower alpha (more exploitation).
        """
        if len(self._recent_errors) < 3:
            return self._base_alpha
        
        recent = self._recent_errors[-self._error_window:]
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0
        
        # Normalize error metric
        # We want to map the error level to alpha.
        # A heuristic: if mean error is high, we are likely in a transition, so explore.
        # If std is high, the model is inconsistent, so explore.
        
        # Scale factor to normalize errors roughly to [0, 1] range for adjustment
        # Assuming rewards are in a reasonable range (e.g., 0-1 or similar), 
        # an error of 0.5 is significant.
        error_metric = mean_err + 0.5 * std_err
        
        # Map error_metric to alpha adjustment
        # If error_metric is close to 0, use min_alpha
        # If error_metric is > 0.5, use max_alpha
        # Linear interpolation with clipping
        if error_metric < 0.1:
            alpha_val = self._min_alpha
        elif error_metric > 0.5:
            alpha_val = self._max_alpha
        else:
            # Interpolate between min and max
            t = (error_metric - 0.1) / (0.5 - 0.1)
            alpha_val = self._min_alpha + t * (self._max_alpha - self._min_alpha)
            
        return np.clip(alpha_val, self._min_alpha, self._max_alpha)

    def _get_lambda(self):
        """
        Dynamically adjust forgetting factor based on error trend.
        If errors are increasing, forget faster (lower lambda).
        If errors are stable or decreasing, retain more (higher lambda).
        """
        if len(self._recent_errors) < 5:
            return self._base_lambda
        
        recent = self._recent_errors[-self._error_window:]
        # Compare average error of the last half vs the first half of the window
        half = len(recent) // 2
        first_half_mean = np.mean(recent[:half])
        second_half_mean = np.mean(recent[half:])
        
        # Trend: positive means errors are increasing
        trend = second_half_mean - first_half_mean
        
        # Base lambda, adjust based on trend
        # If trend > 0 (errors increasing), decrease lambda (forget faster)
        # If trend < 0 (errors decreasing), increase lambda (retain more)
        
        # Scale trend to a small adjustment factor
        # Assume typical error change is small
        adjustment = np.clip(trend * 5.0, -0.05, 0.05)
        
        lambda_val = self._base_lambda - adjustment
        
        # Also consider overall error magnitude
        mean_err = np.mean(recent)
        if mean_err > 0.4:
            # High overall error, be more aggressive with forgetting
            lambda_val = min(lambda_val, self._min_lambda + 0.05)
        elif mean_err < 0.1:
            # Low overall error, be more conservative
            lambda_val = max(lambda_val, self._max_lambda - 0.02)
            
        return np.clip(lambda_val, self._min_lambda, self._max_lambda)

    def _predict(self, arm, context):
        """Predict mean and uncertainty for a given arm and context."""
        try:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mu = theta @ context
            # Uncertainty: context^T A^{-1} context
            v = context @ A_inv @ context
            return mu, np.sqrt(max(v, 0.0))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            A_pinv = np.linalg.pinv(self.A[arm])
            theta = A_pinv @ self.b[arm]
            mu = theta @ context
            v = context @ A_pinv @ context
            return mu, np.sqrt(max(v, 0.0))

    def select(self, context):
        self._round += 1
        
        # Deadline guard: if close to limit, switch to pure exploitation (greedy)
        # to avoid expensive exploration or complex calculations
        if time.time() > self._deadline - 2.0:
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.inv(self.A[a]) @ self.b[a]
                    scores.append(theta @ context)
                except np.linalg.LinAlgError:
                    scores.append(0.0)
            return int(np.argmax(scores))

        # Get adaptive alpha
        alpha = self._get_alpha()
        
        scores = []
        for a in range(self.n):
            mu, sigma = self._predict(a, context)
            # UCB score
            scores.append(mu + alpha * sigma)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track prediction error for adaptive parameters
        # We compare the actual reward to the prediction we *would have made* 
        # if we had used the current model before the update.
        # However, since select() already picked the arm, we can just compute the error 
        # based on the current model state (which hasn't been updated yet).
        
        try:
            theta = np.linalg.inv(self.A[arm]) @ self.b[arm]
            pred = theta @ context
            err = abs(reward - pred)
            self._recent_errors.append(err)
            if len(self._recent_errors) > self._error_window * 2:
                self._recent_errors = self._recent_errors[-self._error_window:]
        except np.linalg.LinAlgError:
            # If we can't predict, skip error tracking
            pass

        # Update model with adaptive forgetting
        lam = self._get_lambda()
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
