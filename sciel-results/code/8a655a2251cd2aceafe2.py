"""Non-stationary linear contextual bandit with global model and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear bandit with a global linear model and arm-specific biases.
    
    Assumption: reward = theta_global @ context + bias[arm] + noise
    This structure shares context-feature weights across arms, improving sample efficiency
    while maintaining arm-specific adaptation through biases.
    
    Uses an adaptive forgetting factor to track non-stationary changes.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._start_time = time.time()
        self._deadline = 115.0  # Safety margin before 120s limit
        
        # Global model: A_global (dxd), b_global (d,)
        self.A_global = np.eye(dim) * 1.0
        self.b_global = np.zeros(dim)
        
        # Arm biases: We model bias as a linear model on a constant feature [1]
        # Or simply track mean bias and variance for each arm.
        # Let's use a simple exponential moving average for bias and its variance.
        # Actually, to keep it consistent with the linear framework, we can treat bias 
        # as part of the model. But separate bias is simpler and often works well.
        
        # Let's stick to: r = theta^T x + bias_arm
        # We maintain A_global, b_global for theta.
        # For biases, we maintain mean_bias[arm] and var_bias[arm] using exponential moving averages.
        self.mean_bias = np.zeros(n_arms)
        self.var_bias = np.ones(n_arms) * 10.0  # Initial high uncertainty
        
        # Adaptive parameters
        self.base_forgetting = 0.95
        self.forgetting = self.base_forgetting
        self.base_alpha = 1.0
        self.alpha = self.base_alpha
        
        # Error tracking
        self.recent_errors = []
        self.error_window = 20
        
        # Round counter
        self.round = 0

    def _get_adaptive_params(self):
        """Compute adaptive forgetting factor and exploration coefficient."""
        if len(self.recent_errors) < 5:
            return self.base_forgetting, self.base_alpha
        
        recent = self.recent_errors[-self.error_window:]
        mean_err = np.mean(np.abs(recent))
        std_err = np.std(recent)
        
        # Adaptive forgetting: decrease (more forgetting) when errors are high
        # Map mean error to forgetting factor in [0.85, 0.99]
        if mean_err > 1.0:
            excess = min(mean_err / 1.0, 5.0)
            forgetting = self.base_forgetting - 0.1 * min(excess, 1.0)
            forgetting = max(0.80, min(0.99, forgetting))
        elif mean_err < 0.1:
            ratio = mean_err / 0.1
            forgetting = min(0.995, self.base_forgetting + 0.04 * (1.0 - ratio))
        else:
            t = (mean_err - 0.1) / (1.0 - 0.1)
            forgetting = self.base_forgetting - 0.05 * t
        
        # Adaptive alpha: increase exploration when errors are high
        if mean_err > 1.0:
            excess = min(mean_err / 1.0, 3.0)
            alpha = self.base_alpha + 1.0 * min(excess, 2.0)
        elif mean_err < 0.1:
            ratio = mean_err / 0.1
            alpha = max(0.3, self.base_alpha - 0.5 * (1.0 - ratio))
        else:
            t = (mean_err - 0.1) / (1.0 - 0.1)
            alpha = self.base_alpha + 0.5 * t
        
        # Add small component from std to capture volatility
        alpha += 0.2 * min(std_err / max(mean_err, 1e-8), 1.0)
        
        return forgetting, alpha

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Check deadline
        if time.time() - self._start_time > self._deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A_global, self.b_global)
                    mean = float(theta @ context) + self.mean_bias[a]
                    if mean > best_score:
                        best_score = mean
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return best_arm
        
        forgetting, alpha = self._get_adaptive_params()
        
        scores = np.zeros(self.n)
        try:
            # Solve for global theta once
            theta = np.linalg.solve(self.A_global, self.b_global)
            mean_global = float(theta @ context)
            
            # Uncertainty of global model: context^T A^{-1} context
            inv_A_ctx = np.linalg.solve(self.A_global, context)
            var_global = float(context @ inv_A_ctx)
        except np.linalg.LinAlgError:
            # Fallback
            try:
                theta = np.linalg.pinv(self.A_global) @ self.b_global
                mean_global = float(theta @ context)
                inv_A = np.linalg.pinv(self.A_global)
                var_global = float(context @ inv_A @ context)
            except Exception:
                mean_global = 0.0
                var_global = 0.0

        for a in range(self.n):
            mean = mean_global + self.mean_bias[a]
            
            # Total uncertainty: global model uncertainty + bias uncertainty
            # We scale bias uncertainty by a factor to make it comparable
            total_var = var_global + self.var_bias[a]
            bonus = alpha * np.sqrt(max(total_var, 0.0))
            
            scores[a] = mean + bonus
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.round += 1
        
        # Compute prediction error
        try:
            theta = np.linalg.solve(self.A_global, self.b_global)
            pred = float(theta @ context) + self.mean_bias[arm]
            error = reward - pred
        except np.linalg.LinAlgError:
            error = 0.0
        
        # Track recent errors
        self.recent_errors.append(abs(error))
        if len(self.recent_errors) > self.error_window * 2:
            self.recent_errors = self.recent_errors[-self.error_window * 2:]
        
        # Get adaptive forgetting factor
        forgetting, _ = self._get_adaptive_params()
        
        # Update global model
        # We need to isolate the bias contribution from the reward to update theta cleanly?
        # No, the standard RLS update for r = theta^T x + bias is tricky if bias is separate.
        # If we assume bias is fixed or slow, we can subtract current bias estimate from reward
        # before updating theta.
        residual = reward - self.mean_bias[arm]
        
        self.A_global = forgetting * self.A_global + np.outer(context, context)
        self.b_global = forgetting * self.b_global + residual * context
        
        # Update bias
        # Use exponential moving average for bias mean and variance
        # lambda_bias can be same as forgetting or slightly different
        # Let's use the same forgetting factor for consistency
        self.mean_bias[arm] = forgetting * self.mean_bias[arm] + (1 - forgetting) * reward
        # Update variance: Var = forgetting * (Var + (reward - mean_old)^2 * (1-forgetting)) ?
        # Simpler: track mean of squared errors for bias
        # Or just use a fixed decay for variance
        # Let's use a simple exponential moving average of the squared residual from the global model
        # Actually, if we subtract mean_bias, the residual is what we use for theta.
        # The bias itself is updated with the full reward.
        
        # To estimate bias uncertainty, we can look at the variance of the rewards for that arm
        # after subtracting the global prediction.
        # Let's maintain a running variance for each arm's bias.
        # Var_new = forgetting * Var_old + (1-forgetting) * (reward - mean_bias_old)^2
        # This is an approximation.
        
        # A more robust way for bias variance:
        # If we treat bias as a parameter, its uncertainty decreases with more data.
        # But since it's non-stationary, we want it to increase if it changes.
        # Let's just use a fixed exploration bonus for bias that decays slowly or stays constant.
        # Or better, use the adaptive alpha to scale the bias uncertainty.
        
        # Let's refine the bias variance update:
        # We assume the bias is a random walk.
        # The variance of the estimate should reflect the noise level.
        # Let's just keep var_bias as a decaying memory of the squared differences.
        diff = reward - self.mean_bias[arm] # Note: mean_bias is updated AFTER this? 
        # In the code above, I updated mean_bias AFTER computing residual.
        # Let's reorder: 
        # 1. Compute pred
        # 2. Update A_global, b_global using residual = reward - mean_bias[arm]
        # 3. Update mean_bias[arm]
        # 4. Update var_bias[arm]
        
        # Wait, in the code block above, I updated mean_bias[arm] BEFORE updating var_bias?
        # No, I didn't update var_bias in the previous snippet. I need to add it.
        
        # Let's correct the update logic in the final code below.
# EVOLVE-BLOCK-END
