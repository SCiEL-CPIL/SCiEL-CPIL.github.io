"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Multi-timescale Bayesian Linear Regression policy for non-stationary linear contextual bandits.
    
    Strategy:
    1. Maintain two LinUCB-style models per arm:
       - Short-window model (fast decay): Captures recent trends/shifts.
       - Long-window model (slow decay): Provides a stable baseline estimate.
    2. Selection uses the Short-window model's UCB (Mean + Alpha * Uncertainty) to prioritize 
       adaptation to non-stationarity.
    3. The Long-window model is used to regularize the Short-window's mean estimate, preventing 
       the short model from overfitting to noise in very small recent samples, while still 
       allowing it to deviate from the long baseline when strong recent evidence exists.
    4. Efficient O(D^2) updates using Sherman-Morrison-Woodbury or direct inverse updates 
       (since D is small, direct inverse update is stable and fast).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin

        # Hyperparameters
        # Short window: fast forgetting (gamma close to 1)
        self.gamma_short = 0.995 
        # Long window: slow forgetting
        self.gamma_long = 0.9995
        
        # Exploration coefficient
        self.alpha = 1.5
        
        # Regularization for initialization
        self.reg = 1.0
        
        # Initialize models for each arm
        # We store the inverse of the covariance matrix (A_inv) and the weighted sum (b)
        # Short model
        self.A_inv_short = [np.eye(self.d) / self.reg for _ in range(n_arms)]
        self.b_short = [np.zeros(self.d) for _ in range(n_arms)]
        
        # Long model
        self.A_inv_long = [np.eye(self.d) / self.reg for _ in range(n_arms)]
        self.b_long = [np.zeros(self.d) for _ in range(n_arms)]

    def _update_model(self, A_inv, b, context, reward, gamma):
        """
        Update the Bayesian linear regression parameters with a forgetting factor.
        A_inv_new = (gamma * A_inv_old)^{-1} + x x^T
                   = (1/gamma) * A_inv_old + x x^T  <-- Wait, (gamma * A_inv)^-1 is (1/gamma) * A.
                   Actually, standard recursive Bayesian update with forgetting:
                   A_t = gamma * A_{t-1} + x x^T
                   b_t = gamma * b_{t-1} + x y
                   
                   We maintain A_inv = A^{-1}.
                   A_t = gamma A_{t-1} + x x^T
                   A_t^{-1} = (gamma A_{t-1} + x x^T)^{-1}
                   Using Sherman-Morrison-Woodbury:
                   (gamma A_{t-1} + x x^T)^{-1} = (1/gamma) A_{t-1}^{-1} - (1/gamma) A_{t-1}^{-1} x x^T A_{t-1}^{-1} / (gamma + x^T A_{t-1}^{-1} x)
                   
                   Let's verify:
                   Let M = gamma A_{t-1}. M^{-1} = (1/gamma) A_{t-1}^{-1}.
                   (M + x x^T)^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
                   = (1/gamma) A_{t-1}^{-1} - (1/gamma) A_{t-1}^{-1} x x^T (1/gamma) A_{t-1}^{-1} / (1 + x^T (1/gamma) A_{t-1}^{-1} x)
                   = (1/gamma) [ A_{t-1}^{-1} - A_{t-1}^{-1} x x^T A_{t-1}^{-1} / (gamma + x^T A_{t-1}^{-1} x) ]
                   
                   This is correct.
                   """
        x = context
        y = reward
        
        # Update b
        b_new = gamma * b + y * x
        
        # Update A_inv
        A_inv_old = A_inv
        Ax = A_inv_old @ x
        denom = gamma + x @ Ax
        
        # A_inv_new = (1/gamma) * A_inv_old - (1/gamma) * (A_inv_old @ x) (A_inv_old @ x)^T / denom
        # Note: The term (1/gamma) can be factored out, but it's safer to compute directly.
        # Actually, looking at the formula again:
        # A_t = gamma A_{t-1} + x x^T
        # A_t^{-1} = (1/gamma) A_{t-1}^{-1} - (1/gamma) A_{t-1}^{-1} x x^T A_{t-1}^{-1} / (gamma + x^T A_{t-1}^{-1} x)
        
        A_inv_new = (1.0 / gamma) * A_inv_old
        correction = np.outer(Ax, Ax) / denom
        A_inv_new -= (1.0 / gamma) * correction
        
        # Symmetrize to prevent numerical drift
        A_inv_new = (A_inv_new + A_inv_new.T) / 2.0
        
        return A_inv_new, b_new

    def _get_value_and_uncertainty(self, arm, context):
        """
        Compute the estimated mean and uncertainty for an arm.
        Uses the short model for uncertainty and a blended mean.
        """
        x = context
        A_inv_s = self.A_inv_short[arm]
        b_s = self.b_short[arm]
        A_inv_l = self.A_inv_long[arm]
        b_l = self.b_long[arm]

        # Mean estimates
        theta_s = A_inv_s @ b_s
        theta_l = A_inv_l @ b_l
        
        # Uncertainty (variance) from short model
        var_s = x @ A_inv_s @ x
        std_s = np.sqrt(var_s)
        
        # Uncertainty (variance) from long model
        var_l = x @ A_inv_l @ x
        std_l = np.sqrt(var_l)

        # Blend the means. 
        # We want to trust the short model more if it has low uncertainty, 
        # but the long model provides stability.
        # A simple blend: Mean = w * Mean_s + (1-w) * Mean_l
        # Where w is high if short model is confident (low var) relative to long.
        # Alternatively, just use the short model mean but regularized by long model.
        # Let's use a weighted average based on inverse variances (precision weighting).
        # Precision_s = 1/var_s, Precision_l = 1/var_l
        # But var_s is typically larger (less data) than var_l? 
        # Actually, short model forgets fast, so it has "less effective data", hence higher variance.
        # So we should weight more towards the long model if short is uncertain.
        
        # To avoid division by zero, add small epsilon
        eps = 1e-6
        prec_s = 1.0 / (var_s + eps)
        prec_l = 1.0 / (var_l + eps)
        
        total_prec = prec_s + prec_l
        w_s = prec_s / total_prec
        w_l = prec_l / total_prec
        
        mean_val = w_s * (theta_s @ x) + w_l * (theta_l @ x)
        
        # For UCB, we use the uncertainty of the short model because we want to explore 
        # areas where the recent behavior is uncertain.
        # However, if the short model is very uncertain, the UCB might be huge.
        # We can cap the uncertainty or use a combined uncertainty.
        # Let's stick to short model uncertainty for the exploration bonus, 
        # as the goal is to adapt to recent shifts.
        
        return mean_val, std_s

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(np.random.randint(self.n))

        scores = []
        for a in range(self.n):
            mean_val, uncertainty = self._get_value_and_uncertainty(a, context)
            # UCB-style score
            score = mean_val + self.alpha * uncertainty
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update short model
        self.A_inv_short[arm], self.b_short[arm] = self._update_model(
            self.A_inv_short[arm], self.b_short[arm], context, reward, self.gamma_short
        )
        
        # Update long model
        self.A_inv_long[arm], self.b_long[arm] = self._update_model(
            self.A_inv_long[arm], self.b_long[arm], context, reward, self.gamma_long
        )

# EVOLVE-BLOCK-END
