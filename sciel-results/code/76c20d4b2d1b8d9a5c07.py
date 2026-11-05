# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary linear bandit with per-arm adaptive discounted RLS and UCB exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim + 1  # Augmented dimension with intercept
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 110s deadline (10s safety margin)
        
        # Per-arm RLS state
        self.V = [np.eye(self.d) for _ in range(n_arms)]  # Inverse covariance
        self.theta = [np.zeros(self.d) for _ in range(n_arms)]  # Weight estimate
        self.count = [0 for _ in range(n_arms)]
        self.alpha = 1.5  # Exploration parameter
        self.total_rounds = 0
        
        # Per-arm adaptive forgetting state
        self.innovation_history = [[] for _ in range(n_arms)]  # Recent innovations
        self.innovation_var = [1.0 for _ in range(n_arms)]  # Innovation variance per arm
        self.gamma_per_arm = [0.98 for _ in range(n_arms)]  # Per-arm forgetting factor
        self.reward_history = [0.0 for _ in range(n_arms)]
        self.gamma_min = 0.95
        self.gamma_max = 0.999
        self.innovation_window = 20

    def _augment(self, context):
        """Augment context with intercept term."""
        context = np.asarray(context, dtype=np.float64)
        return np.append(context, 1.0)

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        return time.time() > self.deadline

    def _select_safe_arm(self):
        """Return a safe arm when deadline is approaching."""
        if any(r != 0 for r in self.reward_history):
            return int(np.argmax(self.reward_history))
        return 0

    def _update_gamma(self, arm, innovation):
        """Adaptively adjust per-arm forgetting factor based on innovation variance."""
        # Store innovation in history
        self.innovation_history[arm].append(innovation)
        if len(self.innovation_history[arm]) > self.innovation_window:
            self.innovation_history[arm].pop(0)
        
        # Compute recent innovation variance
        if len(self.innovation_history[arm]) >= 5:
            recent_innovations = np.array(self.innovation_history[arm])
            self.innovation_var[arm] = np.var(recent_innovations) + 1e-8
        
        # Adapt gamma: higher variance -> lower gamma (faster forgetting)
        # Map innovation variance to gamma in [gamma_min, gamma_max]
        # Use a smooth mapping: gamma = gamma_max - (gamma_max - gamma_min) * min(var/scale, 1)
        scale = 1.0  # Normalization scale for variance
        var_normalized = min(self.innovation_var[arm] / scale, 1.0)
        target_gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * var_normalized
        # Smoothly update gamma
        self.gamma_per_arm[arm] = 0.85 * self.gamma_per_arm[arm] + 0.15 * target_gamma

    def select(self, context):
        # Deadline guard
        if self._check_deadline():
            return self._select_safe_arm()
        
        x = self._augment(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Predicted reward
            pred = self.theta[a] @ x
            
            # Uncertainty bonus: sqrt(x^T V x)
            Vx = self.V[a] @ x
            uncertainty = np.sqrt(max(0.0, x @ Vx))
            
            scores[a] = pred + self.alpha * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if self._check_deadline():
            return
        
        x = self._augment(context)
        self.total_rounds += 1
        gamma = self.gamma_per_arm[arm]
        
        # Compute innovation (prediction error)
        pred = self.theta[arm] @ x
        innovation = reward - pred
        
        # Update reward history for safe arm selection
        self.reward_history[arm] = 0.95 * self.reward_history[arm] + 0.05 * reward
        
        # RLS update with forgetting
        # V_new = (1/gamma) * V_old - (1/gamma) * V_old @ x @ x^T @ V_old / (gamma + x^T @ V_old @ x)
        # theta_new = theta_old + V_new @ x * innovation
        
        Vx = self.V[arm] @ x
        denom = gamma + x @ Vx
        
        if denom > 1e-10:
            # Update V using Sherman-Morrison with forgetting
            # V_new = (1/gamma) * V - (1/gamma) * Vx @ x^T @ V / (gamma + x^T V x)
            # = (1/gamma) * (V - Vx @ x^T @ V / (gamma + x^T V x))
            Vx_xT_V = np.outer(Vx, self.V[arm] @ x)
            self.V[arm] = (1.0 / gamma) * (self.V[arm] - Vx_xT_V / denom)
            
            # Update theta
            self.theta[arm] = self.theta[arm] + (self.V[arm] @ x) * innovation
        
        # Adaptively update gamma for this arm
        self._update_gamma(arm, innovation)
        
        self.count[arm] += 1
        
        # Adaptive exploration: decrease alpha over time but keep some exploration
        self.alpha = max(0.5, 1.5 * np.exp(-self.total_rounds / 300.0))
# EVOLVE-BLOCK-END
