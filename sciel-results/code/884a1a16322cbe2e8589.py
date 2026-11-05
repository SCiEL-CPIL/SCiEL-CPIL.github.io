# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy using a global linear model
    with augmented context (one-hot arm indicators) and adaptive forgetting.
    
    Key design:
    - Single shared weight vector w of size (dim + n_arms).
    - Context is augmented with one-hot arm indicators: x_aug = [context, e_arm]
    - This allows capturing arm-specific biases while sharing context features.
    - Sherman-Morrison update for O(d'^2) per update where d' = dim + n_arms.
    - Adaptive forgetting factor based on recent reward variance.
    - UCB exploration with adaptive C.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms  # augmented dimension
        
        # Time management
        self._t0 = time.time()
        self._deadline = self._t0 + 145.0  # 5s safety margin before 150s kill

        # Global model parameters
        # V = A^{-1} where A = sum(gamma^t * x_aug x_aug^T) + lambda * I
        # Initialize V = I / lambda. Let lambda = 1.0, so V = I.
        self.V = np.eye(self.d_aug, dtype=np.float64)
        self.b = np.zeros(self.d_aug, dtype=np.float64)
        self.total_count = 0

        # Adaptive parameters
        self.base_gamma = 0.95
        self.gamma_min = 0.85
        self.gamma_max = 0.999
        
        # Exploration parameter
        self.base_C = 2.0
        self.C_min = 0.5
        self.C_max = 4.0

        # Tracking for adaptation
        self.reward_history = []
        self.max_history = 100

    def _check_deadline(self):
        """Return True if we should stop exploring and just pick best known arm."""
        return time.time() > self._deadline

    def _adaptive_gamma(self):
        """
        Adapt forgetting factor based on recent reward variance.
        High variance suggests non-stationarity -> lower gamma (faster forgetting).
        """
        if len(self.reward_history) < 20:
            return self.base_gamma

        recent = self.reward_history[-min(50, len(self.reward_history)):]
        var = np.var(recent)
        mean_abs = np.mean(np.abs(recent))

        # Normalize variance relative to mean absolute reward to be scale-invariant
        if mean_abs > 1e-9:
            normalized_var = var / (mean_abs ** 2 + 1e-9)
        else:
            normalized_var = 0.0

        # Map normalized variance to gamma range
        scale = min(1.0, normalized_var * 2.0)
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * scale
        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        """
        Adapt exploration parameter C.
        Higher C for arms with fewer samples to encourage exploration.
        Lower C overall as total rounds increase to exploit.
        """
        if self.total_count < 10:
            return self.C_max

        # Decrease exploration as we learn more globally
        progress = min(1.0, self.total_count / 500.0)
        C = self.C_max - (self.C_max - self.C_min) * progress

        return max(self.C_min, min(self.C_max, C))

    def _augment_context(self, context, arm):
        """Augment context with one-hot arm indicator."""
        x_aug = np.zeros(self.d_aug, dtype=np.float64)
        x_aug[:self.d] = context
        x_aug[self.d + arm] = 1.0
        return x_aug

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        if self._check_deadline():
            # Deadline approaching: pick the arm with highest estimated reward (greedy)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                x_aug = self._augment_context(context, a)
                theta = self.V @ self.b
                score = theta @ x_aug
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        scores = np.zeros(self.n)
        for a in range(self.n):
            x_aug = self._augment_context(context, a)
            # theta = V @ b
            theta = self.V @ self.b
            mu = theta @ x_aug
            
            # Uncertainty bonus: sqrt(x_aug^T V x_aug)
            # v = V @ x_aug
            v = self.V @ x_aug
            uncertainty = np.sqrt(np.dot(x_aug, v))
            
            C = self._adaptive_C(a)
            scores[a] = mu + C * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.total_count += 1
        
        if self._check_deadline():
            pass

        gamma = self._adaptive_gamma()
        
        x_aug = self._augment_context(context, arm)
        
        # Sherman-Morrison update with forgetting factor
        # A_new = gamma * A_old + x_aug x_aug^T
        # V_new = (gamma * A_old + x_aug x_aug^T)^{-1}
        #
        # Derivation:
        # A_new = gamma * (A_old + (1/gamma) x_aug x_aug^T)
        # V_new = (1/gamma) * (A_old + (1/gamma) x_aug x_aug^T)^{-1}
        # Using SM formula: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = A_old, u = (1/gamma) x_aug, v = x_aug
        # (A_old + (1/gamma) x_aug x_aug^T)^{-1} = V_old - V_old ((1/gamma) x_aug) x_aug^T V_old / (1 + x_aug^T V_old ((1/gamma) x_aug))
        # Let Vx = V_old @ x_aug
        # Numerator term: (1/gamma) * Vx @ x_aug^T
        # Denominator term: 1 + (1/gamma) * x_aug^T Vx
        #
        # V_temp = V_old - (1/gamma) * (Vx @ x_aug^T) / (1 + (1/gamma) * x_aug^T Vx)
        # V_new = (1/gamma) * V_temp
        # V_new = (1/gamma) * V_old - (1/gamma)^2 * (Vx @ x_aug^T) / (1 + (1/gamma) * x_aug^T Vx)
        
        V = self.V
        x = x_aug
        gamma_inv = 1.0 / gamma
        
        # Compute Vx
        Vx = V @ x
        # Compute scalar x^T Vx
        xVx = np.dot(x, Vx)
        
        # Denominator
        denom = 1.0 + gamma_inv * xVx
        
        # Update V
        self.V = gamma_inv * V - (gamma_inv ** 2) * np.outer(Vx, Vx) / denom
        
        # Update b with forgetting
        self.b = gamma * self.b + reward * x
        
        # Track reward history for adaptive gamma
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
# EVOLVE-BLOCK-END
