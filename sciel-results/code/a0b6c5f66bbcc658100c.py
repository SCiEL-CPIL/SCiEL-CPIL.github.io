"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Multi-timescale sliding window LinUCB for non-stationary linear contextual bandits.
    
    Maintains two parallel estimators:
    - Short-window model: rapid detection of recent weight shifts
    - Long-window model: stable baseline estimation
    
    Selection uses a combined UCB that prioritizes short-window adaptation
    while using long-window for regularization/confidence.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Short window: fast adaptation
        self.gamma_s = 0.90  # faster forgetting for quicker adaptation
        self.A_s = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_s = [np.zeros(dim) for _ in range(n_arms)]
        
        # Medium window: intermediate timescale
        self.gamma_m = 0.95
        self.A_m = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_m = [np.zeros(dim) for _ in range(n_arms)]
        
        # Long window: stable baseline
        self.gamma_l = 0.97  # slower forgetting for stability
        self.A_l = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_l = [np.zeros(dim) for _ in range(n_arms)]
        
        # UCB parameters
        self.alpha_s = 1.5   # higher exploration weight for short window
        self.alpha_l = 0.2   # lower exploration weight for long window (regularization only)
        
        # Adaptive exploration: track recent rewards to adjust alpha
        self.recent_rewards = []
        self.reward_window_size = 10
        self.base_alpha = 1.5
        
        # Track model disagreement for adaptive exploration
        self.disagreement_history = []
        self.disagreement_window = 8
        
        # Track reward prediction errors to detect non-stationarity
        self.pred_errors = []
        self.error_window = 10
        
        # Track per-arm reward variance for adaptive exploration
        self.arm_rewards = {a: [] for a in range(n_arms)}
        self.arm_reward_window = 8

    def _get_theta(self, A, b):
        """Compute theta = A^{-1} b with regularization."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(self.d)

    def _ucb_score(self, context, A_s, b_s, A_m, b_m, A_l, b_l, alpha_s, alpha_l):
        """
        Compute UCB score using three timescales with adaptive weighting.
        
        Short window drives rapid adaptation, medium provides intermediate stability,
        and long window provides regularization. Weights adapt based on model agreement.
        """
        theta_s = self._get_theta(A_s, b_s)
        theta_m = self._get_theta(A_m, b_m)
        theta_l = self._get_theta(A_l, b_l)
        
        # Predicted rewards from each timescale
        pred_s = theta_s @ context
        pred_m = theta_m @ context
        pred_l = theta_l @ context
        
        # Confidence widths
        try:
            width_s = alpha_s * np.sqrt(context @ np.linalg.solve(A_s, context))
        except np.linalg.LinAlgError:
            width_s = alpha_s * np.sqrt(self.d)
        
        try:
            width_m = 0.5 * alpha_s * np.sqrt(context @ np.linalg.solve(A_m, context))
        except np.linalg.LinAlgError:
            width_m = 0.5 * alpha_s * np.sqrt(self.d)
        
        try:
            width_l = alpha_l * np.sqrt(context @ np.linalg.solve(A_l, context))
        except np.linalg.LinAlgError:
            width_l = alpha_l * np.sqrt(self.d)
        
        # Adaptive weighting based on model agreement
        # If short and medium agree, trust short more; if they disagree, blend more
        agreement_sm = 1.0 / (1.0 + abs(pred_s - pred_m))
        agreement_ml = 1.0 / (1.0 + abs(pred_m - pred_l))
        
        # Weighted prediction: short dominates when models agree, blend when they disagree
        w_s = 0.5 + 0.3 * agreement_sm
        w_m = 0.3 + 0.2 * agreement_ml
        w_l = 0.2
        total_w = w_s + w_m + w_l
        weighted_pred = (w_s * pred_s + w_m * pred_m + w_l * pred_l) / total_w
        
        # Combined exploration: short window primary, medium secondary, long for regularization
        score = weighted_pred + width_s + 0.3 * width_m + 0.1 * width_l
        
        return score

    def _adaptive_alpha(self):
        """
        Adapt exploration rate based on multiple non-stationarity signals:
        1. Reward variance
        2. Model disagreement across timescales
        3. Prediction error trends
        """
        if len(self.recent_rewards) < 5:
            return self.base_alpha
        
        recent = self.recent_rewards[-self.reward_window_size:]
        variance = np.var(recent)
        
        # Signal 1: Reward variance
        signal_var = np.clip(variance, 0, 3)
        
        # Signal 2: Model disagreement
        signal_disagree = 0.0
        if len(self.disagreement_history) >= 3:
            recent_disagreement = np.mean(self.disagreement_history[-self.disagreement_window:])
            signal_disagree = np.clip(recent_disagreement, 0, 2)
        
        # Signal 3: Prediction error trend (increasing errors suggest non-stationarity)
        signal_error = 0.0
        if len(self.pred_errors) >= 5:
            recent_errors = np.array(self.pred_errors[-self.error_window:])
            if len(recent_errors) >= 4:
                # Compare recent half vs earlier half
                mid = len(recent_errors) // 2
                error_trend = np.mean(recent_errors[mid:]) - np.mean(recent_errors[:mid])
                signal_error = np.clip(error_trend, 0, 1)
        
        # Combine signals with weights
        alpha = self.base_alpha * (1.0 + 0.4 * signal_var + 0.6 * signal_disagree + 0.3 * signal_error)
        
        return np.clip(alpha, 1.0, 3.0)

    def select(self, context):
        """Select arm based on multi-timescale UCB with adaptive exploration."""
        alpha_s = self._adaptive_alpha()
        
        scores = []
        for a in range(self.n):
            score = self._ucb_score(
                context,
                self.A_s[a], self.b_s[a],
                self.A_m[a], self.b_m[a],
                self.A_l[a], self.b_l[a],
                alpha_s, self.alpha_l
            )
            scores.append(score)
        
        best_arm = int(np.argmax(scores))
        
        # Track model disagreement for adaptive exploration
        if len(self.recent_rewards) >= 3:
            # Compute prediction differences between timescales for selected arm
            theta_s = self._get_theta(self.A_s[best_arm], self.b_s[best_arm])
            theta_m = self._get_theta(self.A_m[best_arm], self.b_m[best_arm])
            theta_l = self._get_theta(self.A_l[best_arm], self.b_l[best_arm])
            
            pred_s = theta_s @ context
            pred_m = theta_m @ context
            pred_l = theta_l @ context
            
            # Disagreement as max pairwise difference
            pred_diff = max(abs(pred_s - pred_m), abs(pred_m - pred_l), abs(pred_s - pred_l))
            self.disagreement_history.append(pred_diff)
            if len(self.disagreement_history) > self.disagreement_window * 2:
                self.disagreement_history = self.disagreement_history[-self.disagreement_window:]
        
        return best_arm

    def update(self, context, arm, reward):
        """Update all three timescale models."""
        # Update short window (fast forgetting)
        self.A_s[arm] = self.gamma_s * self.A_s[arm] + np.outer(context, context)
        self.b_s[arm] = self.gamma_s * self.b_s[arm] + reward * context
        
        # Update medium window (intermediate forgetting)
        self.A_m[arm] = self.gamma_m * self.A_m[arm] + np.outer(context, context)
        self.b_m[arm] = self.gamma_m * self.b_m[arm] + reward * context
        
        # Update long window (slow forgetting)
        self.A_l[arm] = self.gamma_l * self.A_l[arm] + np.outer(context, context)
        self.b_l[arm] = self.gamma_l * self.b_l[arm] + reward * context
        
        # Track recent rewards for adaptive exploration
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window_size * 2:
            self.recent_rewards = self.recent_rewards[-self.reward_window_size:]
        
        # Track prediction errors for non-stationarity detection
        theta_s = self._get_theta(self.A_s[arm], self.b_s[arm])
        pred = theta_s @ context
        pred_error = abs(reward - pred)
        self.pred_errors.append(pred_error)
        if len(self.pred_errors) > self.error_window * 2:
            self.pred_errors = self.pred_errors[-self.error_window:]
        
        # Track per-arm rewards for variance estimation
        self.arm_rewards[arm].append(reward)
        if len(self.arm_rewards[arm]) > self.arm_reward_window:
            self.arm_rewards[arm] = self.arm_rewards[arm][-self.arm_reward_window:]
# EVOLVE-BLOCK-END
