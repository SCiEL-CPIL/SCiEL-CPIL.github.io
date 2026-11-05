"""Non-stationary linear contextual bandit with Regime-Switching Kalman Filter."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a Regime-Switching Kalman Filter.
    
    Core Idea:
    - Model arm weights as a random walk with regime-switching process noise.
    - Two regimes: "Stable" (low Q) and "Volatile" (high Q).
    - Use a lightweight Hidden Markov Model (HMM) to estimate regime probabilities
      based on innovation magnitude.
    - Blend the Kalman Filter updates from both regimes weighted by their posterior probabilities.
    
    State:
    - theta[a]: Estimated weight vector for arm a.
    - P[a]: Covariance matrix for theta[a].
    - regime_prob[a]: Probability of being in "Volatile" regime.
    
    Update:
    - Compute innovation.
    - Update regime probabilities using HMM transition and emission models.
    - Blend Q and perform a single KF update with the blended Q.
      (Alternatively, maintain two parallel filters, but blending Q is simpler and often effective).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P with large value to allow rapid initial learning
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        
        # Regime parameters
        # Q_stable: low process noise
        # Q_volatile: high process noise
        self.q_stable = 0.01
        self.q_volatile = 1.0
        
        # HMM Parameters
        # Transition probabilities
        # P(stable->stable), P(stable->volatile), P(volatile->stable), P(volatile->volatile)
        self.p_ss = 0.95
        self.p_sv = 0.05
        self.p_vs = 0.20
        self.p_vv = 0.80
        
        # Emission model parameters
        # We model innovation z ~ N(0, sigma^2)
        # In stable regime, innovation is small (sigma_stable)
        # In volatile regime, innovation is large (sigma_volatile)
        # These are heuristic scales relative to observation noise R=1.0
        self.sigma_stable = 1.0
        self.sigma_volatile = 3.0
        
        # Initial regime probability (probability of being in Volatile)
        self.regime_prob = [0.1] * n_arms  # Start with low probability of volatility
        
        # Exploration
        self.alpha = 1.5
        self.alpha_decay = 0.999
        self.alpha_min = 0.5
        
        # Tracking
        self.round_num = 0
        self.arm_counts = [0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _update_regime_prob(self, arm, innovation):
        """
        Update regime probability using HMM.
        p_new(volatile) = (p_old(volatile)*p_vv + p_old(stable)*p_sv) * p_obs(volatile)
                         / (p_old(volatile)*p_vv + p_old(stable)*p_sv) * p_obs(volatile)
                         + (p_old(volatile)*p_vs + p_old(stable)*p_ss) * p_obs(stable)
        """
        p_v = self.regime_prob[arm]
        p_s = 1.0 - p_v
        
        # Predicted probabilities
        p_v_pred = p_v * self.p_vv + p_s * self.p_sv
        p_s_pred = p_v * self.p_vs + p_s * self.p_ss
        
        # Emission probabilities (Gaussian likelihood)
        # Use log-space for stability if needed, but direct is fine for small numbers
        # P(innovation | regime)
        # z = innovation
        # pdf_stable = exp(-z^2 / (2 * sigma_stable^2))
        # pdf_volatile = exp(-z^2 / (2 * sigma_volatile^2))
        
        z2 = innovation * innovation
        # Avoid underflow by normalizing
        log_pdf_s = -0.5 * z2 / (self.sigma_stable * self.sigma_stable)
        log_pdf_v = -0.5 * z2 / (self.sigma_volatile * self.sigma_volatile)
        
        # Softmax for stability
        max_log = max(log_pdf_s, log_pdf_v)
        pdf_s = np.exp(log_pdf_s - max_log)
        pdf_v = np.exp(log_pdf_v - max_log)
        
        # Posterior
        numerator_v = p_v_pred * pdf_v
        numerator_s = p_s_pred * pdf_s
        denominator = numerator_v + numerator_s
        
        if denominator < 1e-10:
            new_p_v = 0.5
        else:
            new_p_v = numerator_v / denominator
            
        # Clip to avoid numerical issues
        self.regime_prob[arm] = max(1e-6, min(1.0 - 1e-6, new_p_v))

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # Uncertainty: context^T P context
            P_ctx = self.P[a] @ context
            var_term = context @ P_ctx
            if var_term < 0:
                var_term = 0.0
            bonus = self.alpha * np.sqrt(var_term)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1
        
        # Decay alpha slightly to reduce exploration over time
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
        # 1. Calculate Innovation to update regime probability
        # We need the predicted innovation.
        # y_hat = theta @ context
        # But wait, the regime update should happen BEFORE the state update?
        # Usually:
        # 1. Predict state (theta_pred, P_pred)
        # 2. Calculate innovation (y - theta_pred @ context)
        # 3. Update regime probability based on innovation
        # 4. Determine Q based on regime probability
        # 5. Perform KF update with that Q
        
        # However, Q is needed for P_pred.
        # This creates a circular dependency if we want Q to depend on the current innovation.
        # Standard approach: Use the PREVIOUS innovation to set Q for the current step,
        # OR assume a baseline Q and update regime prob AFTER.
        # Let's use the previous regime probability to set Q, then update regime prob
        # using the current innovation. This is a standard one-step lag.
        
        p_v = self.regime_prob[arm]
        
        # Blend Q
        q = p_v * self.q_volatile + (1.0 - p_v) * self.q_stable
        
        # 1. Prediction Step
        P_pred = self.P[arm] + q * np.eye(self.d)
        
        # 2. Update Step (Kalman Filter)
        R = 1.0
        
        # Innovation
        y_hat = self.theta[arm] @ context
        innovation = reward - y_hat
        
        # Innovation Covariance
        S = context @ P_pred @ context + R
        
        # Kalman Gain
        if S > 1e-10:
            K = P_pred @ context / S
        else:
            K = np.zeros(self.d)
        
        # State Update
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Covariance Update (Joseph form for stability)
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * R
        
        # Symmetrize
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        # 3. Update Regime Probability
        self._update_regime_prob(arm, innovation)
        
        # 4. Update Tracking Stats
        self.arm_counts[arm] += 1

# EVOLVE-BLOCK-END
