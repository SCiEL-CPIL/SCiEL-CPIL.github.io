# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Per-Arm State-Space Model with Regime-Switching Process Noise.
    
    Maintains two Kalman filter states per arm:
    - "stable" mode: low process noise (gradual drift)
    - "volatile" mode: high process noise (abrupt structural breaks)
    
    Uses a lightweight hidden Markov model to switch between modes based on
    innovation magnitude, distinguishing gradual drift from abrupt changes.
    
    Additionally incorporates:
    - Huber-loss innovation clipping for robustness to outliers
    - Context-norm adaptive process noise
    - UCB exploration with adaptive alpha
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Per-arm Kalman filter states
        # Each arm has two modes: 0=stable, 1=volatile
        self.P = []  # Covariance matrices [arm][mode]
        self.theta = []  # Weight vectors [arm][mode]
        self.mode_prob = []  # Probability of being in volatile mode [arm]
        
        # Mode parameters
        self.stable_noise = 0.005  # Low process noise
        self.volatile_noise = 0.1  # High process noise
        self.measurement_noise = 1.0
        self.switch_to_volatile = 0.05  # Base probability of switching to volatile
        self.switch_to_stable = 0.3  # Base probability of switching to stable
        self.innovation_threshold = 2.0  # Threshold for innovation magnitude
        
        # Huber loss parameters
        self.huber_delta = 2.0
        
        # UCB parameters
        self.base_alpha = 1.0
        self.alpha_decay = 0.999
        self.current_alpha = self.base_alpha
        
        # Initialize per-arm states
        for a in range(n_arms):
            # Stable mode: initialized with high uncertainty
            P_stable = np.eye(dim) * 1.0
            theta_stable = np.zeros(dim)
            
            # Volatile mode: initialized with higher uncertainty
            P_volatile = np.eye(dim) * 2.0
            theta_volatile = np.zeros(dim)
            
            self.P.append([P_stable, P_volatile])
            self.theta.append([theta_stable, theta_volatile])
            self.mode_prob.append([0.9, 0.1])  # Start in stable mode

        # Track recent innovations for regime detection
        self.recent_innovations = [[] for _ in range(n_arms)]
        self.max_innovations_history = 10

    def _time_remaining(self):
        return self.deadline - (time.time() - self.start_time)

    def _huber_loss(self, innovation, delta):
        """Compute Huber loss and its derivative for robustness."""
        abs_innov = abs(innovation)
        if abs_innov <= delta:
            return 0.5 * innovation * innovation, innovation
        else:
            return delta * (abs_innov - 0.5 * delta), delta * np.sign(innovation)

    def _compute_innovation(self, arm, context, reward):
        """Compute the innovation (prediction error) for the given arm."""
        # Use the blended estimate (weighted by mode probabilities)
        prob_v = self.mode_prob[arm][1]
        prob_s = 1.0 - prob_v
        
        # Blended theta and P
        theta_blend = prob_s * self.theta[arm][0] + prob_v * self.theta[arm][1]
        P_blend = prob_s * self.P[arm][0] + prob_v * self.P[arm][1]
        
        # Predicted reward
        pred = theta_blend @ context
        
        # Innovation
        innovation = reward - pred
        
        # Innovation variance
        innov_var = context @ P_blend @ context + self.measurement_noise
        
        return innovation, innov_var, pred

    def _update_mode_probabilities(self, arm, innovation, innov_var):
        """Update mode probabilities based on innovation magnitude."""
        z = abs(innovation) / (np.sqrt(innov_var) + 1e-8)
        
        # Likelihood of innovation under each mode
        # Stable mode: lower variance, so large innovations are less likely
        stable_var = self.stable_noise + self.measurement_noise
        volatile_var = self.volatile_noise + self.measurement_noise
        
        # Simple Gaussian likelihood ratio
        log_lik_stable = -0.5 * (innovation ** 2) / stable_var - 0.5 * np.log(2 * np.pi * stable_var)
        log_lik_volatile = -0.5 * (innovation ** 2) / volatile_var - 0.5 * np.log(2 * np.pi * volatile_var)
        
        # Update probabilities using Bayes rule
        prior_s = self.mode_prob[arm][0]
        prior_v = self.mode_prob[arm][1]
        
        # Posterior
        post_s = prior_s * np.exp(log_lik_stable)
        post_v = prior_v * np.exp(log_lik_volatile)
        
        total = post_s + post_v
        if total < 1e-15:
            total = 1.0
            post_s, post_v = 0.5, 0.5
            
        new_prob_s = post_s / total
        new_prob_v = post_v / total
        
        # Apply switching dynamics
        # With some probability, switch modes
        if z > self.innovation_threshold:
            # Large innovation: increase probability of volatile mode
            switch_boost = 0.1
            new_prob_v = min(0.95, new_prob_v + switch_boost)
            new_prob_s = 1.0 - new_prob_v
        else:
            # Small innovation: gradually return to stable
            new_prob_v = new_prob_v * 0.95 + 0.05 * 0.1
            new_prob_s = 1.0 - new_prob_v
        
        self.mode_prob[arm] = [new_prob_s, new_prob_v]

    def _kalman_update(self, arm, context, reward):
        """Perform Kalman filter update for both modes of the given arm."""
        prob_s, prob_v = self.mode_prob[arm]
        
        for mode in range(2):
            P = self.P[arm][mode]
            theta = self.theta[arm][mode]
            
            # Process noise: adaptive based on context norm
            ctx_norm = np.linalg.norm(context)
            adaptive_factor = 1.0 + 0.1 * ctx_norm
            
            if mode == 0:  # Stable
                Q = self.stable_noise * adaptive_factor
            else:  # Volatile
                Q = self.volatile_noise * adaptive_factor
            
            # Predicted covariance
            P_pred = P + Q * np.eye(self.d)
            
            # Kalman gain
            S = context @ P_pred @ context + self.measurement_noise
            K = P_pred @ context / (S + 1e-8)
            
            # Innovation
            innovation = reward - theta @ context
            
            # Huber loss: clip large innovations
            abs_innov = abs(innovation)
            if abs_innov > self.huber_delta * np.sqrt(S + 1e-8):
                # Use clipped innovation
                clipped_innov = self.huber_delta * np.sqrt(S + 1e-8) * np.sign(innovation)
            else:
                clipped_innov = innovation
            
            # Update
            theta_new = theta + K * clipped_innov
            P_new = (np.eye(self.d) - np.outer(K, context)) @ P_pred
            
            # Ensure symmetry and positive semi-definite
            P_new = (P_new + P_new.T) / 2.0
            # Add small regularization to maintain PD
            P_new += 1e-6 * np.eye(self.d)
            
            self.theta[arm][mode] = theta_new
            self.P[arm][mode] = P_new

    def select(self, context):
        # Deadline guard
        if self._time_remaining() < 5.0:
            # Return greedy action based on current estimates
            scores = []
            for a in range(self.n):
                prob_s, prob_v = self.mode_prob[a]
                theta_blend = prob_s * self.theta[a][0] + prob_v * self.theta[a][1]
                scores.append(theta_blend @ context)
            return int(np.argmax(scores))
        
        context = np.asarray(context, dtype=np.float64)
        
        # Compute UCB scores for each arm
        scores = []
        for a in range(self.n):
            prob_s, prob_v = self.mode_prob[a]
            
            # Blended theta and P
            theta_blend = prob_s * self.theta[a][0] + prob_v * self.theta[a][1]
            P_blend = prob_s * self.P[a][0] + prob_v * self.P[a][1]
            
            # Predicted reward
            mean = theta_blend @ context
            
            # Uncertainty
            var = context @ P_blend @ context
            std = np.sqrt(var)
            
            # Adaptive alpha: decays over time but responds to volatility
            volatility_factor = 1.0 + prob_v * 2.0
            alpha = self.current_alpha * volatility_factor
            
            scores.append(mean + alpha * std)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if self._time_remaining() < 5.0:
            return
        
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Update mode probabilities based on innovation
        innovation, innov_var, pred = self._compute_innovation(arm, context, reward)
        self._update_mode_probabilities(arm, innovation, innov_var)
        
        # Perform Kalman update for both modes
        self._kalman_update(arm, context, reward)
        
        # Track recent innovations
        self.recent_innovations[arm].append(innovation)
        if len(self.recent_innovations[arm]) > self.max_innovations_history:
            self.recent_innovations[arm].pop(0)
        
        # Decay alpha
        self.current_alpha *= self.alpha_decay
# EVOLVE-BLOCK-END
