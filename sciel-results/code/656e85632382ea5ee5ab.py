"""Non-stationary linear contextual bandit with Kalman-filter adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Kalman-filter-based adaptive LinUCB for non-stationary linear bandits.
    
    Models each arm's weight vector as a random walk with process noise Q.
    Uses innovation sequences to estimate observation noise R and process noise Q
    online, driving principled adaptive forgetting instead of heuristic thresholds.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Per-arm: covariance matrix P (inverse of precision A), mean estimate theta
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Process noise (random walk) per arm - scalar times identity initially
        self.Q = [1e-3 * np.eye(dim) for _ in range(n_arms)]
        self.Q_scalar = [1e-3] * n_arms  # Track scalar component for adaptation
        
        # Observation noise estimates per arm
        self.R = [1.0] * n_arms  # Initial observation noise variance
        
        # EMA of squared innovations for R estimation
        self.innov_ema = [1.0] * n_arms
        self.innov_ema_beta = 0.95
        
        # EMA of innovation magnitude for Q adaptation
        self.innov_abs_ema = [0.0] * n_arms
        self.q_ema_beta = 0.9
        
        # UCB exploration parameter
        self.alpha = 1.0
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill
        
        # Bounds for Q adaptation
        self.Q_min = 1e-5
        self.Q_max = 1.0
        self.R_min = 1e-3
        self.R_max = 10.0

    def _predict(self, arm):
        """Kalman prediction step: propagate state covariance with process noise."""
        # P_pred = P + Q (random walk)
        self.P[arm] = self.P[arm] + self.Q[arm]
        # theta stays the same in prediction (random walk has zero mean drift)

    def _get_mu_and_var(self, arm, context):
        """Compute predicted mean and variance for an arm given context."""
        mu = self.theta[arm] @ context
        var = context @ self.P[arm] @ context
        return mu, max(var, 1e-12)

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                self._predict(a)
                mu, var = self._get_mu_and_var(a, context)
                score = mu + self.alpha * np.sqrt(var)
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        scores = []
        for a in range(self.n):
            self._predict(a)
            mu, var = self._get_mu_and_var(a, context)
            score = mu + self.alpha * np.sqrt(var)
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Kalman update step for the selected arm
        # Observation: y = context @ theta + noise
        # H = context (row vector), so H @ theta = context @ theta
        
        # Innovation (prediction error)
        mu_pred = self.theta[arm] @ context
        innovation = reward - mu_pred
        
        # Innovation variance: S = H @ P @ H^T + R = context^T P context + R
        S = context @ self.P[arm] @ context + self.R[arm]
        S = max(S, 1e-10)
        
        # Kalman gain: K = P @ H^T / S
        # P @ H^T = P @ context (column vector)
        K = (self.P[arm] @ context) / S
        
        # Update theta: theta += K @ innovation
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Update P: P = (I - K @ H) @ P
        # Since H is a row vector (context), K @ H = outer(K, context)
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ self.P[arm]
        
        # Symmetrize for numerical stability
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)
        
        # --- Adaptive noise estimation ---
        
        # Update R estimate using innovation EMA
        # If |innovation| is consistently larger than expected, R should increase
        innov_sq = innovation ** 2
        self.innov_ema[arm] = self.innov_ema_beta * self.innov_ema[arm] + (1 - self.innov_ema_beta) * innov_sq
        
        # R should be approximately the expected innovation variance
        # Clamp to reasonable bounds
        self.R[arm] = np.clip(self.innov_ema[arm], self.R_min, self.R_max)
        
        # Update Q estimate:
        # If innovations are consistently large relative to current uncertainty,
        # the process noise Q should increase to allow faster adaptation
        innov_abs = np.abs(innovation)
        self.innov_abs_ema[arm] = self.q_ema_beta * self.innov_abs_ema[arm] + (1 - self.q_ema_beta) * innov_abs
        
        # Heuristic: Q_scalar proportional to recent innovation magnitude
        # Scale by a factor to keep Q in reasonable range
        q_target = self.innov_abs_ema[arm] * 0.1
        self.Q_scalar[arm] = np.clip(q_target, self.Q_min, self.Q_max)
        self.Q[arm] = self.Q_scalar[arm] * np.eye(self.d)
# EVOLVE-BLOCK-END
