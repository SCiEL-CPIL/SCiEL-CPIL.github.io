# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using per-arm Kalman filters
    with adaptive random-walk process noise.
    
    Each arm maintains:
      - Information-form linear model (A = P^{-1}, b = P^{-1} theta)
      - Adaptive process noise Q scaled by recent innovation magnitude
      - UCB exploration bonus from posterior covariance
    
    The Kalman filter tracks drift in theta via process noise Q,
    and Q is adapted based on how much the prediction error exceeds
    the expected noise level (innovation-based adaptation).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = 115.0  # seconds (safety margin before 120s limit)

        # Per-arm Kalman filter state (information form)
        # A = P^{-1}, b = P^{-1} * theta
        self.A = [np.eye(dim) * 100.0 for _ in range(n_arms)]  # P = I/100 initially
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive process noise parameters
        self.Q_base = np.eye(dim) * 0.01  # base process noise
        self.Q_scale = [1.0 for _ in range(n_arms)]  # adaptive scale factor
        self.innovation_history = [0.0 for _ in range(n_arms)]  # EMA of innovation magnitude
        
        # Observation noise variance (assumed)
        self.R = 1.0
        
        # Exploration coefficient
        self.beta = 1.5
        
        # Track number of observations per arm
        self.n_obs = [0 for _ in range(n_arms)]

    def _predict(self, arm):
        """
        Kalman predict step: add process noise to covariance.
        In information form: P_pred = P + Q, so A_pred = (P+Q)^{-1}
        We update A by solving A @ (P + Q) = I, which is A @ (A^{-1} + Q) = I
        => A_pred = (A^{-1} + Q)^{-1}
        
        For efficiency, we use the matrix inversion lemma:
        If P = A^{-1}, then P_pred = P + Q
        A_pred = (P + Q)^{-1}
        
        We compute this via: A_pred = A @ (I + Q @ A)^{-1}
        """
        Q_adapted = self.Q_base * self.Q_scale[arm]
        # A_pred = A @ (I + Q @ A)^{-1}
        # Using solve for stability
        Q_A = Q_adapted @ self.A[arm]
        I_plus_QA = np.eye(self.d) + Q_A
        A_pred = self.A[arm] @ np.linalg.inv(I_plus_QA)
        return A_pred

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline - 5.0:
            # Return best arm based on current estimates
            scores = []
            for a in range(self.n):
                A_inv = np.linalg.inv(self.A[a])
                theta = A_inv @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Predict step (add process noise)
            A_pred = self._predict(a)
            
            # Compute mean and variance
            A_inv = np.linalg.inv(A_pred)
            theta = A_inv @ self.b[a]
            mean = theta @ context
            
            # Posterior variance for this context
            var = context @ A_inv @ context + self.R
            bonus = self.beta * np.sqrt(var)
            
            scores.append(mean + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self._t0 > self._deadline - 5.0:
            return

        # Kalman update step
        # Predict
        A_pred = self._predict(arm)
        A_pred_inv = np.linalg.inv(A_pred)
        theta_pred = A_pred_inv @ self.b[arm]
        
        # Compute innovation
        H = context  # observation model: y = theta^T x + noise
        y_pred = theta_pred @ context
        innovation = reward - y_pred
        
        # Innovation variance
        S = context @ A_pred_inv @ context + self.R
        
        # Kalman gain
        K = A_pred_inv @ context / S
        
        # Update
        theta_upd = theta_pred + K * innovation
        A_upd_inv = A_pred_inv - np.outer(K, context)
        
        # Store in information form
        self.A[arm] = np.linalg.inv(A_upd_inv)
        self.b[arm] = self.A[arm] @ theta_upd
        
        self.n_obs[arm] += 1
        
        # Adapt process noise based on innovation
        # If innovation is large relative to expected S, increase Q
        expected_innovation_std = np.sqrt(S)
        innovation_mag = abs(innovation)
        
        # Update EMA of innovation magnitude
        alpha = 0.1
        self.innovation_history[arm] = (1 - alpha) * self.innovation_history[arm] + alpha * innovation_mag
        
        # Scale Q: if innovations are consistently larger than expected, increase Q
        # This indicates the model is not tracking drift well
        if expected_innovation_std > 1e-6:
            ratio = innovation_mag / expected_innovation_std
            # Increase Q scale if ratio > 1 (unexpected large innovation)
            if ratio > 1.0:
                self.Q_scale[arm] = min(self.Q_scale[arm] * (1 + 0.1 * (ratio - 1)), 10.0)
            else:
                # Slowly decay Q scale back toward 1
                self.Q_scale[arm] = max(self.Q_scale[arm] * 0.99, 0.1)
        else:
            self.Q_scale[arm] = 1.0
# EVOLVE-BLOCK-END
