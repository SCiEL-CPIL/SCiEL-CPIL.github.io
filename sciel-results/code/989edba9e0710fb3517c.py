"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-arm RLS with adaptive forgetting factor based on prediction error."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Forgetting factor base and adaptation parameters
        self.lam_base = 0.95
        self.lam_min = 0.85
        self.lam_max = 0.999
        self.gamma = 0.05  # Adaptation rate for lambda based on error
        
        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) for _ in range(n_arms)]  # Covariance (inverse of A)
        self.last_reward = [0.0] * n_arms
        self.last_context = [np.zeros(dim) for _ in range(n_arms)]
        self.last_arm_selected = -1
        self.last_selected_context = None
        self.last_selected_arm = -1
        
        # Track prediction errors for adaptive lambda
        self.pred_error = [0.0] * n_arms
        self.lambda_adapt = [self.lam_base] * n_arms
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return the arm with highest estimated reward (greedy)
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Predicted reward
            pred = self.theta[a] @ context
            # Uncertainty term
            uncertainty = np.sqrt(context @ self.P[a] @ context)
            # UCB score
            score = pred + 0.5 * uncertainty
            scores.append(score)
        
        arm = int(np.argmax(scores))
        self.last_selected_arm = arm
        self.last_selected_context = context.copy()
        return arm

    def update(self, context, arm, reward):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            return

        # Compute prediction error for the selected arm
        pred = self.theta[arm] @ context
        error = reward - pred
        self.pred_error[arm] = error
        
        # Adapt lambda based on prediction error
        # Large error -> smaller lambda (faster forgetting)
        abs_error = abs(error)
        self.lambda_adapt[arm] = self.lam_base - self.gamma * abs_error
        self.lambda_adapt[arm] = np.clip(self.lambda_adapt[arm], self.lam_min, self.lam_max)
        
        lam = self.lambda_adapt[arm]

        # RLS update
        # P_new = (1/lam) * (P - P @ x @ (lam + x @ P @ x)^-1 @ x @ P)
        # Using matrix inversion lemma
        P = self.P[arm]
        x = context
        xPx = x @ P @ x
        K = P @ x / (lam + xPx)  # Kalman gain
        
        # Update theta
        self.theta[arm] = self.theta[arm] + K * error
        
        # Update P
        self.P[arm] = (1.0 / lam) * (P - np.outer(K, x))
        
        # Ensure symmetry
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
# EVOLVE-BLOCK-END
