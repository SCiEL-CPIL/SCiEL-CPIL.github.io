"""
Non-stationary linear contextual bandit policy using Recursive Least Squares (RLS).
Maintains explicit inverse covariance (P) for O(d^2) per-step updates.
Squared-error-modulated forgetting factor provides a principled regime-change
detector: high error triggers rapid re-initialization (low lambda) while low
error preserves learned weights (high lambda).
"""

import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm RLS state: theta (weight vector) and P (inverse covariance / gain matrix)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P as large identity to encourage early exploration
        self.P = [10.0 * np.eye(dim) for _ in range(n_arms)]

        # Forgetting factor parameters
        self.lambda_min = 0.80   # Aggressive forgetting when error is large
        self.lambda_max = 0.995  # Slow forgetting when error is small
        self.error_scale = 0.2   # Scale for normalizing squared error

        # UCB exploration parameter
        self.alpha = 2.5

        # Track last predictions for error computation
        self._last_preds = [0.0] * n_arms

        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0

        # Round counter
        self.total_rounds = 0

    def _get_lambda(self, squared_error):
        """
        Compute adaptive forgetting factor from squared prediction error.
        High squared error -> low lambda (forget fast, re-learn quickly).
        Low squared error -> high lambda (preserve learned model).
        Uses a smooth exponential mapping.
        """
        # Normalize squared error
        norm_err = squared_error / (self.error_scale * self.error_scale)

        # Map to [lambda_min, lambda_max] using exponential decay
        # When norm_err is 0, lambda = lambda_max
        # When norm_err is large, lambda -> lambda_min
        # lambda = lambda_min + (lambda_max - lambda_min) * exp(-norm_err / tau)
        tau = 1.0
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * np.exp(-norm_err / tau)
        return float(lam)

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        for a in range(self.n):
            # Mean prediction
            mu = self.theta[a] @ context

            # Store prediction for error tracking
            self._last_preds[a] = mu

            # UCB: variance term from P
            var = context @ self.P[a] @ context
            std = np.sqrt(max(0.0, var))

            scores[a] = mu + self.alpha * std

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)

        # Deadline check
        if time.time() > self._deadline:
            return

        self.total_rounds += 1

        # Compute prediction error (squared)
        pred = self._last_preds[arm]
        err = reward - pred
        sq_err = err * err

        # Adaptive forgetting factor
        lam = self._get_lambda(sq_err)

        # RLS update with forgetting
        # P_new = (1/lam) * (P - (P @ x) (x^T @ P) / (lam + x^T @ P @ x))
        # theta_new = theta + P_new @ x * (reward - theta^T x)

        P = self.P[arm]
        theta = self.theta[arm]

        # Compute Px = P @ x
        Px = P @ context

        # Compute x^T P x (scalar)
        xPtx = context @ Px

        # Denominator
        denom = lam + xPtx

        # Update P
        # P_new = (1/lam) * P - (1/lam) * (Px * Px^T) / denom
        #        = (1/lam) * (P - (Px @ Px.T) / denom)
        self.P[arm] = (1.0 / lam) * (P - np.outer(Px, Px) / denom)

        # Update theta
        # theta_new = theta + P_new @ x * err
        # P_new @ x = (1/lam) * (P @ x - Px * (x^T P x) / denom)
        #           = (1/lam) * (Px - Px * xPtx / denom)
        #           = (1/lam) * Px * (1 - xPtx/denom)
        #           = (1/lam) * Px * (lam/denom)
        #           = Px / denom
        gain = Px / denom
        self.theta[arm] = theta + gain * err
