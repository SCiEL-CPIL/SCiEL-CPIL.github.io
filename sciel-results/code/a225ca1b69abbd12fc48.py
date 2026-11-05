"""Dual-timescale Kalman filter for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Dual-timescale Kalman filter approach for non-stationary linear contextual bandit.
    
    State-space model per arm:
    - theta_t = theta_{t-1} + w_t  (random walk with process noise Q)
    - r_t = theta_t^T x_t + v_t    (observation with noise R)
    
    We maintain:
    - Fast timescale: standard Kalman filter for theta estimation
    - Slow timescale: adaptive process noise Q that tracks drift
    
    The process noise Q is scaled by the ratio of recent prediction error to
    expected observation noise, allowing the filter to automatically increase
    uncertainty when the model is wrong (indicating drift) and decrease it
    when predictions are accurate (indicating stationarity).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = self.t0 + 110.0  # safety margin before 120s kill

        # Per-arm Kalman filter state
        # P: covariance of theta estimate (d x d)
        # theta_hat: current estimate of theta (d,)
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.theta_hat = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive process noise
        # Base process noise scale
        self.q_base = 0.05
        # Observation noise
        self.r_obs = 1.0

        # Per-arm innovation statistics for adaptive Q
        self.innovation_ema = [0.0] * n_arms  # EMA of squared innovations
        self.innovation_count = [0] * n_arms

        # Exploration: use Thompson sampling on the Kalman posterior
        # or UCB based on posterior variance
        self.use_ts = True  # Thompson sampling
        self.ucb_beta = 1.5  # UCB exploration parameter

        # Deadline guard
        self._deadline_reached = False

    def _check_deadline(self):
        if self._deadline_reached:
            return False
        if (time.time() - self.t0) >= self.deadline:
            self._deadline_reached = True
            return False
        return True

    def _adaptive_process_noise(self, arm):
        """
        Compute adaptive process noise Q for the given arm.
        Q scales with the ratio of recent prediction error to expected noise.
        """
        if self.innovation_count[arm] < 3:
            return np.eye(self.d) * self.q_base

        # Expected innovation variance if model is correct: r_obs + x^T P x
        # We use the EMA of actual squared innovations vs expected
        # If actual >> expected, model is stale -> increase Q
        # If actual <= expected, model is good -> decrease Q toward base

        # Use a simple heuristic: scale Q by sqrt(innovation_ema / r_obs)
        # capped to avoid explosion
        ratio = self.innovation_ema[arm] / (self.r_obs + 1e-9)
        ratio = min(ratio, 10.0)  # cap
        q_scale = self.q_base * (0.5 + 0.5 * ratio)
        return np.eye(self.d) * q_scale

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback: use current theta estimates
            scores = [self.theta_hat[a] @ ctx for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Predict step: add process noise to covariance
            Q = self._adaptive_process_noise(a)
            P_pred = self.P[a] + Q

            # Observation model: r = theta^T x + v
            # Innovation covariance: S = x^T P_pred x + R
            Px = P_pred @ ctx
            S = ctx @ Px + self.r_obs

            # For scoring, we want E[r] = theta_hat^T x
            # and Var[r] = S
            mean_reward = self.theta_hat[a] @ ctx

            if self.use_ts:
                # Thompson sampling: sample from posterior
                # theta ~ N(theta_hat, P_pred)
                try:
                    L = np.linalg.cholesky(P_pred + 1e-9 * np.eye(self.d))
                    z = np.random.randn(self.d)
                    theta_sample = self.theta_hat[a] + L @ z
                    score = theta_sample @ ctx
                except np.linalg.LinAlgError:
                    score = mean_reward
            else:
                # UCB: mean + beta * std
                std_reward = np.sqrt(max(S, 0.0))
                score = mean_reward + self.ucb_beta * std_reward

            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        self.counts[arm] += 1

        # Predict step
        Q = self._adaptive_process_noise(arm)
        P_pred = self.P[arm] + Q

        # Update step (Kalman)
        Px = P_pred @ ctx
        S = ctx @ Px + self.r_obs

        if S > 1e-12:
            # Kalman gain: K = P_pred x / S
            K = Px / S
            # Innovation
            innovation = reward - self.theta_hat[arm] @ ctx
            # Update theta
            self.theta_hat[arm] = self.theta_hat[arm] + K * innovation
            # Update covariance (Joseph form for numerical stability)
            I_Kx = np.eye(self.d) - np.outer(K, ctx)
            self.P[arm] = I_Kx @ P_pred @ I_Kx.T + np.outer(K, K) * self.r_obs
            # Symmetrize
            self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)
        else:
            # Degenerate case: don't update
            pass

        # Update innovation statistics for adaptive Q
        if self.innovation_count[arm] == 0:
            self.innovation_ema[arm] = innovation ** 2
        else:
            alpha = 0.1
            self.innovation_ema[arm] = (1 - alpha) * self.innovation_ema[arm] + alpha * innovation ** 2
        self.innovation_count[arm] += 1
# EVOLVE-BLOCK-END
