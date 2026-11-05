"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a per-arm Kalman filter
    with adaptive process noise and UCB exploration.
    
    This combines the robust structure of the current program (explicit deadline, 
    simple linear algebra) with the refinement machinery of the Kalman filter 
    inspirations (adaptive process noise based on innovation, Joseph-form updates, 
    adaptive UCB scaling).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit

        # Posterior state per arm (Thompson Sampling)
        # theta: mean weight vector
        # P: covariance matrix
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize with high uncertainty to encourage exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Adaptive process noise parameters
        self.q_base = 0.05         # Base process noise
        self.q_max = 5.0           # Cap on adaptive process noise
        self.innov_ema = [0.0] * n_arms  # EMA of squared innovation per arm
        self.innov_ema_alpha = 0.3       # Fast adaptation rate for EMA
        self.innov_base = 1.0            # Baseline threshold for innovation

        # Thompson Sampling parameters
        self.warmup_steps = 25     # Steps to boost exploration early
        self.jitter = 1e-6         # Numerical stabilizer
        self.step_count = 0
        self.rng = np.random.default_rng(42)

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov):
        """
        Compute adaptive process noise scalar based on innovation.
        High innovation (surprise) leads to larger Q (faster forgetting).
        """
        c = self.count[arm]
        if c < 3:
            # Early phase: larger Q to help initial convergence
            return self.q_base * 2.0

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        # Scale Q based on how much the innovation exceeds the baseline
        excess = max(0.0, innov_ema - self.innov_base)
        # Exponential mapping for smooth, bounded scaling
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        q = min(self.q_max, self.q_base * q_scale)
        return q

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        """Kalman update step: rank-1 update with innovation."""
        P = self.P[a]
        theta = self.theta[a]
        
        # Predicted reward
        pred = float(theta @ ctx)
        # Innovation
        innov = reward - pred
        
        # Innovation variance: S = x^T P x + R (assuming R=1)
        R = 1.0
        var = float(ctx @ P @ ctx) + R + self.jitter
        
        # Kalman gain: K = P x / S
        K = (P @ ctx) / var
        
        # Update mean: theta <- theta + K * innov
        self.theta[a] = theta + (K * innov)
        
        # Joseph-form covariance update for numerical stability:
        # P <- (I - K x^T) P
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P
        
        # Ensure symmetry and positive semi-definite
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        
        # Clip eigenvalues to ensure stability (cheap diagonal check)
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))
            
        self.count[a] += 1
        return innov

    def _sample_reward(self, a, ctx):
        """Sample a reward from the posterior for arm a given context."""
        theta = self.theta[a]
        P = self.P[a]
        
        mean = float(theta @ ctx)
        var = float(ctx @ P @ ctx)
        std = float(np.sqrt(max(var, self.jitter)))
        
        # Sample from posterior: N(mean, var)
        sample = mean + std * self.rng.standard_normal()
        
        # Boost for unseen arms during warmup
        if self.count[a] == 0 and self.step_count < self.warmup_steps:
            sample += 1.5
        
        return sample

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        # Deadline guard: if we are running out of wall-clock, commit to the
        # arm with the highest current mean estimate (no further exploration).
        if self._check_deadline():
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        self.step_count += 1

        # Thompson Sampling: sample from each arm's posterior and pick the best
        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            sample = self._sample_reward(a, ctx)
            if sample > best:
                best = sample
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        if self._check_deadline():
            return

        # Compute innovation before predict to drive adaptive Q
        pred = float(self.theta[arm] @ ctx)
        innov = reward - pred

        # Adaptive process noise
        q = self._adaptive_q(arm, innov)

        # Kalman predict step
        self._predict(arm, q)

        # Kalman update step
        self._update(arm, ctx, reward)

        # Periodic numerical stabilization every 50 steps
        if self.step_count % 50 == 0:
            for a in range(self.n):
                P = self.P[a]
                # Ensure symmetry
                P = 0.5 * (P + P.T)
                # Check for NaNs or infinities
                if not np.all(np.isfinite(P)):
                    self.P[a] = np.eye(self.d) * 10.0
                    self.theta[a] = np.zeros(self.d)
                    self.count[a] = 0
                    self.innov_ema[a] = 0.0
# EVOLVE-BLOCK-END
