"""Non-stationary linear contextual bandit with Thompson Sampling and adaptive posterior inflation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with Thompson Sampling:
    - Per-arm posterior Gaussian beliefs over weight vectors
    - Adaptive covariance inflation based on standardized innovation magnitude
    - Probabilistic arm selection for natural exploration/exploitation balance
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Per-arm posterior: mean (theta) and covariance (V)
        # Initialize with identity covariance (uniform prior)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.V = [np.eye(dim) * 1.0 for _ in range(n_arms)]

        # Innovation tracking for adaptive inflation
        self.innovation_ema = [0.0] * n_arms
        self.innovation_ema_beta = 0.95  # slow EMA for stability

        # Counts per arm
        self.counts = [0] * n_arms
        self.total_rounds = 0

        # Base inflation rate (small, so we don't over-inflate)
        self.base_inflation = 0.01
        self.max_inflation = 0.5
        self.min_inflation = 0.001

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_inflation(self, arm):
        """
        Compute adaptive covariance inflation based on innovation magnitude.
        High innovation -> more inflation (faster forgetting/adaptation)
        Low innovation -> less inflation (preserve learned structure)
        """
        c = self.counts[arm]
        if c < 5:
            return self.base_inflation

        # Use EMA of standardized innovations
        # innovation_ema tracks average squared innovation
        innov = self.innovation_ema[arm]

        # Map innovation magnitude to inflation rate
        # Use a softplus-like mapping: inflation increases with innovation
        # Scale so that typical innovations give moderate inflation
        scale = 0.5  # tuning parameter
        inflation = self.base_inflation + scale * innov

        # Clamp to reasonable range
        inflation = max(self.min_inflation, min(self.max_inflation, inflation))

        return inflation

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy (mean-based)
        if not self._check_deadline():
            scores = np.array([self.theta[a] @ context for a in range(self.n)])
            return int(np.argmax(scores))

        # Thompson Sampling: sample from posterior for each arm
        sampled_scores = []
        for a in range(self.n):
            # Sample theta from N(self.theta[a], self.V[a])
            # Use Cholesky for efficient sampling
            try:
                L = np.linalg.cholesky(self.V[a] + 1e-8 * np.eye(self.d))
            except np.linalg.LinAlgError:
                # Fallback: use identity if Cholesky fails
                L = np.eye(self.d)

            z = np.random.randn(self.d)
            theta_sample = self.theta[a] + L @ z
            score = theta_sample @ context
            sampled_scores.append(score)

        return int(np.argmax(sampled_scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1
        self.total_rounds += 1

        # Compute innovation (prediction error)
        pred = self.theta[arm] @ context
        innovation = reward - pred
        innovation_sq = innovation ** 2

        # Update innovation EMA
        self.innovation_ema[arm] = (self.innovation_ema_beta * self.innovation_ema[arm] +
                                     (1 - self.innovation_ema_beta) * innovation_sq)

        # Adaptive inflation
        inflation = self._adaptive_inflation(arm)

        # Bayesian update:
        # Prior: theta ~ N(theta_old, V_old)
        # Likelihood: reward | theta ~ N(theta^T x, sigma^2) with sigma^2 = 1
        # Posterior: theta | reward ~ N(theta_new, V_new)

        V_old = self.V[arm]
        theta_old = self.theta[arm]

        # Kalman filter update
        # Innovation covariance: S = x^T V x + 1
        Vx = V_old @ context
        S = context @ Vx + 1.0

        if S > 1e-12:
            # Gain
            K = Vx / S

            # Update mean
            self.theta[arm] = theta_old + K * innovation

            # Update covariance (Joseph form for numerical stability)
            I_KxT = np.eye(self.d) - np.outer(K, context)
            self.V[arm] = I_KxT @ V_old @ I_KxT.T + K @ (K * 1.0)  # R = 1

        # Adaptive inflation: add to covariance to handle non-stationarity
        self.V[arm] += inflation * np.eye(self.d)

        # Ensure positive semi-definiteness
        # Symmetrize
        self.V[arm] = (self.V[arm] + self.V[arm].T) / 2.0

        # Clip eigenvalues to be positive
        try:
            eigvals, eigvecs = np.linalg.eigh(self.V[arm])
            eigvals = np.maximum(eigvals, 1e-6)
            self.V[arm] = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            # Fallback: add small identity
            self.V[arm] += 1e-4 * np.eye(self.d)
# EVOLVE-BLOCK-END
