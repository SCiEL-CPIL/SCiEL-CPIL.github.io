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

        # Per-arm posterior parameters: mean (theta) and covariance (Sigma)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.Sigma = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # initial high uncertainty
        self.counts = [0] * n_arms

        # Noise variance estimate (per arm)
        self.sigma2 = [1.0] * n_arms
        self.sigma2_ema_beta = 0.95

        # Innovation tracking for adaptive inflation
        self.innovation_ema = [0.0] * n_arms  # EMA of standardized innovations
        self.innovation_ema_beta = 0.8

        # Base forgetting/inflation parameters
        self.base_gamma = 0.95  # base posterior shrinkage factor
        self.gamma_min = 0.85
        self.gamma_max = 0.99

        # Total rounds
        self.total_rounds = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adaptive posterior shrinkage based on innovation magnitude."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        innov = self.innovation_ema[arm]
        # Large innovations suggest non-stationarity -> want faster forgetting (smaller gamma)
        # Map innovation to gamma using tanh saturation
        t = np.tanh(0.5 * innov)
        # t in [-1, 1], map to [gamma_min, gamma_max]
        # High innovation -> t positive -> want smaller gamma
        gamma = self.base_gamma - t * (self.base_gamma - self.gamma_min)
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy (expected value)
        if not self._check_deadline():
            scores = np.array([self.theta[a] @ context for a in range(self.n)])
            return int(np.argmax(scores))

        # Thompson Sampling: sample from posterior for each arm
        scores = []
        for a in range(self.n):
            # Sample theta from N(theta_a, Sigma_a)
            # Use Cholesky decomposition for efficient sampling
            try:
                L = np.linalg.cholesky(self.Sigma[a] + 1e-10 * np.eye(self.d))
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                L = np.sqrt(np.diag(np.diag(self.Sigma[a] + 1e-10)))
            
            z = np.random.randn(self.d)
            theta_sample = self.theta[a] + L @ z
            score = theta_sample @ context
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1
        self.total_rounds += 1

        # Compute prediction and innovation
        pred = self.theta[arm] @ context
        innovation = reward - pred

        # Update noise variance estimate
        self.sigma2[arm] = self.sigma2_ema_beta * self.sigma2[arm] + (1 - self.sigma2_ema_beta) * (innovation ** 2)
        self.sigma2[arm] = max(self.sigma2[arm], 1e-6)

        # Standardized innovation for adaptive gamma
        std_innov = innovation / np.sqrt(self.sigma2[arm])
        self.innovation_ema[arm] = self.innovation_ema_beta * self.innovation_ema[arm] + (1 - self.innovation_ema_beta) * (std_innov ** 2)

        # Adaptive posterior shrinkage (forgetting)
        gamma = self._adaptive_gamma(arm)

        # Update posterior mean and covariance
        # Prior: theta ~ N(theta_old, Sigma_old)
        # Likelihood: reward | theta, context ~ N(context^T theta, sigma2)
        # Posterior update (Bayesian linear regression)
        
        Sigma_old = self.Sigma[arm]
        theta_old = self.theta[arm]
        
        # Shrink prior to account for non-stationarity
        Sigma_prior = Sigma_old / gamma  # inflate covariance (more uncertainty)
        theta_prior = theta_old  # mean stays the same
        
        # Bayesian update
        # Posterior covariance: (Sigma_prior^{-1} + context context^T / sigma2)^{-1}
        # Using matrix inversion lemma:
        # Sigma_post = Sigma_prior - Sigma_prior @ context @ (sigma2 + context^T Sigma_prior @ context)^{-1} @ context^T @ Sigma_prior
        
        Sp = Sigma_prior
        x = context
        s2 = self.sigma2[arm]
        
        # Compute alpha = sigma2 + x^T Sp x
        Spx = Sp @ x
        alpha = s2 + x @ Spx
        
        if alpha > 1e-12:
            # Posterior covariance
            Sigma_post = Sp - np.outer(Spx, Spx) / alpha
            # Posterior mean
            theta_post = theta_prior + (Sp @ x) * (reward - theta_prior @ x) / alpha
        else:
            Sigma_post = Sp
            theta_post = theta_prior

        # Ensure symmetry and positive semi-definiteness
        Sigma_post = (Sigma_post + Sigma_post.T) / 2.0
        
        self.theta[arm] = theta_post
        self.Sigma[arm] = Sigma_post
# EVOLVE-BLOCK-END
