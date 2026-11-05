"""Non-stationary linear contextual bandit with BOCPD-inspired adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Per-arm discounted least squares with Sherman-Morrison inverse updates
    - Bayesian Online Changepoint Detection (BOCPD) inspired adaptive forgetting
    - UCB exploration with adaptive confidence
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Ridge parameter for numerical stability
        self.ridge = 0.1

        # Per-arm state: A_inv (d x d) = (sum gamma^t x x^T + lambda I)^{-1}, b (d,)
        self.A_inv = [self.ridge * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # BOCPD-inspired state tracking
        # For each arm, we track:
        # - run_length: how many steps since last "reset"
        # - posterior: probability of being in current regime
        # - prediction errors for change detection
        self.run_lengths = [0] * n_arms
        self.posterior = [1.0] * n_arms  # Start confident in current regime
        self.prediction_errors = [0.0] * n_arms  # EMA of prediction errors
        self.error_ema_alpha = 0.2  # Rate for error EMA

        # Adaptive forgetting parameters
        self.base_gamma = 0.95
        self.min_gamma = 0.70
        self.max_gamma = 0.99

        # Exploration parameter
        self.beta0 = 1.2

    def _get_gamma_bocpd(self, arm, current_error):
        """
        BOCPD-inspired adaptive forgetting factor.
        
        The idea: maintain a posterior over the "age" of the current regime.
        If prediction errors are consistently high, the posterior shifts towards
        shorter run lengths, implying a recent changepoint, which means we should
        forget faster (lower gamma).
        
        We use a simplified BOCPD where:
        - The hazard rate (probability of a changepoint) is adaptive
        - The posterior over run length determines the effective memory
        """
        # Update error EMA
        self.prediction_errors[arm] = (1 - self.error_ema_alpha) * self.prediction_errors[arm] + \
                                      self.error_ema_alpha * current_error

        # Estimate hazard rate: higher error -> higher probability of changepoint
        # Use a smooth mapping from error to hazard
        # Hazard should be in (0, 1), representing P(changepoint at this step)
        err = max(self.prediction_errors[arm], 1e-6)
        
        # Adaptive hazard: base hazard that increases with error
        base_hazard = 0.01
        hazard = base_hazard * (1 + 2 * np.log1p(err / 0.1))
        hazard = min(hazard, 0.5)  # Cap hazard at 50%

        # Update posterior over run length
        # Simplified: posterior_new = (1 - hazard) * posterior_old
        # If a changepoint is detected (low posterior), reset run length
        self.posterior[arm] *= (1 - hazard)
        
        # If posterior drops below threshold, we're likely in a new regime
        if self.posterior[arm] < 0.1:
            # "Reset" - force faster forgetting
            self.posterior[arm] = 1.0
            self.run_lengths[arm] = 0
            return self.min_gamma  # Fastest forgetting after detected change

        # Map posterior to gamma
        # High posterior (confident in current regime) -> high gamma (slow forgetting)
        # Low posterior (uncertain, possible change) -> lower gamma (faster forgetting)
        gamma = self.min_gamma + (self.max_gamma - self.min_gamma) * self.posterior[arm]
        
        # Also incorporate run length: longer runs allow slightly more confidence
        # but with diminishing returns
        run_bonus = min(self.run_lengths[arm] / 100.0, 1.0) * 0.05
        gamma = min(gamma + run_bonus, self.max_gamma)
        
        return max(self.min_gamma, min(self.max_gamma, gamma))

    def select(self, context):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Return a safe default arm
            return 0

        scores = np.zeros(self.n)
        for a in range(self.n):
            if self.counts[a] == 0:
                # No data yet: pure exploration
                scores[a] = self.beta0 * np.sqrt(context @ self.A_inv[a] @ context)
            else:
                theta = self.A_inv[a] @ self.b[a]
                mu = theta @ context
                var = context @ self.A_inv[a] @ context
                # Adaptive beta: slightly increases over time to maintain exploration
                beta = self.beta0 * np.sqrt(1.0 + self.t * 0.0005)
                scores[a] = mu + beta * np.sqrt(max(var, 1e-8))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.counts[arm] += 1

        # Compute prediction for BOCPD
        theta = self.A_inv[arm] @ self.b[arm]
        pred = theta @ context
        error = reward - pred
        current_error_sq = error * error

        # Get adaptive gamma using BOCPD-inspired mechanism
        gamma = self._get_gamma_bocpd(arm, current_error_sq)

        # Update run length
        self.run_lengths[arm] += 1

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Update A_inv using Sherman-Morrison-like approach with forgetting
        # A_new = gamma * A_old + x x^T
        # A_inv_new = (gamma * A_old + x x^T)^{-1}
        # Using matrix inversion lemma:
        # A_inv_new = (1/gamma)*A_inv_old - (1/gamma^2)*outer(Ax, Ax) / (1 + (1/gamma)*x^T A_inv_old x)
        
        A_inv_old = self.A_inv[arm]
        Ax = A_inv_old @ context
        denom = 1.0 + (1.0 / gamma) * (context @ Ax)

        A_inv_new = (1.0 / gamma) * A_inv_old - (1.0 / (gamma * gamma)) * np.outer(Ax, Ax) / denom

        # Symmetrize to reduce numerical drift
        A_inv_new = 0.5 * (A_inv_new + A_inv_new.T)

        self.A_inv[arm] = A_inv_new
# EVOLVE-BLOCK-END
