"""Non-stationary linear contextual bandit with Sliding-Window RLS and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Sliding-Window Recursive Least Squares (RLS).
    
    Instead of a random-walk Kalman filter that only updates the selected arm,
    this approach maintains a sliding window of recent observations for each arm.
    The weight vector theta_a is estimated by solving a regularized least squares
    problem on the most recent W observations for that arm.
    
    This allows:
    1. Implicit handling of non-stationarity via the window size (older data is discarded).
    2. Better tracking of drift because the estimate is continuously refined by recent data,
       not just updated when the arm is pulled.
    3. UCB exploration based on the posterior covariance from the windowed RLS.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Sliding window size: number of recent samples to keep per arm
        # Smaller window = faster adaptation to drift, larger window = lower variance in stable periods
        self.window_size = 50

        # Regularization parameter for RLS (ridge regression)
        # Prevents singularity when window is small or contexts are collinear
        self.lam = 0.1

        # Per-arm storage: list of (context, reward) tuples, most recent at end
        self.histories = [[] for _ in range(n_arms)]

        # Per-arm cached estimates: theta (mean) and P (covariance proxy)
        # We will recompute these lazily or update incrementally.
        # For simplicity and robustness, we recompute from the window on select/update
        # if the window has changed significantly, but incremental RLS is better.
        # Let's use incremental RLS with a forgetting factor that effectively implements a window.
        # Actually, a true sliding window RLS is complex to implement incrementally without
        # storing the inverse covariance of the window.
        
        # Alternative: Use a fixed forgetting factor RLS (Exponential Weighting).
        # This is mathematically equivalent to a sliding window with exponential decay.
        # It is much simpler to implement incrementally and handles non-stationarity well.
        
        # Let's switch to Exponentially Weighted RLS (EWRLS).
        # This is a standard approach for non-stationary linear bandits.
        
        # Forgetting factor gamma: 0 < gamma <= 1
        # gamma close to 1 = long memory (stationary)
        # gamma close to 0 = short memory (highly non-stationary)
        self.gamma = 0.95
        
        # Per-arm state:
        # C_a: Inverse covariance matrix (d x d)
        # theta_a: Weight vector (d x 1)
        self.C = [np.eye(dim) / self.lam for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # UCB exploration coefficient
        # We need to scale this appropriately for the covariance magnitude
        self.alpha = 1.5
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0

    def _check_deadline(self):
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def select(self, context):
        """Select arm using UCB with EWRLS posterior."""
        if self._check_deadline():
            # Fallback: greedy on mean
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)

        ucb_scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Posterior variance: x^T C x (since C is inverse covariance)
            # Note: In Bayesian linear regression, Var(theta) = (X^T X + lambda I)^{-1}
            # So Var(pred) = x^T (X^T X + lambda I)^{-1} x
            var = context @ self.C[a] @ context
            std = np.sqrt(max(var, 0.0))
            ucb = mean + self.alpha * std
            ucb_scores.append(ucb)

        return int(np.argmax(ucb_scores))

    def update(self, context, arm, reward):
        """Exponentially Weighted RLS update for the selected arm."""
        context = np.asarray(context, dtype=np.float64)
        x = context
        r = float(reward)

        # EWRLS Update Rules:
        # C_{t+1} = (1/gamma) * (C_t - (C_t x x^T C_t) / (gamma + x^T C_t x))
        # theta_{t+1} = theta_t + C_{t+1} x (r - x^T theta_t)
        
        # Step 1: Compute intermediate terms
        Cx = self.C[arm] @ x
        denom = self.gamma + x @ Cx
        
        # Step 2: Update Inverse Covariance
        # C_new = (1/gamma) * C_old - (1/gamma) * (C_old x x^T C_old) / denom
        # To avoid numerical issues, we can compute directly:
        # C_new = (1/gamma) * (C_old - (C_old x x^T C_old) / denom)
        
        C_old = self.C[arm]
        outer = np.outer(Cx, x) # C_old x x^T
        C_new = (C_old - outer / denom) / self.gamma
        
        # Ensure symmetry and PSD
        C_new = (C_new + C_new.T) / 2.0
        
        # Step 3: Update Theta
        innovation = r - (self.theta[arm] @ x)
        theta_new = self.theta[arm] + C_new @ x * innovation
        
        # Store updated states
        self.C[arm] = C_new
        self.theta[arm] = theta_new

# EVOLVE-BLOCK-END
