# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Weighted K-Nearest Neighbors (WKNN) Bandit.
    Non-parametric, instance-based approach.
    Uses exponential time decay and context distance weighting for non-stationarity.
    """

    def __init__(self, n_arms, dim, K=20, gamma=0.95, sigma=1.0, beta=2.0):
        self.n = n_arms
        self.d = dim
        self.K = K
        self.gamma = gamma
        self.sigma = sigma
        self.beta = beta
        
        # History for each arm: list of (context, reward, time)
        self.histories = [[] for _ in range(n_arms)]
        self.t = 0
        self.max_history = 500  # Limit memory per arm

    def _predict_arm(self, context, arm_idx):
        history = self.histories[arm_idx]
        if len(history) == 0:
            # If no data, return 0 mean and high uncertainty
            return 0.0, 1.0

        # Calculate distances to all points in history
        # context shape: (d,)
        # history contexts: (N, d)
        ctxs = np.array([h[0] for h in history])
        rewards = np.array([h[1] for h in history])
        times = np.array([h[2] for h in history])
        
        # Time decay: older points get lower weight
        # w_time = gamma ** (t_current - t_point)
        time_diffs = self.t - times
        w_time = self.gamma ** time_diffs
        
        # Context distance: Euclidean distance
        # (N, 1, d) - (1, d) -> (N, d) -> sum axis 1 -> (N,)
        dists = np.linalg.norm(ctxs - context, axis=1)
        
        # Spatial weight: Gaussian kernel
        w_space = np.exp(-0.5 * (dists / self.sigma) ** 2)
        
        # Combined weights
        weights = w_time * w_space
        
        # Normalize weights
        sum_w = np.sum(weights)
        if sum_w < 1e-10:
             return 0.0, 1.0
        weights = weights / sum_w
        
        # Weighted mean
        mean = np.sum(weights * rewards)
        
        # Weighted variance (for UCB bonus)
        # Var = sum(w * (r - mean)^2)
        var = np.sum(weights * (rewards - mean) ** 2)
        
        # If we have few effective neighbors, variance might be low but confidence low.
        # We can add a prior variance component if effective sample size is low.
        # Effective N = (sum w)^2 / sum(w^2)
        eff_n = (sum_w ** 2) / (np.sum(weights ** 2) + 1e-10)
        
        # Scale variance by 1/eff_n to reflect uncertainty in estimate
        # This is a heuristic for Bayesian uncertainty in KNN
        uncertainty = np.sqrt(var / max(eff_n, 1.0) + 1.0 / max(eff_n, 1.0))
        
        return mean, uncertainty

    def select(self, context):
        context = np.asarray(context, dtype=float).ravel()
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            mean, unc = self._predict_arm(context, a)
            # UCB score
            scores[a] = mean + self.beta * unc
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float).ravel()
        self.histories[arm].append((context, float(reward), self.t))
        
        # Trim history to avoid memory explosion and slow down
        if len(self.histories[arm]) > self.max_history:
            # Remove oldest
            self.histories[arm] = self.histories[arm][-self.max_history:]
            
        self.t += 1
# EVOLVE-BLOCK-END
