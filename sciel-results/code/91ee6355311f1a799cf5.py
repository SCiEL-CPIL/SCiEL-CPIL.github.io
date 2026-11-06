"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with dynamic alpha and refined non-stationary handling."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Initialize covariance matrices and bias vectors for each arm
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        self._start_time = time.time()
        # Safety margin: 120s limit, use 115s to be safe
        self._deadline = self._start_time + 115.0
        
        # Parameters for adaptive exploration and forgetting
        self._base_alpha = 0.8
        self._min_alpha = 0.4
        self._max_alpha = 1.2
        
        self._base_lambda = 0.95
        self._min_lambda = 0.85
        self._max_lambda = 0.99
        
        self._error_window = 10
        self._recent_errors = []
        
        # State for tracking
        self._last_arm = None
        self._last_context = None
        self._round = 0
        
        # Cache for theta vectors to avoid recomputing inverse every time
        self._theta_cache = [np.zeros(dim) for _ in range(n_arms)]
        self._A_inv_cache = [np.eye(dim) for _ in range(n_arms)]

    def _update_cache(self, arm):
        """Update cached inverse and theta for a given arm."""
        try:
            self._A_inv_cache[arm] = np.linalg.inv(self.A[arm])
            self._theta_cache[arm] = self._A_inv_cache[arm] @ self.b[arm]
        except np.linalg.LinAlgError:
            self._A_inv_cache[arm] = np.linalg.pinv(self.A[arm])
            self._theta_cache[arm] = self._A_inv_cache[arm] @ self.b[arm]

    def _get_alpha(self):
        """
        Dynamically adjust UCB alpha based on recent prediction errors using smooth mapping.
        """
        if len(self._recent_errors) < 3:
            return self._base_alpha
        
        recent = self._recent_errors[-self._error_window:]
        mean_err = np.mean(recent)
        std_err = np.std(recent) if len(recent) > 1 else 0.0
        
        # Use a smooth logistic-like mapping
        error_metric = mean_err + 0.3 * std_err
        
        # Smooth mapping: alpha = min + (max - min) / (1 + exp(-k*(error_metric - mid)))
        k = 4.0
        mid = 0.25
        alpha_val = self._min_alpha + (self._max_alpha - self._min_alpha) / (1.0 + np.exp(-k * (error_metric - mid)))
            
        return np.clip(alpha_val, self._min_alpha, self._max_alpha)

    def _get_lambda(self):
        """
        Dynamically adjust forgetting factor using smooth error-based adaptation.
        """
        if len(self._recent_errors) < 5:
            return self._base_lambda
        
        recent = self._recent_errors[-self._error_window:]
        mean_err = np.mean(recent)
        
        # Smooth mapping: higher error -> lower lambda (faster forgetting)
        # lambda = max - (max - min) / (1 + exp(-k*(err - mid)))
        k = 3.0
        mid = 0.3
        lambda_val = self._max_lambda - (self._max_lambda - self._min_lambda) / (1.0 + np.exp(-k * (mean_err - mid)))
            
        return np.clip(lambda_val, self._min_lambda, self._max_lambda)

    def select(self, context):
        self._round += 1
        
        # Deadline guard: if close to limit, switch to pure exploitation (greedy)
        if time.time() > self._deadline - 2.0:
            scores = []
            for a in range(self.n):
                scores.append(self._theta_cache[a] @ context)
            return int(np.argmax(scores))

        # Get adaptive alpha
        alpha = self._get_alpha()
        
        scores = []
        for a in range(self.n):
            # Use cached theta and A_inv
            theta = self._theta_cache[a]
            mu = theta @ context
            
            # Uncertainty: context^T A^{-1} context
            # We can compute this using the cached A_inv
            A_inv = self._A_inv_cache[a]
            v = context @ A_inv @ context
            sigma = np.sqrt(max(v, 0.0))
            
            scores.append(mu + alpha * sigma)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Track prediction error for adaptive parameters
        try:
            pred = self._theta_cache[arm] @ context
            err = abs(reward - pred)
            self._recent_errors.append(err)
            if len(self._recent_errors) > self._error_window * 2:
                self._recent_errors = self._recent_errors[-self._error_window:]
        except Exception:
            pass

        # Update model with adaptive forgetting
        lam = self._get_lambda()
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # Update cache for the updated arm
        self._update_cache(arm)
# EVOLVE-BLOCK-END
