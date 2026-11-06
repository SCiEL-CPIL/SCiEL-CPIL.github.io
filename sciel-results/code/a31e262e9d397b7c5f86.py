"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB for non-stationary linear bandits.
    
    Combines:
    1. Per-arm adaptive forgetting factor (lambda) based on recent MSE.
    2. Adaptive exploration parameter (alpha) based on global MSE.
    3. Efficient matrix inverse updates via Sherman-Morrison-Woodbury.
    4. Cached theta vectors updated incrementally for fast select().
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize per-arm linear models
        # inv_A is the inverse covariance matrix, b is the moment vector
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Cached theta = inv_A @ b, updated incrementally
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        # Lambda controls how much old data is retained.
        # High lambda (close to 1) = long memory. Low lambda = short memory.
        self.lam_min = 0.90
        self.lam_max = 0.998
        
        # Buffer for recent squared errors per arm to determine local stability
        self.error_buffers = [deque(maxlen=15) for _ in range(n_arms)]
        self.error_window_size = 15
        
        # UCB exploration parameter
        # Alpha controls the width of the confidence interval.
        self.alpha_min = 0.4
        self.alpha_max = 2.0
        self.alpha = 1.0
        
        # Global error tracking for adaptive alpha
        self.global_errors = deque(maxlen=30)
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # 10s safety margin before 150s kill
        
        # Step counter
        self.t = 0

    def _elapsed(self):
        return time.time() - self._start_time

    def _get_lambda(self, arm):
        """
        Calculate the adaptive forgetting factor for a specific arm
        based on recent squared prediction errors (MSE).
        Smooth mapping: high MSE -> low lambda (fast forgetting).
        """
        buf = self.error_buffers[arm]
        if len(buf) == 0:
            return self.lam_max
            
        # Convert deque to array for efficient MSE calculation
        errs = np.array(list(buf))
        mse = np.mean(errs ** 2)
        
        # Smooth mapping to [lam_min, lam_max]
        # If mse is 0, norm=0, lam=lam_max (remember everything)
        # If mse is large, norm->1, lam->lam_min (forget fast)
        mse_scale = 0.1
        norm = mse / (mse + mse_scale)
        lam = self.lam_max - norm * (self.lam_max - self.lam_min)
        
        return max(self.lam_min, min(self.lam_max, lam))

    def _update_alpha(self):
        """
        Adapt exploration parameter alpha based on global error level.
        High global error -> more exploration (higher alpha).
        """
        if len(self.global_errors) < 5:
            return
            
        errs = np.array(list(self.global_errors))
        mse = np.mean(errs ** 2)
        
        # Target alpha: more exploration when errors are high
        # Scale factor 0.2 means if mse > 0.2, we push towards alpha_max
        target = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.2))
        
        # Smooth update to avoid jitter
        self.alpha = 0.8 * self.alpha + 0.2 * target
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _update_theta(self, arm, context, reward, lam):
        """
        Update inv_A using Sherman-Morrison-Woodbury identity and update theta cache.
        A_new = lam * A_old + x x^T
        inv(A_new) = inv(lam * A_old + x x^T)
        
        Let M = lam * A_old. Then inv(M) = (1/lam) * inv_A_old.
        We need inv(M + x x^T).
        Using Sherman-Morrison:
        inv(M + x x^T) = inv(M) - inv(M) x x^T inv(M) / (1 + x^T inv(M) x)
        
        Note: We maintain inv_A such that inv_A @ A = I.
        So if A_new = lam * A_old + xx^T,
        inv_A_new = inv(lam * A_old + xx^T)
        
        Let B = lam * A_old. inv_B = (1/lam) * inv_A_old.
        inv_A_new = inv_B - (inv_B @ x @ x.T @ inv_B) / (1 + x.T @ inv_B @ x)
        """
        x = context
        inv_B = (1.0 / lam) * self.inv_A[arm]
        
        # Compute scalar denominator
        # x.T @ inv_B @ x
        tmp = inv_B @ x
        denom = 1.0 + x @ tmp
        
        # Numerator term: inv_B @ x @ x.T @ inv_B
        # This is (inv_B @ x) (inv_B @ x).T
        numerator = np.outer(tmp, tmp)
        
        self.inv_A[arm] = inv_B - numerator / denom
        
        # Update b
        b_old = self.b[arm]
        b_new = lam * b_old + reward * x
        
        # Update theta cache
        # theta_new = inv_A_new @ b_new
        # We can compute this directly for small dims, or use incremental update
        # For simplicity and correctness, compute directly since dim is small (typically <= 10)
        self.theta[arm] = self.inv_A[arm] @ b_new
        self.b[arm] = b_new

    def select(self, context):
        # Deadline check
        if self._elapsed() > self._deadline:
            # Fallback: greedy based on current estimates to avoid hanging
            scores = []
            for a in range(self.n):
                scores.append(float(self.theta[a] @ context))
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            # Mean prediction using cached theta
            mu = float(self.theta[a] @ context)
            
            # Variance term
            # var = context @ inv_A @ context
            var = float(context @ self.inv_A[a] @ context)
            std = np.sqrt(max(var, 0.0))
            
            # UCB score
            score = mu + self.alpha * std
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Deadline check
        if self._elapsed() > self._deadline:
            return
        
        self.t += 1
        
        # 1. Compute residual using current model BEFORE update
        pred = float(self.theta[arm] @ context)
        residual = reward - pred
        residual_sq = residual ** 2
        
        # 2. Update error buffers
        self.error_buffers[arm].append(residual_sq)
        self.global_errors.append(residual_sq)
        
        # 3. Determine adaptive parameters
        lam = self._get_lambda(arm)
        self._update_alpha()
        
        # 4. Update model matrices and theta cache
        self._update_theta(arm, context, reward, lam)

# EVOLVE-BLOCK-END
