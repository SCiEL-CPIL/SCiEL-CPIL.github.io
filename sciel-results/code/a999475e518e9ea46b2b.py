"""Bayesian Linear Bandit with Smooth Adaptive Forgetting via EMA Residuals."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Thompson Sampling for non-stationary linear bandits.
    
    Uses a Bayesian framework where the posterior over weights is Gaussian.
    The covariance matrix is updated using the matrix inversion lemma for efficiency.
    A smooth adaptive forgetting mechanism driven by an EMA of Mahalanobis
    residuals replaces threshold-based logic for better tracking of non-stationarity.
    Cholesky factors are maintained to avoid redundant factorizations in select().
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize posterior parameters for each arm
        # Prior: N(0, I)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.V = [np.eye(dim) for _ in range(n_arms)]
        # Maintain Cholesky factors L where V = L @ L.T
        self.L = [np.eye(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters (smooth EMA-based)
        self.base_decay = 0.98
        self.min_decay = 0.5
        self.max_decay = 1.0
        self.ema_alpha = 0.3  # EMA smoothing factor for residual tracking
        
        # Track EMA of squared Mahalanobis residuals per arm
        self.ema_residual_sq = [1.0] * n_arms  # Initialize to 1 (unit prior variance)
        
        # Time tracking for safety
        self.start_time = time.time()
        self.deadline = 140.0  # 10s margin before 150s kill

    def _compute_decay(self, arm, z_score_sq):
        """
        Compute smooth adaptive decay factor based on EMA of squared residuals.
        High residual variance -> lower decay (faster forgetting).
        Low residual variance -> higher decay (slower forgetting).
        """
        # Update EMA of squared z-score
        self.ema_residual_sq[arm] = (1 - self.ema_alpha) * self.ema_residual_sq[arm] + self.ema_alpha * z_score_sq
        
        # Map EMA to decay smoothly using a soft function
        # When EMA is small (stable), decay close to max_decay (slow forgetting)
        # When EMA is large (unstable), decay close to min_decay (fast forgetting)
        ema_val = self.ema_residual_sq[arm]
        # Soft mapping: decay = min_decay + (max_decay - min_decay) * exp(-ema_val / scale)
        # Scale controls sensitivity; use 1.0 as a reasonable default
        scale = 1.0
        decay = self.min_decay + (self.max_decay - self.min_decay) * np.exp(-ema_val / scale)
        
        return np.clip(decay, self.min_decay, self.max_decay)

    def _update_cholesky(self, V_new):
        """
        Update Cholesky factor for the new covariance matrix.
        Uses a stable approach: compute Cholesky of V_new directly.
        For small dimensions this is fast and reliable.
        """
        try:
            L_new = np.linalg.cholesky(V_new)
        except np.linalg.LinAlgError:
            # Add small jitter for numerical stability
            jitter = 1e-6 * np.eye(V_new.shape[0])
            L_new = np.linalg.cholesky(V_new + jitter)
        return L_new

    def select(self, context):
        # Safety check
        if time.time() - self.start_time > self.deadline:
            return int(np.random.randint(self.n))
        
        context = np.asarray(context, dtype=np.float64)
        
        scores = []
        for a in range(self.n):
            # Thompson Sampling: sample theta from posterior N(theta, V)
            # Use maintained Cholesky factor L where V = L @ L.T
            # theta_sample = theta + L @ z
            try:
                L = self.L[a]
                z = np.random.randn(self.d)
                theta_sample = self.theta[a] + L @ z
            except Exception:
                # Fallback if Cholesky factor is invalid
                try:
                    L = np.linalg.cholesky(self.V[a])
                    z = np.random.randn(self.d)
                    theta_sample = self.theta[a] + L @ z
                except np.linalg.LinAlgError:
                    theta_sample = self.theta[a]
            
            scores.append(theta_sample @ context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Safety check
        if time.time() - self.start_time > self.deadline:
            return
        
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute residual and z-score for adaptive forgetting
        pred = self.theta[arm] @ context
        residual = reward - pred
        
        # Compute uncertainty (std) of the prediction
        var = context @ (self.V[arm] @ context)
        std = np.sqrt(max(var, 1e-8))
        
        # Squared z-score
        z_score_sq = (residual / std) ** 2
        
        # 2. Compute smooth adaptive decay
        decay = self._compute_decay(arm, z_score_sq)
        
        # 3. Apply Forgetting
        # V_new = (1/decay) * V_old  (equivalent to V_inv *= decay)
        V = self.V[arm] / decay
        
        # 4. Update Posterior (Bayesian Linear Regression Update via Sherman-Morrison)
        # V_updated = V - V x (x^T V x + 1)^(-1) x^T V
        x = context
        Vx = V @ x
        denom = x @ Vx + 1.0
        V_new = V - np.outer(Vx, Vx) / denom
        
        # Update theta
        # theta_new = theta + V_new @ x * (r - theta^T x)
        self.theta[arm] = self.theta[arm] + (V_new @ x) * residual
        
        # Update stored matrices and Cholesky factor
        self.V[arm] = V_new
        self.L[arm] = self._update_cholesky(V_new)
# EVOLVE-BLOCK-END
