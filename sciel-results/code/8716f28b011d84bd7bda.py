"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Thompson Sampling with adaptive Mahalanobis-based forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive decay parameters
        self.base_decay = 0.99
        self.min_decay = 0.90
        self.max_decay = 0.999
        self.mahalanobis_threshold = 3.0
        self.adaptation_rate = 0.1
        self.current_decay = self.base_decay
        # Track prediction residuals for adaptive decay
        self.residual_history = []
        self.residual_window = 20
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 110s safety margin before 120s limit
        self._chols = [np.linalg.cholesky(self.A[a]) for a in range(n_arms)]

    def _compute_adaptive_decay(self, arm, context, reward):
        """Compute adaptive decay factor based on Mahalanobis distance of residual."""
        # Get current estimate
        try:
            L = self._chols[arm]
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            # Compute prediction and residual
            pred = theta @ context
            residual = reward - pred
            # Compute Mahalanobis distance
            # Mahalanobis distance: sqrt(residual^2 / (context^T A^{-1} context + 1))
            inv_context_var = context @ np.linalg.solve(self.A[arm], context)
            mahalanobis_dist = abs(residual) / np.sqrt(inv_context_var + 1e-8)
            
            # Update residual history
            self.residual_history.append(mahalanobis_dist)
            if len(self.residual_history) > self.residual_window:
                self.residual_history.pop(0)
            
            # Compute adaptive decay
            # Large Mahalanobis distance indicates regime shift -> lower decay (faster forgetting)
            if mahalanobis_dist > self.mahalanobis_threshold:
                # Sharp reduction in decay to accelerate adaptation
                target_decay = self.min_decay
            else:
                # Normal operation, use base decay with slight adaptation
                mean_resid = np.mean(self.residual_history) if self.residual_history else 0
                if mean_resid > 2.0:
                    target_decay = self.base_decay * 0.95
                else:
                    target_decay = self.base_decay
            
            # Smoothly adapt decay
            self.current_decay += self.adaptation_rate * (target_decay - self.current_decay)
            self.current_decay = np.clip(self.current_decay, self.min_decay, self.max_decay)
            
        except Exception:
            self.current_decay = self.base_decay

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return best arm based on current estimates
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except:
                    pass
            return int(best_arm)
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            try:
                # Thompson Sampling: sample theta from posterior
                # theta ~ N(A^{-1}b, A^{-1})
                L = np.linalg.cholesky(self.A[a] + 1e-8 * np.eye(self.d))
                theta_hat = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                # Sample from N(theta_hat, A^{-1})
                z = np.random.randn(self.d)
                theta_sample = theta_hat + np.linalg.solve(L.T, z)
                scores.append(theta_sample @ context)
            except Exception:
                # Fallback: use point estimate
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return
        
        context = np.asarray(context, dtype=np.float64)
        
        # Compute adaptive decay before updating
        self._compute_adaptive_decay(arm, context, reward)
        
        decay = self.current_decay
        
        # Update A and b with forgetting
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context
        
        # Update Cholesky factorization
        try:
            self._chols[arm] = np.linalg.cholesky(self.A[arm] + 1e-8 * np.eye(self.d))
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more regularization
            self.A[arm] = self.A[arm] + 1e-6 * np.eye(self.d)
            try:
                self._chols[arm] = np.linalg.cholesky(self.A[arm])
            except:
                self._chols[arm] = np.eye(self.d)
# EVOLVE-BLOCK-END
