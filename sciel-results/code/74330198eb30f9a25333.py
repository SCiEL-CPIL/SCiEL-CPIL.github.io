"""Thompson Sampling with sliding-window posterior for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling.
    
    Strategy:
    1. Maintain a recency-weighted posterior (sliding window) for each arm.
       This naturally adapts to non-stationarity without explicit detection.
    2. Use Thompson Sampling for action selection to balance exploration/exploitation.
    3. Use an adaptive noise floor (lambda) to prevent numerical instability and
       allow exploration in early stages or after regime shifts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Window size for the sliding window posterior
        # Smaller window -> faster adaptation to changes, higher variance
        # Larger window -> lower variance, slower adaptation
        self.window_size = 50
        
        # Initial regularization (prior precision)
        # A larger lambda means stronger prior (shrinkage towards 0)
        self.lambda_prior = 1.0
        
        # Adaptive factor for lambda
        self.lambda_min = 0.1
        self.lambda_max = 10.0
        self.lambda_decay = 0.99
        
        # Initialize per-arm state
        # We store the accumulated weighted sum of outer products and reward*context
        # A_t = sum_{i in window} w_i * x_i x_i^T + lambda * I
        # b_t = sum_{i in window} w_i * r_i * x_i
        self.A = [np.eye(dim) * self.lambda_prior for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Store recent contexts and rewards for the sliding window
        # We use a deque-like structure or just a list and pop from front
        # Since window size is small, list is fine. 
        # To be efficient, we can just recompute if window is small, 
        # or maintain running sums with decay. 
        # Running sums with decay (geometric) is more efficient and smooth.
        # Let's use geometric decay (forgetting factor) which is equivalent to 
        # a weighted sliding window.
        self.forget_factor = 0.95
        
        # Reset A and b to just the prior for geometric forgetting
        self.A = [np.eye(dim) * self.lambda_prior for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Track time for safety
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit
        
        # Counter for steps
        self.t = 0

    def select(self, context):
        # Check time budget
        if time.time() - self.start_time > self.deadline:
            # Fallback to greedy based on current mean
            # This is just a safety net, shouldn't trigger often
            scores = []
            for a in range(self.n):
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        
        # Thompson Sampling: sample theta from posterior N(theta_hat, Sigma)
        # theta_hat = A^{-1} b
        # Sigma = A^{-1}
        
        actions = []
        for a in range(self.n):
            # Use Cholesky for stability if possible, else inv
            try:
                # A is symmetric positive definite
                L = np.linalg.cholesky(self.A[a])
                # Solve L L^T theta_hat = b
                # Or just compute inv for simplicity given small dim
                inv_A = np.linalg.inv(self.A[a])
                theta_hat = inv_A @ self.b[a]
                
                # Sample theta ~ N(theta_hat, inv_A)
                # We can sample by theta = theta_hat + L_inv * z
                # where z ~ N(0, I)
                # L_inv is the inverse of Cholesky factor
                # Or simpler: sample using multivariate_normal if dim is small
                # np.random.multivariate_normal(theta_hat, inv_A)
                
                # Efficient sampling:
                # z = np.random.randn(self.d)
                # theta = theta_hat + (np.linalg.inv(L) @ z)
                # But inv(L) is expensive. 
                # Alternative: solve L y = z -> y, then theta = theta_hat + y? No.
                # If A = L L^T, then A^{-1} = L^{-T} L^{-1}.
                # Let v ~ N(0, I). Then A^{-1/2} v is not standard.
                # Standard way: 
                # 1. Compute Cholesky L of A.
                # 2. Solve L z = epsilon (where epsilon ~ N(0, I)) for z.
                # 3. theta = theta_hat + z.
                # Wait, Cov(theta) = A^{-1}. 
                # If A = L L^T, then A^{-1} = L^{-T} L^{-1}.
                # Let u ~ N(0, I). Then L^{-T} L^{-1} u is the covariance.
                # Let w = L^{-1} u. Then w ~ N(0, L^{-T} L^{-1})? No.
                # If u ~ N(0, I), then L^{-1} u ~ N(0, L^{-1} L^{-T}) = N(0, (L L^T)^{-1}) = N(0, A^{-1}).
                # So yes, solve L w = u for w. Then theta = theta_hat + w.
                
                epsilon = np.random.randn(self.d)
                w = np.linalg.solve(L, epsilon)
                theta_sample = theta_hat + w
                
            except np.linalg.LinAlgError:
                # Fallback to inverse
                try:
                    inv_A = np.linalg.inv(self.A[a])
                    theta_hat = inv_A @ self.b[a]
                    epsilon = np.random.randn(self.d)
                    # Approximate sampling with diagonal
                    std = np.sqrt(np.diag(inv_A))
                    theta_sample = theta_hat + epsilon * std
                except:
                    # Last resort: random
                    theta_sample = np.random.randn(self.d)
            
            actions.append(theta_sample @ context)
        
        return int(np.argmax(actions))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Update with geometric forgetting
        # A_t = gamma * A_{t-1} + x x^T
        # b_t = gamma * b_{t-1} + r x
        
        # However, pure geometric forgetting can lead to A becoming singular 
        # or having very small eigenvalues if data is sparse in certain directions.
        # The prior lambda should also decay or be maintained.
        # A common approach: A_t = gamma * (A_{t-1} - lambda I) + lambda I + x x^T
        # This keeps the prior active.
        
        # Let's stick to the simpler form but ensure lambda is added back
        # effectively by not letting the prior decay to zero too fast, 
        # or by adding lambda * I every step? No, that accumulates.
        
        # Better: 
        # A_t = gamma * A_{t-1} + x x^T
        # If we initialize A_0 = lambda I, then A_t = gamma^t lambda I + sum...
        # This means the prior influence decays. This is good for non-stationarity.
        
        self.A[arm] = self.forget_factor * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.forget_factor * self.b[arm] + reward * context
        
        # Optional: Regularize if A becomes ill-conditioned
        # Check condition number occasionally? Too expensive.
        # The lambda_prior in init provides initial stability.
        # As t increases, the influence of init lambda decays.
        # If we want to maintain a floor, we can add a small epsilon I every step.
        # self.A[arm] += 1e-6 * np.eye(self.d)
        
        # Adaptive forgetting? 
        # If reward is surprising (high residual), decrease forget_factor (forget less? or more?)
        # High residual -> regime shift -> forget more (increase forget_factor towards 1? No, 
        # forget_factor < 1. Higher forget_factor = slower forgetting. 
        # Lower forget_factor = faster forgetting).
        # So if residual is high, we want to forget past data faster -> lower forget_factor.
        
        # Calculate residual
        try:
            inv_A = np.linalg.inv(self.A[arm])
            theta_hat = inv_A @ self.b[arm]
            pred = theta_hat @ context
            residual = reward - pred
            
            # Scale residual by uncertainty?
            # Uncertainty = sqrt(context^T A^{-1} context)
            unc = np.sqrt(np.dot(context, inv_A @ context))
            if unc > 1e-6:
                std_res = residual / unc
            else:
                std_res = residual
            
            # If |std_res| > 2, it's a big surprise.
            # Adapt forget_factor: 
            # Base 0.95. If surprise, drop to 0.9 (forget faster).
            # If consistent, move back to 0.95.
            
            if abs(std_res) > 2.0:
                # Regime shift likely. Forget faster.
                self.forget_factor = max(0.8, self.forget_factor * 0.9)
            else:
                # No shift. Recover to base rate.
                self.forget_factor = min(0.95, self.forget_factor * 1.05)
                
        except:
            pass

# EVOLVE-BLOCK-END
