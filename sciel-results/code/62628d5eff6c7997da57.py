"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Discounted Thompson Sampling for non-stationary linear bandits.
    Uses per-arm linear models with a fixed discount factor to forget old data.
    Thompson Sampling provides natural exploration based on posterior uncertainty.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize per-arm linear models
        # A is the covariance matrix (inverse), b is the moment vector
        # Using ridge regularization alpha=1.0 for stability
        self.alpha = 1.0
        self.A = [np.eye(dim) * self.alpha for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Discount factor for non-stationarity
        # 0.95 is a standard choice; allows tracking changes while retaining some history
        self.lambda_ = 0.95
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 145.0  # 5s safety margin before 150s kill

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context)
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n):
            try:
                # Compute Cholesky decomposition for stable inversion
                L = np.linalg.cholesky(self.A[a])
                
                # Compute mean theta = inv(A) b
                # Solve L.T y = b, then L theta = y
                y = np.linalg.solve(L.T, self.b[a])
                theta = np.linalg.solve(L, y)
                
                # Sample from posterior N(theta, inv(A))
                # inv(A) = inv(L.T) inv(L)
                # If z ~ N(0, I), then delta = inv(L) z has covariance inv(L) inv(L.T) = inv(A)
                z = np.random.randn(self.d)
                delta = np.linalg.solve(L, z)
                theta_tilde = theta + delta
                
                # Compute reward estimate for this arm
                score = np.dot(theta_tilde, context)
                
                if score > best_score:
                    best_score = score
                    best_arm = a
            except np.linalg.LinAlgError:
                # Fallback to random if numerical issues occur
                score = np.random.randn()
                if score > best_score:
                    best_score = score
                    best_arm = a
                    
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # Deadline check
        if time.time() > self._deadline:
            return

        # Update model with discounting
        lam = self.lambda_
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

# EVOLVE-BLOCK-END
