import numpy as np

class Policy:
    """Discounted Linear Thompson Sampling for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Discount factor for non-stationarity
        self.lam = 1.0     # Regularization parameter
        
        # Initialize precision matrices (A) and vectors (b) for each arm
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select(self, context):
        x = np.array(context)
        scores = []
        
        for a in range(self.n):
            try:
                # Compute posterior mean: mu = A^-1 b
                mu = np.linalg.solve(self.A[a], self.b[a])
                
                # Sample from N(0, A^-1) using Cholesky decomposition
                # A = L L^T => A^-1 = L^-T L^-1
                L = np.linalg.cholesky(self.A[a])
                
                # Sample w ~ N(0, I)
                w = np.random.randn(self.d)
                
                # Solve L v = w
                v = np.linalg.solve(L, w)
                
                # Solve L^T s = v
                s = np.linalg.solve(L.T, v)
                
                # theta = mu + s
                theta = mu + s
                
                score = theta @ x
                scores.append(score)
                
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                scores.append(np.random.randn())

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.array(context)
        # Update with discounting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
        self.b[arm] = self.gamma * self.b[arm] + reward * x
