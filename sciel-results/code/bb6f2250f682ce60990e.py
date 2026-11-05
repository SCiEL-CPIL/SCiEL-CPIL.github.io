import numpy as np
from collections import deque

class Policy:
    """Sliding Window LinUCB for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 50
        self.lambda_reg = 1.0
        self.c_alpha = 1.0
        # Store history for each arm
        self.history = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        # Pre-allocate identity for regularization
        self.I = np.eye(self.d)

    def select(self, context):
        scores = []
        for a in range(self.n):
            if len(self.history[a]) == 0:
                # Pure exploration for unseen arms
                scores.append(self.c_alpha * np.sqrt(self.d))
                continue
            
            hist = self.history[a]
            # Construct X (K x d) and y (K)
            # Using a list comprehension for speed in small K
            X = np.array([item[0] for item in hist])
            y = np.array([item[1] for item in hist])
            
            # Regularized covariance
            A = X.T @ X + self.lambda_reg * self.I
            # Solve for theta: (X^T X + lambda I) theta = X^T y
            try:
                theta = np.linalg.solve(A, X.T @ y)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
            
            # Predicted mean
            pred = theta @ context
            
            # Variance term
            # We need A_inv @ context. Since we have A, we can solve A v = context
            try:
                v = np.linalg.solve(A, context)
            except np.linalg.LinAlgError:
                v = np.zeros(self.d)
            
            var = np.sqrt(np.dot(context, v))
            
            scores.append(pred + self.c_alpha * var)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.history[arm].append((context, reward))
