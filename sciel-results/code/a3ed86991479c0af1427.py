import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Adaptive forgetting for non-stationarity
        # Start with a relatively fast forgetting rate to track changes
        self.gamma = 0.90
        self.gamma_min = 0.70
        self.gamma_max = 0.95
        
        # Per-arm state for Exponentially Weighted Linear Regression
        # We maintain:
        # A[a]: Precision matrix (sum of discounted x x^T)
        # b[a]: Precision-weighted mean vector (sum of discounted x * r)
        
        # Initialize with small ridge for stability
        self.ridge = 1e-3
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Store last contexts and rewards for drift detection
        self.last_contexts = [None] * n_arms
        self.last_rewards = [None] * n_arms
        self.prev_errors = [0.0] * n_arms
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            context_vec = np.array(context)
            scores = []
            for a in range(self.n):
                theta = np.linalg.solve(self.A[a], self.b[a])
                scores.append(theta @ context_vec)
            return int(np.argmax(scores))

        context_vec = np.array(context)
        
        # Compute greedy scores using current estimates
        scores = []
        for a in range(self.n):
            # Solve A[a] theta = b[a]
            # Using solve is more stable than inv
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                # Fallback to identity if singular (should be rare with ridge)
                theta = self.b[a] / self.ridge
            scores.append(theta @ context_vec)
        
        # Add exploration bonus
        # We can use an UCB-style bonus based on the uncertainty of the estimate.
        # Uncertainty ~ sqrt(x^T (A^{-1}) x)
        # But computing A^{-1} is expensive. 
        # Alternatively, use a simple epsilon-greedy with adaptive epsilon.
        # Or, use Thompson Sampling approximation?
        
        # Given the "structural change" hint and previous success of lightweight methods,
        # let's try a simple exploration mechanism:
        # If the model is confident (high eigenvalues of A), exploit.
        # If low confidence, explore.
        
        # Let's use a fixed small epsilon for early exploration, then decay?
        # Or better: use the norm of the estimate vs context.
        
        # Actually, a common lightweight approach is just Greedy with a small random perturbation
        # or simply Greedy if the model is good.
        
        # Let's try a simple UCB:
        # Bonus = alpha * sqrt(x^T A^{-1} x)
        # To avoid inverting A, we can approximate A^{-1} x by solving A y = x.
        
        alpha = 1.0  # Exploration coefficient
        scores_with_bonus = []
        for a in range(self.n):
            try:
                # Solve A[a] y = context_vec
                y = np.linalg.solve(self.A[a], context_vec)
                uncertainty = np.sqrt(context_vec @ y)
            except np.linalg.LinAlgError:
                uncertainty = 10.0  # Large uncertainty if singular
            
            score = scores[a] + alpha * uncertainty
            scores_with_bonus.append(score)
        
        return int(np.argmax(scores_with_bonus))

    def update(self, context, arm, reward):
        self.t += 1
        context_vec = np.array(context)
        
        # Update model for the selected arm
        # A_new = gamma * A_old + outer(x, x)
        # b_new = gamma * b_old + reward * x
        
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context_vec, context_vec)
        self.b[arm] = self.gamma * self.b[arm] + reward * context_vec
        
        # Drift detection to adapt gamma
        # If the reward error is high, increase forgetting (lower gamma)
        # If the reward error is low, decrease forgetting (higher gamma)
        
        # Compute predicted reward for this arm
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            predicted_reward = theta @ context_vec
        except np.linalg.LinAlgError:
            predicted_reward = 0.0
            
        error = abs(reward - predicted_reward)
        
        # Exponential Moving Average of error
        self.prev_errors[arm] = 0.9 * self.prev_errors[arm] + 0.1 * error
        
        # Adapt gamma:
        # If error is high, we want to forget old data faster -> lower gamma
        # If error is low, we want to keep old data -> higher gamma
        
        # Map error to gamma adjustment
        # Assume typical error range is [0, 5]
        # If error > 2, reduce gamma
        # If error < 1, increase gamma
        
        if error > 2.0:
            self.gamma = max(self.gamma_min, self.gamma - 0.02)
        elif error < 1.0:
            self.gamma = min(self.gamma_max, self.gamma + 0.01)
        
        # Ensure gamma stays within bounds
        self.gamma = np.clip(self.gamma, self.gamma_min, self.gamma_max)
        
        # Check for numerical stability
        if not np.all(np.isfinite(self.A[arm])) or not np.all(np.isfinite(self.b[arm])):
            self.A[arm] = np.eye(self.d) * self.ridge
            self.b[arm] = np.zeros(self.d)
