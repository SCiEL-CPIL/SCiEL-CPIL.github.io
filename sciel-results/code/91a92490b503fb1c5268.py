import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive Sliding Window Linear Thompson Sampling for Non-Stationary Linear Bandits.
    
    Adapts the window size based on prediction error to balance bias and variance
    in the presence of changing arm weights.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Adaptive Window Parameters
        self.W = 50  # Initial window size
        self.min_W = 10
        self.max_W = 200
        self.adjust_interval = 20 # Adjust window every 20 steps
        
        # Noise variance estimate for Thompson Sampling
        # In non-stationary settings, we assume unit variance for the sampling
        # but the window size controls the effective uncertainty.
        self.sigma2 = 1.0
        
        # Regularization
        self.lambda_reg = 1.0
        
        # State for each arm
        # Queue of (context, reward)
        self.queues = [deque() for _ in range(n_arms)]
        # Sufficient statistics
        self.A = [np.eye(dim) * self.lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0

    def _recompute_stats(self, arm):
        """Recompute A and b from the current queue for a given arm."""
        A = np.eye(self.d) * self.lambda_reg
        b = np.zeros(self.d)
        for x, r in self.queues[arm]:
            A += np.outer(x, x)
            b += r * x
        self.A[arm] = A
        self.b[arm] = b

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Fallback: greedy
            means = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    means.append(theta @ context)
                except:
                    means.append(0.0)
            return int(np.argmax(means))
        
        scores = []
        for a in range(self.n):
            try:
                # Ensure A is invertible
                A = self.A[a] + 1e-8 * np.eye(self.d)
                b = self.b[a]
                
                # Mean
                theta_mean = np.linalg.solve(A, b)
                
                # Covariance: sigma^2 * A^{-1}
                # We sample from N(theta_mean, sigma^2 * A^{-1})
                # Cholesky of A
                L = np.linalg.cholesky(A)
                z = np.random.randn(self.d)
                w = np.linalg.solve(L.T, z)
                # Sample: theta_mean + sigma * w
                # Note: if A = L L^T, then A^{-1} = L^{-T} L^{-1}.
                # Var = sigma^2 A^{-1}. Std dev = sigma * A^{-1/2}.
                # x = theta_mean + sigma * L^{-T} z
                theta_sample = theta_mean + self.sigma2 * w
                
                score = theta_sample @ context
                scores.append(score)
            except np.linalg.LinAlgError:
                try:
                    theta = np.linalg.solve(self.A[a] + 1e-6 * np.eye(self.d), self.b[a])
                    score = theta @ context
                except:
                    score = 0.0
                scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Add new observation
        self.queues[arm].append((context, reward))
        
        # Incremental update of A and b
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Check if we need to adjust window size
        if self.t % self.adjust_interval == 0 and self.t > 0:
            # Estimate MSE for this arm
            if len(self.queues[arm]) > 5:
                try:
                    A = self.A[arm] + 1e-8 * np.eye(self.d)
                    theta = np.linalg.solve(A, self.b[arm])
                    mse = 0.0
                    count = 0
                    for x, r in self.queues[arm]:
                        pred = theta @ x
                        mse += (r - pred) ** 2
                        count += 1
                    if count > 0:
                        mse /= count
                        # Adjust W based on MSE
                        # High MSE -> weights are changing -> smaller window
                        # Low MSE -> weights are stable -> larger window
                        if mse > 2.0:
                            self.W = max(self.min_W, int(self.W * 0.9))
                        elif mse < 0.5:
                            self.W = min(self.max_W, int(self.W * 1.1))
                except:
                    pass
            
            # If W changed, we need to trim queues and recompute stats
            # We check if any queue is larger than W
            # To save time, we only recompute if W actually changed in this step
            # Or we can just trim and recompute periodically?
            # Let's trim and recompute for all arms if W changed.
            # How do we know if W changed? We can store old_W.
            
        # Trim queues to current W
        # And recompute stats if the queue was trimmed (i.e., it was full and we removed one)
        # OR if we just changed W.
        # To be safe and simple:
        # If len(queue) > W, remove oldest and update A, b incrementally.
        # If W changed, we must recompute from scratch to be consistent with the new window.
        
        # Let's simplify: 
        # 1. If we just adjusted W, recompute stats for all arms from their queues (trimmed to W).
        # 2. Otherwise, just incrementally update and trim.
        
        # We need a flag to know if W changed in this update cycle.
        # Since the logic above is inside `if self.t % ...`, we can set a flag.
        
        # Revised Update Logic:
        
        # 1. Incremental update already done above.
        # 2. Handle Window Size Adjustment
        old_W = self.W
        if self.t % self.adjust_interval == 0 and self.t > 0:
             # Calculate MSE and update self.W as above
             # (Code from above block)
             pass # The logic above was illustrative. Let's implement it properly here.

        # Let's rewrite update cleanly.

    def update(self, context, arm, reward):
        self.queues[arm].append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        w_changed = False
        if self.t % self.adjust_interval == 0 and self.t > 0:
            old_W = self.W
            # Estimate MSE for the updated arm to decide global W change
            # Using the most recently updated arm is a heuristic, but sufficient
            if len(self.queues[arm]) > 5:
                try:
                    A = self.A[arm] + 1e-8 * np.eye(self.d)
                    theta = np.linalg.solve(A, self.b[arm])
                    mse = 0.0
                    count = 0
                    for x, r in self.queues[arm]:
                        pred = theta @ x
                        mse += (r - pred) ** 2
                        count += 1
                    if count > 0:
                        mse /= count
                        if mse > 2.0:
                            self.W = max(self.min_W, int(self.W * 0.9))
                        elif mse < 0.5:
                            self.W = min(self.max_W, int(self.W * 1.1))
                except:
                    pass
            
            if self.W != old_W:
                w_changed = True

        # Trim queue and update stats if W changed or queue is too long
        if w_changed:
            # Recompute stats for all arms from scratch based on new W
            for a in range(self.n):
                # Trim queue
                while len(self.queues[a]) > self.W:
                    self.queues[a].popleft()
                self._recompute_stats(a)
        else:
            # Normal trim
            if len(self.queues[arm]) > self.W:
                x_old, r_old = self.queues[arm].popleft()
                self.A[arm] -= np.outer(x_old, x_old)
                self.b[arm] -= r_old * x_old
        
        self.t += 1
