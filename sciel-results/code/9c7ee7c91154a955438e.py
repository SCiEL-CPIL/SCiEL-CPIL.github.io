"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Sliding Window Linear Regression with Thompson Sampling.
    
    Instead of recursive exponential forgetting which can accumulate error or 
    react slowly to sudden changes, we maintain a strict window of the last W 
    observations per arm. The sufficient statistics (A, b) are updated by adding 
    the new observation and subtracting the oldest one when the window is full.
    This ensures the model is always a least-squares fit to the most recent data,
    providing robust adaptation to non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Window size: How many recent samples to keep for regression.
        # A smaller window adapts faster to changes but has higher variance.
        # For non-stationary bandits, 100-200 is a common robust choice.
        self.window_size = 150
        
        # Initialize sufficient statistics
        # We start with identity matrix regularization to ensure invertibility
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Queues to store (context, reward) for subtraction from statistics
        # We store the context vector and the reward to reconstruct outer product and weighted vector
        self.history = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Time management for safety
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def select(self, context):
        # Deadline check: if we are running out of time, fall back to a simple greedy
        # In this context, we assume select is called frequently. 
        # If time is critical, we could skip TS and use mean, but TS is fast enough 
        # for small dims.
        
        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Compute posterior mean and variance
            # theta_hat = A^{-1} b
            # Var(theta) = A^{-1}
            
            try:
                # Use solve for better numerical stability and speed
                # We need A^{-1} for the covariance and for theta.
                # Actually, we can solve A * theta = b
                theta_hat = np.linalg.solve(self.A[a], self.b[a])
                
                # For Thompson Sampling, we need to sample from N(theta_hat, A^{-1})
                # We can compute the inverse once or use Cholesky.
                # Since A is updated by adding/subtracting rank-1 updates, it remains SPD.
                # Cholesky is faster and more stable for sampling.
                L = np.linalg.cholesky(self.A[a])
                
                # Solve L L^T x = b? No, we need A^{-1}.
                # Let's just compute the inverse for the covariance since dim is likely small (5-10).
                # Or better: Sample z ~ N(0, I), then theta_sample = theta_hat + A^{-1/2} z?
                # Standard TS: theta_sample = theta_hat + L^{-1} z where L is Cholesky of Cov?
                # Cov = A^{-1}. Let A = L L^T. Then A^{-1} = L^{-T} L^{-1}.
                # So we need to solve L^T y = z, then L theta_diff = y.
                
                z = np.random.randn(self.d)
                # Solve L^T y = z
                y = np.linalg.solve(L.T, z)
                # Solve L theta_diff = y
                theta_diff = np.linalg.solve(L, y)
                
                theta_sample = theta_hat + theta_diff
                
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular or Cholesky fails
                # This can happen if window is too small or contexts are collinear
                # Add jitter
                try:
                    A_jittered = self.A[a] + 1e-6 * np.eye(self.d)
                    theta_hat = np.linalg.solve(A_jittered, self.b[a])
                    L = np.linalg.cholesky(A_jittered)
                    z = np.random.randn(self.d)
                    y = np.linalg.solve(L.T, z)
                    theta_diff = np.linalg.solve(L, y)
                    theta_sample = theta_hat + theta_diff
                except:
                    theta_sample = np.zeros(self.d)

            scores.append(theta_sample @ context)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # 1. Remove the oldest observation if the window is full
        if len(self.history[arm]) == self.window_size:
            old_context, old_reward = self.history[arm][0]
            # Subtract its contribution from A and b
            self.A[arm] -= np.outer(old_context, old_context)
            self.b[arm] -= old_reward * old_context
            
            # Safety check: Due to floating point errors, A might drift out of SPD.
            # We can periodically re-orthogonalize or add a small regularization if eigenvalues go negative.
            # For simplicity, we rely on the fact that we started with I and add/subtract symmetric PSD matrices.
            # However, numerical drift can cause issues. 
            # A simple fix: if select fails, we handled it with jitter. 
            # To be more robust, we can check the determinant or min eigenvalue occasionally, 
            # but that's expensive. The jitter in select is sufficient for most cases.
            
        # 2. Add the new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # 3. Push to history
        self.history[arm].append((context, reward))
        
        self.t += 1
# EVOLVE-BLOCK-END
