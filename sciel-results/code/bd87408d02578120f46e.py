"""Bootstrap spawn: direct approach for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm sliding window regression (hard truncation of old data)
    - UCB-style exploration based on window covariance
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Sliding window parameters
        self.window_size = 50  # Keep last 50 samples per arm
        self.lam = 1.0         # Regularization for least squares

        # Per-arm data storage
        self.contexts = [np.zeros((self.window_size, dim)) for _ in range(n_arms)]
        self.rewards = [np.zeros(self.window_size) for _ in range(n_arms)]
        self.counts = [0] * n_arms  # Total samples seen for each arm
        self.current_index = [0] * n_arms  # Circular buffer index

        # Exploration
        self.C = 1.5

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_model_params(self, arm):
        """
        Compute theta and V for a specific arm using the current sliding window.
        Uses ridge regression: theta = (X^T X + lam*I)^-1 X^T y
        V = (X^T X + lam*I)^-1
        """
        c = self.counts[arm]
        if c == 0:
            # Prior: theta=0, V=I
            return np.zeros(self.d), np.eye(self.d)

        # Determine how many valid samples are in the window
        valid_samples = min(c, self.window_size)
        
        # Extract valid samples from circular buffer
        # The most recent sample is at current_index[arm] - 1 (mod window_size)
        # We need the last 'valid_samples' entries.
        # Instead of copying, we can construct X and y directly.
        
        # Optimization: If we have fewer than window_size samples, we just take all.
        # If full, we take all (since we overwrite oldest).
        # We need to be careful with the order for X^T X? No, sum is commutative.
        
        # Get indices of valid data
        # If c < window_size, valid data is at indices 0 to c-1
        # If c >= window_size, valid data is the entire buffer (all overwritten)
        
        # For simplicity and speed, we'll just sum up the products.
        # We iterate through the valid samples.
        
        XtX = np.eye(self.d) * self.lam
        Xty = np.zeros(self.d)
        
        # Efficiently retrieve data
        if c < self.window_size:
            # Data is at [0, c)
            X = self.contexts[arm][:c]
            y = self.rewards[arm][:c]
        else:
            # Data is in the circular buffer. 
            # The oldest valid sample is at current_index[arm] (which was just overwritten? No, we update before select? No, update is after select. 
            # In update, we write to current_index, then increment.
            # So current_index points to the NEXT slot to write.
            # The oldest valid sample is at current_index[arm] (the one we are about to overwrite? No, if full, we overwrite the oldest.
            # Let's trace:
            # init: idx=0.
            # update 1: write to 0, idx=1. c=1. Valid: [0:1] -> [0]
            # update 50: write to 49, idx=0. c=50. Valid: all 50.
            # update 51: write to 0 (overwrites old 0), idx=1. c=51. Valid: all 50 (indices 1..49, 0).
            # So if c >= window_size, all slots in the buffer are valid.
            X = self.contexts[arm]
            y = self.rewards[arm]
            
        XtX += X.T @ X
        Xty += X.T @ y

        V = np.linalg.inv(XtX)
        theta = V @ Xty
        
        return theta, V

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta, _ = self._get_model_params(a)
                scores.append(theta @ context)
            return int(np.argmax(scores))

        # UCB
        scores = []
        for a in range(self.n):
            theta, V = self._get_model_params(a)
            mu = theta @ context
            # Uncertainty bonus
            bonus = self.C * np.sqrt(context @ V @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update circular buffer
        idx = self.current_index[arm]
        self.contexts[arm][idx] = context
        self.rewards[arm][idx] = reward
        
        self.current_index[arm] = (idx + 1) % self.window_size
        self.counts[arm] += 1
# EVOLVE-BLOCK-END
