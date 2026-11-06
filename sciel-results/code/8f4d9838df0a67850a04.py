"""Non-stationary linear contextual bandit with global model + arm biases."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear bandit using a global linear model with arm-specific biases.
    
    Model: reward = context @ theta + bias[arm] + noise
    
    Uses forgetting factor for non-stationarity. The global model captures shared
    context-feature relationships, while arm biases capture arm-specific offsets.
    This reduces parameter count vs per-arm models and improves sample efficiency.
    
    Exploration: UCB bonus based on uncertainty of the global model.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.99  # Forgetting factor
        self.alpha = 1.0   # Exploration parameter
        self.lambda_reg = 1.0  # Ridge regularization
        
        # Global linear model
        self.A = np.eye(dim) * self.lambda_reg
        self.b = np.zeros(dim)
        
        # Arm-specific biases (scalar per arm)
        self.bias_A = np.ones(n_arms) * self.lambda_reg  # Accumulator for bias
        self.bias_b = np.zeros(n_arms)  # Sum of rewards weighted by context
        
        # Track elapsed time for deadline guard
        self._start_time = time.time()
        self._deadline = 150.0  # Wall-clock limit in seconds
        self._safety_margin = 5.0  # Safety margin in seconds
        
        # Counters
        self._round = 0

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        elapsed = time.time() - self._start_time
        return elapsed > (self._deadline - self._safety_margin)

    def select(self, context):
        """Select arm based on UCB scores."""
        self._round += 1
        
        # If near deadline, use greedy exploitation only
        if self._check_deadline():
            # Quick greedy: use current estimates without exploration bonus
            theta = np.linalg.solve(self.A, self.b)
            scores = []
            for a in range(self.n):
                # Estimate bias for arm a
                if self.bias_A[a] > 1e-10:
                    bias_est = self.bias_b[a] / self.bias_A[a]
                else:
                    bias_est = 0.0
                scores.append(theta @ context + bias_est)
            return int(np.argmax(scores))
        
        # Compute theta estimate
        theta = np.linalg.solve(self.A, self.b)
        
        # Compute uncertainty
        inv_A = np.linalg.inv(self.A)
        context_vec = np.asarray(context)
        uncertainty = context_vec @ inv_A @ context_vec
        bonus = self.alpha * np.sqrt(max(uncertainty, 0.0))
        
        # Score each arm: global prediction + arm bias estimate
        scores = []
        for a in range(self.n):
            # Estimate bias for arm a
            if self.bias_A[a] > 1e-10:
                bias_est = self.bias_b[a] / self.bias_A[a]
            else:
                bias_est = 0.0
            
            # Uncertainty contribution for bias (approximate)
            bias_unc = 1.0 / self.bias_A[a]
            bias_bonus = self.alpha * np.sqrt(max(bias_unc, 0.0))
            
            scores.append(theta @ context_vec + bias_est + bonus + bias_bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update model with new observation."""
        self._round += 1
        context_vec = np.asarray(context)
        
        # Check deadline - if exceeded, just return
        if self._check_deadline():
            # Still update to maintain consistency, but keep it fast
            pass
        
        # Update global model with forgetting
        self.A = self.gamma * self.A + np.outer(context_vec, context_vec)
        self.b = self.gamma * self.b + reward * context_vec
        
        # Update arm-specific bias
        self.bias_A[arm] = self.gamma * self.bias_A[arm] + 1.0
        self.bias_b[arm] = self.gamma * self.bias_b[arm] + reward
# EVOLVE-BLOCK-END
