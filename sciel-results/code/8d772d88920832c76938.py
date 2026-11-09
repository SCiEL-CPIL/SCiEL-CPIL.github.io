# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """Kernelized non-stationary linear bandit with time-dependent RBF feature mapping."""
    
    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        
        # Time-dependent RBF feature mapping parameters
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0
        
        # RBF kernel parameters
        self.gamma = 0.1  # RBF bandwidth
        self.decay_rate = 0.05  # How fast time features decay
        
        # For each arm, maintain a linear model in the expanded feature space
        # Feature space: [original_context, time_features]
        # We'll use a fixed set of time-based RBF features
        
        # Number of time basis functions
        self.n_time_basis = 5
        self.time_centers = np.linspace(0, 1000, self.n_time_basis)
        
        # Total feature dimension = original dim + time basis
        self.total_dim = dim + self.n_time_basis
        
        # Initialize A_inv and b for each arm in the expanded space
        reg = 1.0
        self.A_inv = [np.eye(self.total_dim) / reg for _ in range(n_arms)]
        self.b = [np.zeros(self.total_dim) for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.5
        
        # Track recent rewards for potential adaptation
        self.reward_mean = [0.0 for _ in range(n_arms)]
        self.reward_count = [0 for _ in range(n_arms)]
    
    def _compute_time_features(self, t):
        """Compute time-dependent RBF features for current time step."""
        # Normalize time to [0, 1] range for better numerical behavior
        t_norm = t / 1000.0
        
        # RBF features: exp(-gamma * (t - center)^2)
        time_feats = np.zeros(self.n_time_basis)
        for i in range(self.n_time_basis):
            time_feats[i] = np.exp(-self.gamma * (t_norm - self.time_centers[i] / 1000.0) ** 2)
        
        # Add a constant term for bias
        time_feats = np.concatenate([time_feats, [1.0]])
        
        return time_feats
    
    def _expand_context(self, context):
        """Expand context with time-dependent features."""
        context = np.asarray(context, dtype=np.float64)
        time_feats = self._compute_time_features(self.t)
        # Combine original context and time features
        expanded = np.concatenate([context, time_feats])
        return expanded
    
    def _select_arm(self, context):
        """Select arm using UCB in the expanded feature space."""
        self.t += 1
        
        expanded_ctx = self._expand_context(context)
        
        scores = []
        for a in range(self.n):
            A_inv = self.A_inv[a]
            b = self.b[a]
            
            # theta = A_inv @ b
            theta = A_inv @ b
            mu = theta @ expanded_ctx
            
            # Compute variance term: expanded_ctx^T A_inv expanded_ctx
            v = expanded_ctx @ A_inv @ expanded_ctx
            
            # UCB score
            score = mu + self.alpha * np.sqrt(max(v, 0.0))
            scores.append(score)
        
        best_arm = int(np.argmax(scores))
        return best_arm
    
    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0
        
        arm = self._select_arm(context)
        return arm
    
    def update(self, context, arm, reward):
        expanded_ctx = self._expand_context(context)
        
        # Update reward tracking
        self.reward_mean[arm] = (self.reward_mean[arm] * self.reward_count[arm] + reward) / (self.reward_count[arm] + 1)
        self.reward_count[arm] += 1
        
        # Update A_inv and b using Sherman-Morrison-Woodbury
        # A_new = A_old + xx^T
        # A_inv_new = A_inv_old - (A_inv_old @ x @ x^T @ A_inv_old) / (1 + x^T @ A_inv_old @ x)
        
        x = expanded_ctx
        A_inv_old = self.A_inv[arm]
        
        # Calculate denominator
        A_inv_x = A_inv_old @ x
        denom = 1.0 + x @ A_inv_x
        
        # Update A_inv
        correction = np.outer(A_inv_x, A_inv_x) / denom
        self.A_inv[arm] = A_inv_old - correction
        
        # Update b
        self.b[arm] = self.b[arm] + reward * x

# EVOLVE-BLOCK-END
