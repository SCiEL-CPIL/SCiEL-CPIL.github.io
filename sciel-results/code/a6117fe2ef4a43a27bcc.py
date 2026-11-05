# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit with a shared linear model.
    
    Uses a single linear model over an augmented feature space (context + one-hot arm indicator).
    This enforces parameter sharing across arms, improving sample efficiency.
    Implements exponentially weighted least squares (EWLS) with Sherman-Morrison updates.
    The forgetting factor (gamma) is adapted based on prediction error (innovation).
    """
    
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: context features + one-hot arm indicators
        self.D = self.d + self.n
        
        # Initialize the unified model
        # A is the Hessian (covariance) matrix, b is the gradient vector
        # We use a ridge initialization to ensure invertibility
        self.ridge = 1.0
        self.A = np.eye(self.D) * self.ridge
        self.b = np.zeros(self.D)
        
        # Precompute inverse for Sherman-Morrison
        self.A_inv = np.eye(self.D) / self.ridge
        self.theta = np.zeros(self.D)
        
        # Adaptive forgetting factor
        # Start with a moderate gamma (0.95) to allow initial learning but responsive to drift
        self.gamma = 0.95
        # Min and max bounds for gamma
        self.gamma_min = 0.80
        self.gamma_max = 0.99
        
        # Exploration parameter
        self.alpha = 1.5
        
        # Deadline tracking
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin
        self.round_count = 0
        
        # Innovation tracking for adaptive gamma
        # We keep a short history of squared innovations to estimate recent drift
        self.innovation_history = []
        self.innovation_window = 10
        
    def _get_feature_vector(self, context, arm):
        """
        Construct the augmented feature vector for the unified model.
        Feature vector = [context, one_hot(arm)]
        """
        # context is already np.array of shape (d,)
        # one_hot is of shape (n,)
        one_hot = np.zeros(self.n)
        one_hot[arm] = 1.0
        return np.concatenate((context, one_hot))
    
    def _sherman_morrison_update(self, x, reward, gamma):
        """
        Update A_inv and theta using Sherman-Morrison formula.
        x: feature vector (D,)
        reward: scalar
        gamma: forgetting factor
        """
        inv = self.A_inv
        # x_new = x / gamma
        x_g = x / gamma
        Ax = inv @ x_g
        denom = 1.0 + x_g @ Ax
        
        if abs(denom) < 1e-12:
            # Fallback: recompute inverse (rare)
            self.A = gamma * self.A + np.outer(x, x)
            self.A_inv = np.linalg.inv(self.A)
            self.theta = self.A_inv @ self.b
            return
        
        # A_new^{-1} = (1/gamma) * (A_old^{-1} - outer(Ax, Ax) / denom)
        self.A_inv = (inv - np.outer(Ax, Ax) / denom) / gamma
        
        # Update b: b_new = gamma * b_old + reward * x
        self.b = gamma * self.b + reward * x
        # Update theta = A_new^{-1} @ b_new
        self.theta = self.A_inv @ self.b
        
        # Update A for potential fallback
        self.A = gamma * self.A + np.outer(x, x)

    def select(self, context):
        self.round_count += 1
        
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return int(np.argmax(context)) % self.n
        
        context = np.asarray(context, dtype=np.float64)
        
        scores = []
        for a in range(self.n):
            x = self._get_feature_vector(context, a)
            
            # Mean prediction
            mean = self.theta @ x
            
            # UCB Bonus
            # var = x^T A_inv x
            var = x @ self.A_inv @ x
            bonus = self.alpha * np.sqrt(max(var, 0.0))
            
            scores.append(mean + bonus)
        
        scores = np.array(scores)
        
        # Epsilon-greedy for initial exploration
        if self.round_count < 50 and np.random.random() < 0.2:
            return int(np.random.randint(self.n))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Construct feature vector
        x = self._get_feature_vector(context, arm)
        
        # Calculate prediction (before update)
        pred = self.theta @ x
        innovation = reward - pred
        innovation_sq = innovation ** 2
        
        # Update innovation history
        self.innovation_history.append(innovation_sq)
        if len(self.innovation_history) > self.innovation_window:
            self.innovation_history.pop(0)
        
        # Adaptive Forgetting
        # If recent innovations are large, decrease gamma (forget faster)
        # If recent innovations are small, increase gamma (forget slower)
        if len(self.innovation_history) > 0:
            avg_innovation_sq = np.mean(self.innovation_history)
            # Map avg_innovation_sq to target gamma
            # We want high error -> low gamma.
            # Use a soft mapping. 
            # Let's say if avg_innovation_sq is 0, target_gamma = 0.99
            # If avg_innovation_sq is large (e.g., 4.0), target_gamma = 0.80
            # target_gamma = gamma_min + (gamma_max - gamma_min) * exp(-avg_innovation_sq / scale)
            # Or simpler: linear clamp
            # Let's use a tanh-based mapping for smoothness
            scale = 1.0
            # Normalize error to [0, 1] roughly
            err_norm = np.tanh(avg_innovation_sq / scale)
            # Map [0, 1] to [gamma_max, gamma_min]
            target_gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * err_norm
            
            # Smoothly update gamma towards target
            # Faster adaptation to drift: high weight on target
            self.gamma = 0.8 * self.gamma + 0.2 * target_gamma
            self.gamma = np.clip(self.gamma, self.gamma_min, self.gamma_max)
        
        # Perform the update
        self._sherman_morrison_update(x, reward, self.gamma)
        
        # Slight decay of alpha to exploit more over time, but keep a floor
        self.alpha = max(0.5, self.alpha * 0.9999)

# EVOLVE-BLOCK-END
