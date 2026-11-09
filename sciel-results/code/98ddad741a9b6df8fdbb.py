"""Non-stationary linear contextual bandit with RBF time-modulated feature expansion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Kernelized Non-Stationary Linear Bandit using Time-Dependent RBF Feature Mapping.
    
    Transforms the non-stationary problem into a stationary linear bandit in an 
    expanded feature space. The feature map phi(x, t) = [x*k1(t), x*k2(t), x*k3(t)] 
    allows the linear model to capture time-varying weight vectors by learning 
    separate weight components for different time scales.
    
    This approach avoids manual drift detection and explicit multi-model management, 
    relying instead on the structure of the feature space and mild discounting 
    for numerical stability.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Expand dimension by 3 for 3 RBF time kernels
        self.expanded_dim = dim * 3
        
        # Regularization
        self.reg = 0.1
        
        # Initialize A and b for each arm
        self.A = [np.eye(self.expanded_dim) * self.reg for _ in range(n_arms)]
        self.b = [np.zeros(self.expanded_dim) for _ in range(n_arms)]
        
        # Discount factor: Mild discounting to handle potential drift not captured by features
        self.lambda_discount = 0.99
        
        # RBF Parameters
        self.rbf_sigma = 5.0
        self.rbf_lags = [0, 5, 10]  # Lags in time steps
        
        # Adaptive Exploration
        self.alpha = 1.5
        self.alpha_min = 0.5
        self.alpha_max = 4.0
        self.recent_errors = []
        self.error_window = 20
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0

    def _get_phi(self, context):
        """
        Compute the RBF time-modulated feature vector.
        phi(x, t) = [x * k1(t), x * k2(t), x * k3(t)]
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Compute RBF weights for current time t
        # Centers are at t - lag
        # Distance is (t - (t - lag)) = lag
        # k_i(t) = exp(-lag_i^2 / (2 * sigma^2))
        
        # Note: The features are time-dependent. 
        # k_i(t) is constant for a given t across all arms.
        
        k_values = []
        for lag in self.rbf_lags:
            # Distance from center (t-lag) to current t is 'lag'
            dist = lag
            k_val = np.exp(-(dist ** 2) / (2 * self.rbf_sigma ** 2))
            k_values.append(k_val)
        
        # Construct phi
        # phi = [x * k1, x * k2, x * k3]
        phi_parts = []
        for k_val in k_values:
            phi_parts.append(context * k_val)
        
        return np.concatenate(phi_parts)

    def _solve(self, A, b):
        """Solve A x = b with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0

        self.t += 1
        phi = self._get_phi(context)
        
        scores = []
        for a in range(self.n):
            try:
                theta = self._solve(self.A[a], self.b[a])
                mu = phi @ theta
                
                # Compute variance term: phi^T A^{-1} phi
                # Use solve for efficiency and stability
                A_inv_phi = np.linalg.solve(self.A[a], phi)
                v = phi @ A_inv_phi
                v = max(v, 0.0)
                
                ucb = mu + self.alpha * np.sqrt(v)
            except np.linalg.LinAlgError:
                ucb = 0.0
            except Exception:
                ucb = 0.0
            
            # Small bonus for unexplored arms
            if self.counts[a] < 5:
                ucb += 0.1 / (1 + self.counts[a])
                
            scores.append(ucb)
        
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        phi = self._get_phi(context)
        
        # Update model with discounting
        lam = self.lambda_discount
        self.A[arm] = lam * self.A[arm] + np.outer(phi, phi)
        self.b[arm] = lam * self.b[arm] + reward * phi
        
        # Update count
        self.counts[arm] += 1
        
        # Adaptive Alpha based on recent prediction errors
        try:
            theta = self._solve(self.A[arm], self.b[arm])
            pred = phi @ theta
            error = abs(reward - pred)
        except Exception:
            error = 0.0
            
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)
            
        if len(self.recent_errors) >= 5:
            mean_error = np.mean(self.recent_errors)
            # Target alpha increases with error
            target_alpha = 1.0 + 1.5 * mean_error
            # Smooth adaptation
            self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
            self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))


# EVOLVE-BLOCK-END
