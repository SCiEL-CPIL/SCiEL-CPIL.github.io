import numpy as np
import time
from collections import deque

class Policy:
    """
    Global Sliding-Window Linear Contextual Bandit.
    
    Strategy:
    - Instead of per-arm models with exponential forgetting, we use a single global model.
    - The context is augmented with one-hot arm indicators (dimension d + n_arms).
    - We maintain a sliding window of the last W observations for all arms.
    - The model is re-fitted via ridge regression on this window periodically or when the window is full.
    - This allows immediate exploitation of cross-arm correlations (if context features are shared)
      and hard discarding of stale data (better for non-stationarity than exponential decay).
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.aug_dim = dim + n_arms
        
        # Sliding window storage
        self.window_size = 150  # Tunable: large enough for stable estimate, small enough for adaptation
        self.contexts = deque(maxlen=self.window_size)
        self.arms = deque(maxlen=self.window_size)
        self.rewards = deque(maxlen=self.window_size)
        
        # Model parameters
        self.theta = np.zeros(self.aug_dim)
        self.lambda_ridge = 1.0  # Regularization strength
        
        # Exploration
        self.alpha = 1.5
        self.alpha_min = 0.5
        self.alpha_max = 3.0
        
        # Timing
        self.start_time = time.time()
        self.deadline = 110.0
        
        # Refit counter
        self.refit_interval = 10
        self.steps_since_refit = 0
        
        # Pre-allocate matrices for speed
        self.X = None
        self.y = None

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _augment_context(self, context, arm):
        """Augment context with one-hot arm indicator."""
        # context is shape (d,)
        # arm_indicator is shape (n,)
        # result is shape (d+n,)
        indicator = np.zeros(self.n)
        indicator[arm] = 1.0
        return np.concatenate([context, indicator])

    def _refit_model(self):
        """Re-fit the global ridge regression model on the sliding window."""
        if len(self.contexts) == 0:
            return
            
        # Convert deques to arrays
        X_list = []
        y_list = []
        for ctx, arm, rew in zip(self.contexts, self.arms, self.rewards):
            X_list.append(self._augment_context(ctx, arm))
            y_list.append(rew)
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Ridge Regression: theta = (X^T X + lambda I)^-1 X^T y
        # For numerical stability and speed, use solve instead of inv
        A = X.T @ X + self.lambda_ridge * np.eye(self.aug_dim)
        b = X.T @ y
        
        try:
            self.theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            self.theta = np.linalg.pinv(A) @ b

    def select(self, context):
        if not self._check_deadline():
            # Deadline guard: pick best mean
            scores = []
            for a in range(self.n):
                ctx_aug = self._augment_context(context, a)
                scores.append(ctx_aug @ self.theta)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            ctx_aug = self._augment_context(context, a)
            mu = ctx_aug @ self.theta
            
            # Variance estimation for UCB
            # We approximate variance using the diagonal of the covariance matrix
            # Cov = (X^T X + lambda I)^-1
            # To avoid storing/computing full inverse every step, we can cache it or approximate.
            # For simplicity and speed, we'll compute the variance term on the fly if needed,
            # but since we re-fit periodically, we can cache the inverse of A from the last refit.
            
            # Let's cache the inverse of A in _refit_model
            if not hasattr(self, 'inv_A'):
                self.inv_A = np.eye(self.aug_dim)
            
            var = ctx_aug @ self.inv_A @ ctx_aug
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        # Add to sliding window
        self.contexts.append(context)
        self.arms.append(arm)
        self.rewards.append(reward)
        
        self.steps_since_refit += 1
        
        # Refit periodically
        if self.steps_since_refit >= self.refit_interval or len(self.contexts) == 1:
            self._refit_model()
            self.steps_since_refit = 0
            # Cache inverse of A for variance calculation in select()
            if len(self.contexts) > 0:
                X_list = []
                for ctx, a in zip(self.contexts, self.arms):
                    X_list.append(self._augment_context(ctx, a))
                X = np.array(X_list)
                A = X.T @ X + self.lambda_ridge * np.eye(self.aug_dim)
                try:
                    self.inv_A = np.linalg.inv(A)
                except:
                    self.inv_A = np.eye(self.aug_dim)
        
        # Adaptive Alpha: 
        # If recent rewards are high relative to predictions, we are doing well, reduce exploration?
        # Or if we are in a non-stationary phase, we might want to explore more.
        # Simple heuristic: keep alpha fixed or decay it slightly over time to exploit more.
        # Let's stick to a fixed alpha for stability in this new architecture, 
        # as the sliding window itself handles the non-stationarity.
        pass
