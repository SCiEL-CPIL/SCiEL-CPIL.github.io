"""
Non-stationary linear contextual bandit policy using a shared global model
with a fixed-size sliding window.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Strategy:
    - Uses a SINGLE global linear model shared across all arms.
    - Maintains a fixed-size sliding window of the last W observations.
    - This allows the model to fully "forget" old data when the window slides,
      which is robust to non-stationarity without needing adaptive beta tuning.
    - Uses LinUCB-style exploration on the global model.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        
        # Global model parameters
        self.A = np.eye(dim) * self.lam
        self.b = np.zeros(dim)
        
        # Sliding window storage
        # Store (context, reward) pairs. We don't need to store arm separately
        # because the model is global (y = x^T theta).
        # However, to be safe if the model was per-arm, we'd need it.
        # Here, the model is y = x^T theta, so arm identity is not part of the feature vector
        # unless we one-hot encode it. 
        # Wait, standard linear bandit is y = x^T theta_arm.
        # If I use a GLOBAL model, I am assuming y = x^T theta for all arms.
        # This is a strong assumption. 
        # Let's re-read the task: "five-arm linear Gaussian contextual bandit".
        # Usually this means r_a = x^T theta_a + noise.
        # If I use a global model, I must include the arm index in the features.
        # Let's augment the context with a one-hot encoding of the arm.
        # New dim = dim + n_arms.
        
        self.aug_dim = dim + n_arms
        self.A = np.eye(self.aug_dim) * self.lam
        self.b = np.zeros(self.aug_dim)
        
        # Window size
        # Needs to be large enough to learn, small enough to adapt.
        # 100-200 is a reasonable start for non-stationary.
        self.window_size = 150
        self.history = []
        
        # Exploration parameter
        self.alpha = 1.5
        
        # Tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

    def _augment_context(self, context, arm):
        """
        Augment context with one-hot arm indicator.
        context: (d,)
        arm: int
        returns: (d + n_arms,)
        """
        x_aug = np.zeros(self.aug_dim)
        x_aug[:self.d] = context
        x_aug[self.d + arm] = 1.0
        return x_aug

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fast fallback: pick arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    x_aug = self._augment_context(context, a)
                    theta = np.linalg.solve(self.A, self.b)
                    scores.append(theta @ x_aug)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        try:
            theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            # If solve fails, fallback to greedy or random
            return int(np.random.randint(self.n))

        for a in range(self.n):
            x_aug = self._augment_context(context, a)
            mu = theta @ x_aug
            
            # Variance calculation for UCB
            # var = x^T A^{-1} x
            try:
                inv_A_x = np.linalg.solve(self.A, x_aug)
                var = x_aug @ inv_A_x
                if var < 0:
                    var = 0.0
                uc = self.alpha * np.sqrt(var)
            except np.linalg.LinAlgError:
                uc = 0.0
                
            scores.append(mu + uc)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Deadline guard for minimal update
        if time.time() > self.deadline:
            # Just update the window, skip matrix updates to save time?
            # Actually, matrix update is O(D^2). Window append is O(1).
            # But if we are close to deadline, we should just stop updating
            # or do the absolute minimum.
            # Let's just append and pop, and do a quick rank-1 update.
            x_aug = self._augment_context(context, arm)
            
            # Append to history
            self.history.append((x_aug, reward))
            if len(self.history) > self.window_size:
                old_x, old_r = self.history.pop(0)
                # Remove old data from A and b
                self.A -= np.outer(old_x, old_x)
                self.b -= old_r * old_x
            
            # Add new data
            self.A += np.outer(x_aug, x_aug)
            self.b += reward * x_aug
            return

        x_aug = self._augment_context(context, arm)
        
        # Append to history
        self.history.append((x_aug, reward))
        
        if len(self.history) > self.window_size:
            # Remove the oldest observation
            old_x, old_r = self.history.pop(0)
            # Update A and b by removing the old contribution
            # A = A - outer(old_x, old_x)
            # b = b - old_r * old_x
            self.A -= np.outer(old_x, old_x)
            self.b -= old_r * old_x
            
        # Add the new observation
        self.A += np.outer(x_aug, x_aug)
        self.b += reward * x_aug
        
        # Optional: Periodic regularization or re-orthogonalization?
        # For sliding window, A might become ill-conditioned if data is collinear.
        # Adding a small lambda * I periodically can help, but it changes the effective window.
        # For now, we rely on the initial lambda and the window size.
        
        # Adaptive Alpha?
        # In a sliding window, the uncertainty naturally reflects the amount of data in the window.
        # If the window is full, uncertainty is lower. If empty, higher.
        # We can keep alpha fixed or slowly decay it.
        # Let's keep alpha fixed for simplicity in this structural change.
        # self.alpha = max(1.0, self.alpha * 0.999)
# EVOLVE-BLOCK-END
