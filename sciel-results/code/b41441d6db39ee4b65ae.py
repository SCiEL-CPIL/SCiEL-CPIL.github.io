import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # RLS Parameters
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        # Initialize P (inverse covariance) with high uncertainty
        # Small reg means high initial P, encouraging exploration
        self.reg = 1.0
        self.P = [np.eye(self.d) / self.reg for _ in range(self.n)]
        
        # Forgetting factor parameters
        self.lambda_min = 0.85
        self.lambda_max = 0.99
        self.lambda_base = 0.95
        
        # Adaptive lambda tracking
        # Store recent squared innovations
        self.hist_len = 20
        self.innov_hist = [np.zeros(self.hist_len) for _ in range(self.n)]
        self.hist_idx = [0 for _ in range(self.n)]
        
        # UCB parameter
        self.alpha = 1.5
        
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy
            context_vec = np.array(context)
            scores = [np.dot(self.theta[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            mu = np.dot(self.theta[a], context_vec)
            # Variance estimate: x^T P x
            P_x = np.dot(self.P[a], context_vec)
            var = np.dot(context_vec, P_x)
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context_vec = np.array(context)
        
        # 1. Calculate Innovation
        pred = np.dot(self.theta[arm], context_vec)
        innovation = reward - pred
        sq_innov = innovation ** 2
        
        # 2. Update Innovation History
        self.innov_hist[arm][self.hist_idx[arm]] = sq_innov
        self.hist_idx[arm] = (self.hist_idx[arm] + 1) % self.hist_len
        
        # 3. Adaptive Forgetting Factor
        # If recent errors are large, we want to forget faster (lower lambda)
        # Calculate mean squared error of recent innovations
        mse = np.mean(self.innov_hist[arm])
        
        # Map MSE to lambda
        # If MSE is 0, lambda = lambda_max (stable)
        # If MSE is large, lambda = lambda_min (fast adaptation)
        # Use a simple scaling
        # Assume typical reward variance is ~1. If mse > 1, something is changing.
        # lambda = lambda_max - (lambda_max - lambda_min) * min(1, mse / threshold)
        threshold = 1.0 
        factor = np.clip(mse / threshold, 0.0, 1.0)
        lam = self.lambda_max - (self.lambda_max - self.lambda_min) * factor
        
        # 4. RLS Update
        # P_new = (1/lam) * (P - (P x x^T P) / (lam + x^T P x))
        P = self.P[arm]
        theta = self.theta[arm]
        
        Px = np.dot(P, context_vec)
        denom = lam + np.dot(context_vec, Px)
        
        if denom < 1e-12:
            denom = 1e-12
            
        gain = Px / denom
        
        # Update P
        # P_new = (1/lam) * (P - outer(gain, x))  <-- Wait, gain is vector, x is vector
        # The term is P x x^T P. Since Px = P x, then P x x^T P = (P x) (x^T P) = Px (Px)^T
        # So P_new = (1/lam) * (P - np.outer(Px, Px) / denom)
        # Note: gain = Px / denom, so outer(Px, Px)/denom = denom * outer(gain, gain) ? No.
        # outer(Px, Px) / denom = denom * outer(gain, gain) is wrong.
        # outer(Px, Px) / denom = (denom * gain) (denom * gain)^T / denom = denom * outer(gain, gain).
        # So P_new = (1/lam) * (P - denom * outer(gain, gain))
        
        P_new = (1.0 / lam) * (P - np.outer(Px, Px) / denom)
        
        # Symmetrize for stability
        P_new = (P_new + P_new.T) / 2.0
        
        # Update Theta
        # theta_new = theta + gain * innovation
        theta_new = theta + gain * innovation
        
        self.P[arm] = P_new
        self.theta[arm] = theta_new
        
        self.t += 1
