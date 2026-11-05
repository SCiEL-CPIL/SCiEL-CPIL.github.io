import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Contextual bandit: single shared weight vector
        self.theta = np.zeros(dim)
        self.P = np.eye(dim) * 10.0
        self.count = 0
        
        # Adaptive process noise
        self._innov_ema = 0.0
        self._innov_alpha = 0.1
        self._base_Q = 5e-4
        self._Q_max = 0.05
        
        # Exploration
        self.alpha = 1.0
        self._alpha_min = 0.2
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0
        
        # Reset mechanism
        self._consec_large = 0
        self._reset_threshold = 4
        self._reset_cov_scale = 8.0

    def _adaptive_Q(self):
        if self.count < 5:
            return self._base_Q
        scaled = np.tanh(self._innov_ema / 0.3)
        return self._base_Q + (self._Q_max - self._base_Q) * scaled

    def _maybe_reset(self, innov_sq):
        if innov_sq > 3.0:
            self._consec_large += 1
        else:
            self._consec_large = max(0, self._consec_large - 1)
        
        if self._consec_large >= self._reset_threshold:
            self.P = np.eye(self.d) * self._reset_cov_scale
            self._consec_large = 0

    def select(self, context):
        if time.time() - self._start_time > self._deadline:
            scores = []
            for a in range(self.n):
                # In contextual bandit, arm choice is encoded in context
                # For now, use greedy with shared theta
                scores.append(self.theta @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            pred = self.theta @ context
            Px = self.P @ context
            var = max(0.0, context @ Px)
            std = np.sqrt(var)
            
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.15))
            score = pred + alpha_t * std
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        x = np.asarray(context, dtype=np.float64)
        r = reward
        
        if time.time() - self._start_time > self._deadline:
            self.count += 1
            return

        Q = self._adaptive_Q()
        P_pred = self.P + Q * np.eye(self.d)
        
        y = r - self.theta @ x
        
        sigma2 = 0.1
        Px = P_pred @ x
        denom = x @ Px + sigma2
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom
        
        self.theta = self.theta + K * y
        
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * sigma2
        self.P = (P_new + P_new.T) / 2.0
        
        innov_sq = (y * y) / denom
        self._innov_ema = (1.0 - self._innov_alpha) * self._innov_ema + self._innov_alpha * innov_sq
        
        self._maybe_reset(innov_sq)
        
        self.count += 1
