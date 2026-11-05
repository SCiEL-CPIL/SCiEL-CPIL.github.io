import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0
        
        # Ensemble of Kalman Filters
        # K different levels of process noise (forgetting)
        self.K = 3
        self.q_values = [0.01, 0.1, 1.0] # Low, Med, High noise
        
        # State: theta[arm][k], P[arm][k]
        self.theta = [[np.zeros(dim) for _ in range(self.K)] for _ in range(n_arms)]
        self.P = [[np.eye(dim) * 100.0 for _ in range(self.K)] for _ in range(n_arms)]
        
        # Weights for ensemble members per arm
        # w[arm][k]
        self.w = [[1.0 / self.K for _ in range(self.K)] for _ in range(n_arms)]
        
        # Exploration
        self.alpha = 1.5
        self.alpha_decay = 0.999
        self.alpha_min = 0.5
        
        # Weight update params
        self.beta = 0.5 # Learning rate for weight update
        self.eta = 0.05 # Prior weight to prevent collapse
        
        self.round_num = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        if not self._check_deadline():
            # Greedy fallback using weighted mean
            scores = []
            for a in range(self.n):
                mu = sum(self.w[a][k] * (self.theta[a][k] @ context) for k in range(self.K))
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Weighted mean
            mu = 0.0
            # Weighted variance (approximate: sum w_k * (var_k + (mu_k - mu)^2))
            # But for speed, just use weighted var term
            var_term = 0.0
            
            mus = []
            vars = []
            for k in range(self.K):
                m = self.theta[a][k] @ context
                p_ctx = self.P[a][k] @ context
                v = context @ p_ctx
                if v < 0: v = 0.0
                mus.append(m)
                vars.append(v)
                
            mu = sum(self.w[a][k] * mus[k] for k in range(self.K))
            # Variance of the mixture
            # E[X^2] - (E[X])^2
            # Approx: sum w_k * (var_k + (mu_k - mu)^2)
            for k in range(self.K):
                var_term += self.w[a][k] * (vars[k] + (mus[k] - mu)**2)
                
            if var_term < 0: var_term = 0.0
            bonus = self.alpha * np.sqrt(var_term)
            scores.append(mu + bonus)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1
        
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
        errors = []
        
        for k in range(self.K):
            q = self.q_values[k]
            
            # Prediction
            P_pred = self.P[arm][k] + q * np.eye(self.d)
            
            # Update
            R = 1.0
            y_hat = self.theta[arm][k] @ context
            innovation = reward - y_hat
            errors.append(innovation)
            
            S = context @ P_pred @ context + R
            
            if S > 1e-10:
                K_gain = P_pred @ context / S
            else:
                K_gain = np.zeros(self.d)
                
            self.theta[arm][k] = self.theta[arm][k] + K_gain * innovation
            
            I_KH = np.eye(self.d) - np.outer(K_gain, context)
            self.P[arm][k] = I_KH @ P_pred @ I_KH.T + np.outer(K_gain, K_gain) * R
            self.P[arm][k] = 0.5 * (self.P[arm][k] + self.P[arm][k].T)
            
        # Update Weights
        # w_k_new = w_k_old * exp(-beta * error_k^2)
        new_w = []
        for k in range(self.K):
            w_k = self.w[arm][k]
            e = errors[k]
            # Clip error to prevent exp underflow/overflow issues if error is huge
            e_sq = e * e
            if e_sq > 100: e_sq = 100
            new_w_k = w_k * np.exp(-self.beta * e_sq)
            new_w.append(new_w_k)
        
        # Add prior to prevent zero weights
        for k in range(self.K):
            new_w[k] = (1 - self.eta) * new_w[k] + self.eta * (1.0 / self.K)
            
        # Normalize
        total_w = sum(new_w)
        if total_w > 1e-10:
            for k in range(self.K):
                self.w[arm][k] = new_w[k] / total_w
        else:
            for k in range(self.K):
                self.w[arm][k] = 1.0 / self.K
