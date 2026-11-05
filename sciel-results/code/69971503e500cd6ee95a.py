import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Dual State Variables (Stable and Volatile)
        self.theta_stable = [np.zeros(dim) for _ in range(n_arms)]
        self.theta_volatile = [np.zeros(dim) for _ in range(n_arms)]
        
        # Initial Covariance: Moderate for stable, High for volatile
        self.P_stable = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.P_volatile = [np.eye(dim) * 50.0 for _ in range(n_arms)]
        
        self.count = [0 for _ in range(n_arms)]
        
        # HMM Parameters
        # P(stable->stable), P(stable->volatile), P(volatile->stable), P(volatile->volatile)
        self.p_ss = 0.95
        self.p_sv = 0.05
        self.p_vs = 0.20
        self.p_vv = 0.80
        
        # Process Noise for each regime
        self.Q_stable = 1e-4
        self.Q_volatile = 0.1
        
        # Observation Noise
        self.sigma2 = 0.1
        
        # Initial Regime Probabilities (Prior)
        self.pi_stable = [0.9 for _ in range(n_arms)]
        self.pi_volatile = [0.1 for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.2
        self._alpha_min = 0.3
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0 # Seconds

    def _update_kalman_states(self, arm, x, r):
        """
        Update both Kalman filter states and the regime probabilities.
        """
        a = arm
        # 1. Predict Step for both states
        P_st_pred = self.P_stable[a] + self.Q_stable * np.eye(self.d)
        P_vol_pred = self.P_volatile[a] + self.Q_volatile * np.eye(self.d)
        
        # 2. Calculate Innovation for both states
        y_st = r - self.theta_stable[a] @ x
        y_vol = r - self.theta_volatile[a] @ x
        
        # 3. Update Step (Kalman) for Stable State
        Px_st = P_st_pred @ x
        denom_st = x @ Px_st + self.sigma2
        if denom_st < 1e-12:
            denom_st = 1e-12
        K_st = Px_st / denom_st
        
        self.theta_stable[a] = self.theta_stable[a] + K_st * y_st
        I = np.eye(self.d)
        IKxT_st = I - np.outer(K_st, x)
        P_st_new = IKxT_st @ P_st_pred @ IKxT_st.T + np.outer(K_st, K_st) * self.sigma2
        self.P_stable[a] = (P_st_new + P_st_new.T) / 2.0
        
        # 4. Update Step (Kalman) for Volatile State
        Px_vol = P_vol_pred @ x
        denom_vol = x @ Px_vol + self.sigma2
        if denom_vol < 1e-12:
            denom_vol = 1e-12
        K_vol = Px_vol / denom_vol
        
        self.theta_volatile[a] = self.theta_volatile[a] + K_vol * y_vol
        IKxT_vol = I - np.outer(K_vol, x)
        P_vol_new = IKxT_vol @ P_vol_pred @ IKxT_vol.T + np.outer(K_vol, K_vol) * self.sigma2
        self.P_volatile[a] = (P_vol_new + P_vol_new.T) / 2.0
        
        # 5. Update Regime Probabilities (HMM)
        # Log-likelihoods
        ll_st = -0.5 * (y_st * y_st / denom_st) - 0.5 * np.log(denom_st)
        ll_vol = -0.5 * (y_vol * y_vol / denom_vol) - 0.5 * np.log(denom_vol)
        
        # Convert to probabilities (softmax)
        max_ll = max(ll_st, ll_vol)
        exp_st = np.exp(ll_st - max_ll)
        exp_vol = np.exp(ll_vol - max_ll)
        sum_exp = exp_st + exp_vol
        
        if sum_exp < 1e-12:
            p_obs_st = 0.5
            p_obs_vol = 0.5
        else:
            p_obs_st = exp_st / sum_exp
            p_obs_vol = exp_vol / sum_exp
            
        # Prior probabilities (transition)
        prior_st = self.pi_stable[a] * self.p_ss + self.pi_volatile[a] * self.p_vs
        prior_vol = self.pi_stable[a] * self.p_sv + self.pi_volatile[a] * self.p_vv
        
        # Posterior probabilities
        post_st = prior_st * p_obs_st
        post_vol = prior_vol * p_obs_vol
        total_post = post_st + post_vol
        
        if total_post < 1e-12:
            self.pi_stable[a] = 0.5
            self.pi_volatile[a] = 0.5
        else:
            self.pi_stable[a] = post_st / total_post
            self.pi_volatile[a] = post_vol / total_post

    def select(self, context):
        # Deadline check
        if time.time() - self._start_time > self._deadline:
            # Fallback to greedy using stable state
            scores = [self.theta_stable[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Mixture of Predictions
            pred_st = self.theta_stable[a] @ context
            pred_vol = self.theta_volatile[a] @ context
            
            # Mixture of Uncertainties
            Px_st = self.P_stable[a] @ context
            var_st = context @ Px_st
            if var_st < 0:
                var_st = 0.0
            std_st = np.sqrt(var_st)
            
            Px_vol = self.P_volatile[a] @ context
            var_vol = context @ Px_vol
            if var_vol < 0:
                var_vol = 0.0
            std_vol = np.sqrt(var_vol)
            
            # Weighted Prediction
            w_st = self.pi_stable[a]
            w_vol = self.pi_volatile[a]
            
            pred_mix = w_st * pred_st + w_vol * pred_vol
            
            # Weighted Uncertainty (approximation: variance of mixture)
            # Var = w_st*Var_st + w_vol*Var_vol + w_st*w_vol*(Pred_st - Pred_vol)^2
            diff_pred = pred_st - pred_vol
            var_mix = w_st * var_st + w_vol * var_vol + w_st * w_vol * (diff_pred * diff_pred)
            std_mix = np.sqrt(var_mix)
            
            # Adaptive exploration: reduce bonus as we learn more
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))
            
            score = pred_mix + alpha_t * std_mix * count_factor
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward
        
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        self._update_kalman_states(a, x, r)
        
        self.count[a] += 1
