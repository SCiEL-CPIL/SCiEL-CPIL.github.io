<<<<<<< SEARCH
    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        if self._check_deadline():
            return

        # Compute innovation before predict
        pred = float(self.theta[arm] @ ctx)
        innov_raw = reward - pred

        # Context norm squared for adaptive Q
        ctx_norm_sq = float(ctx @ ctx)

        # Adaptive process noise
        q = self._adaptive_q(arm, innov_raw, ctx_norm_sq)

        # Kalman predict step
        self._predict(arm, q)

        # Kalman update step (uses robustified innovation internally)
        self._update(arm, ctx, reward)

        # Check for catastrophic shift and reset if needed
        self._check_reset(arm, innov_raw)

        # Periodic numerical stabilization every 50 steps
        if self.step_count % 50 == 0:
            for a in range(self.n):
                P = self.P[a]
                if not np.all(np.isfinite(P)):
                    self.P[a] = np.eye(self.d) * 10.0
                    self.theta[a] = np.zeros(self.d)
                    self.count[a] = 0
                    self.innov_ema[a] = 0.0
                    self.consecutive_large_innov[a] = 0
=======
    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        if self._check_deadline():
            return

        # Compute innovation before predict using blended estimate
        pred = float(self.theta[arm] @ ctx)
        innov_raw = reward - pred

        # Update regime probability based on innovation
        std_blend = float(np.sqrt(max(ctx @ self.P[arm] @ ctx, 0.0)))
        regime_prob = self._update_regime_prob(arm, innov_raw, std_blend)

        # Context norm squared for adaptive Q
        ctx_norm_sq = float(ctx @ ctx)

        # Adaptive process noise for each regime
        q_stable = self._adaptive_q(arm, innov_raw, ctx_norm_sq, 0.0)
        q_volatile = self._adaptive_q(arm, innov_raw, ctx_norm_sq, 1.0)

        # Kalman predict step for both regimes
        self._predict_dual(arm, q_stable, q_volatile)

        # Dual-state Kalman update step
        self._update_dual(arm, ctx, reward, regime_prob)

        # Check for catastrophic shift and reset if needed
        self._check_reset(arm, innov_raw)

        # Periodic numerical stabilization every 50 steps
        if self.step_count % 50 == 0:
            for a in range(self.n):
                for P_list in [self.P_stable, self.P_volatile, self.P]:
                    P = P_list[a]
                    if not np.all(np.isfinite(P)):
                        P_list[a] = np.eye(self.d) * 10.0
                        self.theta[a] = np.zeros(self.d)
                        self.theta_stable[a] = np.zeros(self.d)
                        self.theta_volatile[a] = np.zeros(self.d)
                        self.count[a] = 0
                        self.innov_ema[a] = 0.0
                        self.consecutive_large_innov[a] = 0
                        self.regime_prob[a] = 0.2
>>>>>>> REPLACE
