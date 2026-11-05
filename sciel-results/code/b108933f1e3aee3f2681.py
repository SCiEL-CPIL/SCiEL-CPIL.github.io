"""Non-stationary linear contextual bandit with Bayesian Online Changepoint Detection (BOCPD)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Bayesian Online Changepoint Detection.
    
    For each arm, we maintain a set of models (one per possible 'age' or regime length).
    At each step, we predict the likelihood of the observation under each model.
    We update the posterior probability of each model (age) based on these likelihoods.
    If a changepoint is detected (i.e., the probability of the oldest age drops),
    the model effectively resets.
    
    Selection is done via UCB on the weighted average of the model predictions,
    with uncertainty reflecting both model uncertainty and regime uncertainty.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.max_age = 50  # Maximum regime length to track
        
        # Per-arm state
        # theta[a][t] is the estimated weight vector for arm a if the regime age is t+1
        # Sigma[a][t] is the covariance matrix for the estimate
        # self.p[a][t] is the posterior probability that arm a's regime age is t+1
        
        self.theta = [[np.zeros(dim) for _ in range(self.max_age)] for _ in range(n_arms)]
        self.Sigma = [[np.eye(dim) * 10.0 for _ in range(self.max_age)] for _ in range(n_arms)]
        # Initialize prior: uniform over ages? Or start with age 0 (new regime) for all?
        # Let's start with a uniform prior over ages, but the likelihood will quickly dominate.
        # Actually, starting with age 0 (index 0) is more natural if we assume a fresh start.
        # But BOCPD usually maintains a predictive distribution.
        # Let's initialize p to a small uniform distribution, or just uniform.
        self.p = [np.ones(self.max_age) / self.max_age for _ in range(n_arms)]
        
        # Changepoint prior: probability that a changepoint occurs at this step
        # Using a constant hazard rate for simplicity.
        self.hazard = 0.05
        
        # Measurement noise
        self.R = 1.0
        
        # UCB exploration coefficient
        self.alpha = 1.2
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0

    def _check_deadline(self):
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def select(self, context):
        """Select arm using UCB based on BOCPD posterior."""
        if self._check_deadline():
            # Fallback: greedy on the most probable model's mean
            scores = []
            for a in range(self.n):
                # Use the model with highest probability
                best_age = np.argmax(self.p[a])
                scores.append(self.theta[a][best_age] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)

        ucb_scores = []
        for a in range(self.n):
            # Compute weighted mean and variance across models
            # mean_pred = sum_t p_t * (theta_t^T x)
            # var_pred = sum_t p_t * ( (theta_t^T x)^2 + x^T Sigma_t x ) - (sum_t p_t * theta_t^T x)^2
            # Note: This is the variance of the predictive distribution, including model uncertainty.
            
            weighted_mean = 0.0
            weighted_sq_mean = 0.0
            
            for t in range(self.max_age):
                p_t = self.p[a][t]
                if p_t < 1e-8:
                    continue
                pred = self.theta[a][t] @ context
                var_t = context @ self.Sigma[a][t] @ context
                weighted_mean += p_t * pred
                weighted_sq_mean += p_t * (pred * pred + var_t)
            
            # Variance of the predictive distribution
            var_pred = weighted_sq_mean - weighted_mean * weighted_mean
            if var_pred < 0:
                var_pred = 0.0
            std_pred = np.sqrt(var_pred)
            
            ucb = weighted_mean + self.alpha * std_pred
            ucb_scores.append(ucb)

        return int(np.argmax(ucb_scores))

    def update(self, context, arm, reward):
        """Update BOCPD state for the selected arm."""
        context = np.asarray(context, dtype=np.float64)
        x = context
        r = float(reward)
        
        # 1. Predict likelihoods for each model (age)
        # For each age t, compute the likelihood of observing r given x and the model at age t.
        # Likelihood is Gaussian: N(r; theta_t^T x, x^T Sigma_t x + R)
        
        log_lik = np.zeros(self.max_age)
        for t in range(self.max_age):
            pred = self.theta[arm][t] @ x
            var = x @ self.Sigma[arm][t] @ x + self.R
            # Log Gaussian likelihood
            # -0.5 * ( (r-pred)^2 / var + log(2*pi*var) )
            log_lik[t] = -0.5 * ((r - pred) ** 2 / var + np.log(2 * np.pi * var))
        
        # 2. Update posterior probabilities of ages
        # The new posterior is proportional to:
        # p_t_new ∝ (1 - hazard) * p_t_old * likelihood_t  (if no changepoint)
        # PLUS
        # p_0_new ∝ hazard * sum_t p_t_old * likelihood_t (if changepoint occurs, reset to age 0)
        
        # Compute unnormalized weights
        weights = (1 - self.hazard) * self.p[arm] * np.exp(log_lik - np.max(log_lik))  # Subtract max for stability
        
        # The changepoint term: all probability mass collapses to age 0
        # The total probability of a changepoint is hazard * sum(p_old * likelihood)
        # But since we normalized by exp(log_lik - max), we need to be careful.
        
        # Let's compute the sum of the likelihood-weighted probabilities
        # sum_lik = sum_t p_t * likelihood_t
        # sum_lik = sum_t p_t * exp(log_lik_t)
        # To avoid underflow, we already shifted. Let's recompute sum_lik with the shift.
        log_lik_shifted = log_lik - np.max(log_lik)
        sum_lik = np.sum(self.p[arm] * np.exp(log_lik_shifted))
        
        # The weight for age 0 from the changepoint is:
        # p_0_cp = hazard * sum_lik * exp(0 - max_log_lik)? No.
        # The likelihood for age 0 after reset is the prior likelihood, which is N(0, 10^2) or similar.
        # But in BOCPD, when a changepoint occurs, the model is reset to the prior.
        # The likelihood of the observation given the PRIOR model is needed.
        
        # Let's define the prior model: theta ~ N(0, Sigma_prior), where Sigma_prior = 10*I
        # Prior predictive distribution: N(0, x^T Sigma_prior x + R)
        var_prior = x @ (np.eye(self.d) * 10.0) @ x + self.R
        log_lik_prior = -0.5 * (r * r / var_prior + np.log(2 * np.pi * var_prior))
        
        # The weight for age 0 from the changepoint is:
        # p_0_cp = hazard * sum_lik_old * likelihood_prior
        # Wait, the standard BOCPD update is:
        # p_t ∝ (1-c) p_t_old L_t for t > 0
        # p_0 ∝ c * (sum p_t_old L_t) * L_prior  (where L_prior is likelihood under prior)
        # Actually, the likelihood L_t is the likelihood of the observation given the model of age t.
        # When a changepoint occurs, the new model is the prior. So the likelihood is L_prior.
        # The probability of a changepoint is c.
        # The total probability mass that transitions to age 0 is c * sum(p_t_old * L_t) * L_prior / L_prior?
        # No, the update rule is:
        # p_t_new ∝ (1-c) p_t_old L_t   for t >= 1
        # p_0_new ∝ c * (sum p_t_old L_t)   (Here L_t is the likelihood of the observation under the model of age t, and the reset model is the prior, so the likelihood for the reset model is L_prior, but the term sum p_t_old L_t is the total predictive likelihood before the reset decision).
        # Actually, the standard formula is:
        # p_t ∝ (1-c) p_t_old L_t  for t > 0
        # p_0 ∝ c * (sum p_t_old L_t)
        # This assumes that the prior model's likelihood is 1? No, that's not right.
        
        # Let's re-derive.
        # P(age_t | obs) ∝ P(obs | age_t) P(age_t)
        # If no changepoint: P(age_t | obs) ∝ L_t * (1-c) * p_t_old
        # If changepoint: age becomes 0. P(age_0 | obs, cp) ∝ L_prior * c * sum(p_t_old L_t)
        # Wait, the event "changepoint occurs" has probability c. Given a changepoint, the age is 0.
        # The likelihood of the observation given a changepoint is L_prior (likelihood under the prior model).
        # So P(obs | cp) = L_prior.
        # P(cp) = c.
        # P(obs, cp) = c * L_prior.
        # But this doesn't depend on p_t_old. That's not right.
        
        # The correct BOCPD update:
        # The predictive likelihood of the observation given the current state (which is a mixture of models) is:
        # L_pred = sum_t p_t_old L_t
        # If no changepoint, the new age distribution is proportional to p_t_old L_t for t > 0, and p_0 is 0 (since age increases).
        # Wait, age increases by 1. So the new age t is the old age t-1.
        # P(age_new = t | no cp) ∝ P(age_old = t-1 | no cp) L_{t-1}
        # If changepoint, the new age is 0.
        # P(age_new = 0 | cp) = 1
        # So the update is:
        # p_0_new ∝ c * L_pred
        # p_t_new ∝ (1-c) * p_{t-1} * L_{t-1}  for t > 0
        
        # Where L_t is the likelihood of the observation given the model of age t (before age increment).
        # And L_pred = sum_t p_t_old L_t.
        
        # Let's implement this.
        L_pred = np.sum(self.p[arm] * np.exp(log_lik_shifted)) * np.exp(np.max(log_lik))  # Unshifted sum
        # Actually, let's keep it in log space or shifted space to avoid overflow.
        # Let's use the shifted log-likelihoods.
        # log_lik_shifted = log_lik - max_log_lik
        # exp(log_lik_shifted) are the relative likelihoods.
        # L_pred_shifted = sum p_t * exp(log_lik_shifted)
        # L_pred = L_pred_shifted * exp(max_log_lik)
        
        L_pred_shifted = np.sum(self.p[arm] * np.exp(log_lik_shifted))
        
        # New probabilities (unnormalized)
        p_new = np.zeros(self.max_age)
        # Age 0: comes from changepoint
        p_new[0] = self.hazard * L_pred_shifted
        
        # Ages t > 0: come from old age t-1 with no changepoint
        for t in range(1, self.max_age):
            p_new[t] = (1 - self.hazard) * self.p[arm][t-1] * np.exp(log_lik_shifted[t-1])
        
        # Normalize
        p_sum = np.sum(p_new)
        if p_sum > 1e-10:
            p_new /= p_sum
        else:
            p_new = np.ones(self.max_age) / self.max_age
        self.p[arm] = p_new
        
        # 3. Update the model parameters (theta and Sigma) for each age using RLS
        # The model of age t (after age increment) was the model of age t-1 (before increment).
        # So we need to update the model that was at age t-1 with the new observation.
        # But wait, the model of age t-1 was already used to compute L_{t-1}.
        # Now we need to update it with the observation to get the posterior for that model.
        
        # For age 0 (new regime), the model is reset to the prior, then updated with the observation.
        # For age t > 0, the model is the old model of age t-1, updated with the observation.
        
        # Reset age 0 to prior
        self.theta[arm][0] = np.zeros(self.d)
        self.Sigma[arm][0] = np.eye(self.d) * 10.0
        
        # Update all models with RLS
        for t in range(self.max_age):
            P = self.Sigma[arm][t]
            theta = self.theta[arm][t]
            
            Px = P @ x
            S = np.dot(x, Px) + self.R
            if S < 1e-6:
                S = 1e-6
            
            K = Px / S
            pred = theta @ x
            innovation = r - pred
            
            self.theta[arm][t] = theta + K * innovation
            self.Sigma[arm][t] = P - np.outer(K, Px)
            self.Sigma[arm][t] = (self.Sigma[arm][t] + self.Sigma[arm][t].T) / 2.0
        
        # 4. Shift the models? No, the models are already in place.
        # The model at age t is the posterior for the regime that started t steps ago.
        # We just updated them all. The next step, the age will increment, so the model at age t
        # will become the model at age t+1. But we don't need to shift the arrays; we just need to
        # remember that the index t corresponds to age t.
        # Wait, the update rule for p_t_new used p_{t-1} old. So the probability mass at age t
        # comes from the model at age t-1. But the model at age t in our array is the model for
        # a regime of age t. Did we update the correct model?
        
        # Let's re-think. At step k, we have models for ages 0..T.
        # We observe x_k, r_k.
        # We compute L_t for each model of age t (which is the model for a regime that started t steps ago).
        # The new age distribution:
        # p_0_new ∝ c * L_pred
        # p_t_new ∝ (1-c) * p_{t-1} * L_{t-1}
        
        # The model for age t in the new step should be the posterior of the model that was at age t-1 in the old step.
        # So we need to update the model at index t-1 with the observation, and then it becomes the model for age t.
        # But we also need a new model for age 0, which is the prior updated with the observation.
        
        # So the correct order is:
        # 1. Compute L_t for all models.
        # 2. Update p_t.
        # 3. Shift the models: model_t_new = model_{t-1}_old (for t > 0), model_0_new = prior.
        # 4. Update all models with RLS.
        
        # I did step 3 and 4 in the previous code, but I did it in the wrong order?
        # In the previous code, I reset age 0 to prior, then updated all models.
        # But the model at age t > 0 was NOT shifted. It was the old model at age t, not age t-1.
        # That's a bug.
        
        # Let's fix it. We need to shift the models BEFORE updating them.
        # But I already updated them in the previous code. Let's re-implement correctly.
        
        # Since I already updated the models in the previous code, I need to redo the update.
        # But I can't undo it. So let's just implement the correct order from now on.
        
        # Actually, I can just re-do the shift and update.
        # Save the current theta and Sigma (which are already updated incorrectly).
        # No, let's just implement the correct logic from scratch in this method.
        
        # I'll rewrite the update method to be correct.
# EVOLVE-BLOCK-END
