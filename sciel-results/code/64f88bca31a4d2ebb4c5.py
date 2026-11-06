"""Multi-scale ensemble strategy for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Multi-scale ensemble policy.
    
    Each arm maintains a bank of linear models with different forgetting factors (memory horizons).
    The policy selects the arm with the highest predicted reward from the model in the bank
    that has shown the best recent predictive performance (lowest residual variance).
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Geometrically spaced forgetting factors
        # beta close to 1 = long memory (slow adaptation)
        # beta close to 0 = short memory (fast adaptation)
        # We use 5 models per arm
        self.n_models = 5
        self.betas = [0.99, 0.95, 0.90, 0.80, 0.70]
        
        # Per-arm, per-model state: A matrix, b vector
        # A is (dim x dim), b is (dim,)
        self.A = []
        self.b = []
        
        # Residual tracking for model selection
        # residuals[a][m] is a list of recent squared errors for model m of arm a
        self.residuals = []
        self.residual_window = 10
        
        for a in range(n_arms):
            A_arm = []
            b_arm = []
            res_arm = []
            for m in range(self.n_models):
                # Initialize with small ridge for stability
                A_arm.append(np.eye(dim) * 1.0)
                b_arm.append(np.zeros(dim))
                res_arm.append([])
            self.A.append(A_arm)
            self.b.append(b_arm)
            self.residuals.append(res_arm)
            
    def _get_prediction(self, arm, model_idx, context):
        """Get prediction from a specific model for a specific arm."""
        try:
            theta = np.linalg.solve(self.A[arm][model_idx], self.b[arm][model_idx])
            return float(theta @ context)
        except np.linalg.LinAlgError:
            return 0.0

    def _get_model_variance(self, arm, model_idx, context):
        """Get variance estimate for a specific model."""
        try:
            inv_A_ctx = np.linalg.solve(self.A[arm][model_idx], context)
            var = float(context @ inv_A_ctx)
            if var < 0:
                var = 0.0
            return var
        except np.linalg.LinAlgError:
            return 1.0

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: pick arm with best prediction from the most responsive model (last one)
            scores = []
            for a in range(self.n):
                scores.append(self._get_prediction(a, self.n_models - 1, context))
            return int(np.argmax(scores))

        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n):
            # For each arm, find the model with the best recent performance
            # We define "best performance" as lowest mean squared residual recently
            # If no residuals, we fall back to the model with highest uncertainty reduction?
            # Actually, simpler: pick the model that has the lowest recent residual variance.
            # If history is short, we might prefer the most responsive one to learn fast?
            # Let's pick the model with the lowest average recent squared error.
            
            model_scores = []
            for m in range(self.n_models):
                res_list = self.residuals[a][m]
                if len(res_list) > 0:
                    # Use the last few residuals
                    recent_res = res_list[-self.residual_window:]
                    avg_sq_err = np.mean(recent_res)
                else:
                    # No history: assume high error to encourage exploration of all models?
                    # Or assume 0? If we assume 0, we pick the first model (longest memory) initially.
                    # Let's assume a large value so that if we have data, we use it.
                    # But for the very first step, we need to pick something.
                    # Let's prioritize the most responsive model (lowest beta) when no data is available
                    # to get quick feedback.
                    avg_sq_err = 1e6 - self.betas[m] # Lower beta -> higher priority initially? No, lower beta is more responsive.
                    # Let's just say if no data, we use the most responsive model (index n_models-1)
                    # by giving it a low score (good) and others high.
                    if m == self.n_models - 1:
                        avg_sq_err = 0.0
                    else:
                        avg_sq_err = 1.0

                # Get prediction and variance for this model
                pred = self._get_prediction(a, m, context)
                var = self._get_model_variance(a, m, context)
                
                # Combine prediction and uncertainty?
                # The directive says: "selects the arm that maximizes the reward prediction 
                # from the ensemble member with the lowest recent residual variance"
                # This implies we first select the *model* based on residual variance, 
                # then use that model's prediction (plus maybe uncertainty?) for the arm score.
                
                # Let's stick to the directive strictly first: 
                # 1. Identify best model for arm 'a' based on recent residual variance.
                # 2. Use that model's prediction for the arm's score.
                # We will accumulate the "best model index" for this arm.
                
                model_scores.append((avg_sq_err, m, pred, var))
            
            # Find the model with the lowest avg_sq_err
            # If tie, pick the one with higher variance (more uncertainty) to encourage exploration?
            # Or just the first one found.
            best_model_info = min(model_scores, key=lambda x: x[0])
            best_m = best_model_info[1]
            best_pred = best_model_info[2]
            best_var = best_model_info[3]
            
            # Score the arm using the best model's prediction
            # Should we add exploration bonus?
            # The directive doesn't explicitly mention UCB, but "maximizes the reward prediction".
            # However, pure exploitation is bad for bandits. 
            # Let's add a small exploration term based on the variance of the selected model.
            # alpha = 1.0
            score = best_pred + 1.0 * np.sqrt(best_var)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        self.t += 1
        
        # Deadline guard for minimal update
        if time.time() > self.deadline:
            # Just update the most responsive model to keep some learning
            m = self.n_models - 1
            beta = self.betas[m]
            self.A[arm][m] = beta * self.A[arm][m]
            self.b[arm][m] = beta * self.b[arm][m]
            self.A[arm][m] += np.outer(context, context)
            self.b[arm][m] += reward * context
            return

        # Update all models for the selected arm
        for m in range(self.n_models):
            beta = self.betas[m]
            
            # Get prediction before update to calculate residual
            pred = self._get_prediction(arm, m, context)
            residual_sq = (reward - pred) ** 2
            
            # Store residual
            res_list = self.residuals[arm][m]
            res_list.append(residual_sq)
            if len(res_list) > self.residual_window * 2: # Keep a bit more than needed
                res_list.pop(0)
            
            # Update model parameters
            self.A[arm][m] = beta * self.A[arm][m]
            self.b[arm][m] = beta * self.b[arm][m]
            self.A[arm][m] += np.outer(context, context)
            self.b[arm][m] += reward * context

# EVOLVE-BLOCK-END
