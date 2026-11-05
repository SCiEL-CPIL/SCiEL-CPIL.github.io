"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Multi-scale ensemble of linear models for non-stationary linear contextual bandit.
    
    Each arm maintains a bank of regressors with fixed, geometrically spaced forgetting factors.
    The policy selects the arm that maximizes the reward prediction from the ensemble member
    with the lowest recent residual variance, effectively using model selection to dynamically
    switch memory horizons without explicit regime detection.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        
        # Number of models in the ensemble (geometrically spaced forgetting factors)
        self.n_models = 5
        # Forgetting factors: geometrically spaced from short memory to long memory
        # beta values: 0.85, 0.90, 0.93, 0.96, 0.98
        self.betas = np.array([0.85, 0.90, 0.93, 0.96, 0.98])
        
        # For each arm, maintain a bank of (A, b) pairs, one per model
        # A[m] = np.eye(dim) * lam, b[m] = zeros(dim)
        self.A = [[np.eye(dim) * self.lam for _ in range(self.n_models)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_models)] for _ in range(n_arms)]
        
        # Track recent residuals for each arm-model pair to compute residual variance
        # residuals[arm][model] = list of recent squared residuals
        self.residuals = [[[] for _ in range(self.n_models)] for _ in range(n_arms)]
        self.residual_window = 30
        
        # Time tracking for deadline guard
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin
        
        # Random seed for any stochastic components
        self.rng = np.random.default_rng(42)
        
        # Exploration parameter for UCB-like bonus
        self.alpha = 1.0

    def _get_residual_variance(self, arm, model):
        """Compute the mean squared residual for a given arm-model pair."""
        res_list = self.residuals[arm][model]
        if len(res_list) < 2:
            return float('inf')  # Unknown variance, penalize
        return np.mean(res_list)

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback: pick the arm with the highest mean estimate from the best model
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                best_model_score = -np.inf
                for m in range(self.n_models):
                    try:
                        theta = np.linalg.solve(self.A[a][m], self.b[a][m])
                        mean = theta @ context
                        if mean > best_model_score:
                            best_model_score = mean
                    except:
                        pass
                if best_model_score > best_score:
                    best_score = best_model_score
                    best_arm = a
            return best_arm

        context = np.array(context, dtype=float)
        
        scores = []
        for a in range(self.n):
            best_arm_score = -np.inf
            for m in range(self.n_models):
                try:
                    # Compute theta_hat = A^{-1} b
                    A_reg = self.A[a][m] + 1e-8 * np.eye(self.d)
                    theta_hat = np.linalg.solve(A_reg, self.b[a][m])
                    
                    # Compute the UCB bonus using the diagonal of A^{-1}
                    # For efficiency, we can estimate the variance from the diagonal
                    # Var(theta_hat) = A^{-1}, so the confidence interval width is
                    # alpha * sqrt(x^T A^{-1} x)
                    # We approximate this by computing A^{-1} x
                    A_inv_x = np.linalg.solve(A_reg, context)
                    variance = context @ A_inv_x
                    if variance < 0:
                        variance = 0.0
                    ucb_bonus = self.alpha * np.sqrt(variance)
                    
                    # Get residual variance for this model
                    res_var = self._get_residual_variance(a, m)
                    
                    # Score = mean + UCB bonus - penalty for high residual variance
                    # We use a weighted combination: prefer models with low residual variance
                    # The residual variance acts as a confidence weight
                    if res_var == float('inf'):
                        weight = 0.5  # Low confidence if no data
                    else:
                        # Inverse of residual variance as weight, normalized
                        weight = 1.0 / (1.0 + res_var)
                    
                    # Combine mean prediction and UCB bonus, weighted by model confidence
                    score = weight * (theta_hat @ context + ucb_bonus)
                    
                    if score > best_arm_score:
                        best_arm_score = score
                        
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue
            
            scores.append(best_arm_score)
        
        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.array(context, dtype=float)
        
        # Update each model in the ensemble for this arm
        for m in range(self.n_models):
            beta = self.betas[m]
            
            # Compute prediction from this model before update
            try:
                A_reg = self.A[arm][m] + 1e-8 * np.eye(self.d)
                theta_hat = np.linalg.solve(A_reg, self.b[arm][m])
                prediction = theta_hat @ context
            except:
                prediction = 0.0
            
            # Compute residual
            residual = reward - prediction
            residual_sq = residual * residual
            
            # Update residual history
            self.residuals[arm][m].append(residual_sq)
            if len(self.residuals[arm][m]) > self.residual_window:
                self.residuals[arm][m].pop(0)
            
            # Update A and b with forgetting
            self.A[arm][m] = beta * self.A[arm][m] + np.outer(context, context)
            self.b[arm][m] = beta * self.b[arm][m] + reward * context
# EVOLVE-BLOCK-END
