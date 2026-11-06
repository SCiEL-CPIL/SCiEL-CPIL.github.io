# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using a multi-scale ensemble of linear models.
    
    Each arm maintains a bank of regressors with fixed, geometrically spaced forgetting factors.
    The policy selects the arm that maximizes the reward prediction from the ensemble member
    with the lowest recent residual variance, effectively using model selection to dynamically
    switch memory horizons without explicit regime detection.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Ridge regularization / Prior precision scaling
        
        # Number of models per arm in the ensemble
        self.n_models = 5
        
        # Geometrically spaced forgetting factors
        # beta_min is fast forgetting, beta_max is slow forgetting
        self.betas = np.geomspace(0.70, 0.98, self.n_models)
        
        # Per-arm, per-model parameters
        # A[arm][model] = precision matrix
        # b[arm][model] = A @ mu
        # residual_ema[arm][model] = EMA of squared residuals for model selection
        self.A = [[np.eye(dim) * self.lam for _ in range(self.n_models)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_models)] for _ in range(n_arms)]
        self.residual_ema = [[0.0 for _ in range(self.n_models)] for _ in range(n_arms)]
        
        # EMA decay for residual tracking (model selection)
        self.ema_decay = 0.95
        
        # Step counter
        self.step = 0
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin for 120s limit

    def _safe(self):
        return time.time() >= self.deadline

    def _get_best_model_prediction(self, arm, context):
        """
        Get the prediction from the ensemble member with the lowest recent residual variance.
        Returns (mean_score, var_score) from the best model.
        """
        best_model = 0
        best_residual = float('inf')
        
        for m in range(self.n_models):
            if self.residual_ema[arm][m] < best_residual:
                best_residual = self.residual_ema[arm][m]
                best_model = m
        
        # Get mean and variance from the best model
        try:
            mu = np.linalg.solve(self.A[arm][best_model], self.b[arm][best_model])
            mean_score = float(mu @ context)
            
            v = np.linalg.solve(self.A[arm][best_model], context)
            var_score = float(context @ v)
            
            return mean_score, max(var_score, 0.0)
        except Exception:
            return 0.0, 1.0

    def select(self, context):
        if self._safe():
            # Emergency: pick arm with highest mean prediction from best model
            scores = []
            for a in range(self.n):
                mean_score, _ = self._get_best_model_prediction(a, context)
                scores.append(mean_score)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mean_score, var_score = self._get_best_model_prediction(a, context)
            # Thompson sampling: add exploration noise
            z = np.random.randn()
            score = mean_score + np.sqrt(var_score) * z
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.step += 1
        
        # Update all models for the selected arm
        for m in range(self.n_models):
            beta = self.betas[m]
            
            # Compute residual BEFORE updating
            try:
                mu = np.linalg.solve(self.A[arm][m], self.b[arm][m])
                pred = float(mu @ context)
                residual = reward - pred
            except Exception:
                residual = 0.0
            
            # Update residual EMA for model selection
            self.residual_ema[arm][m] = self.ema_decay * self.residual_ema[arm][m] + (1 - self.ema_decay) * (residual ** 2)
            
            # Update the model using Bayesian update with decay
            self.A[arm][m] = beta * self.A[arm][m] + np.outer(context, context)
            self.b[arm][m] = beta * self.b[arm][m] + reward * context

# EVOLVE-BLOCK-END
