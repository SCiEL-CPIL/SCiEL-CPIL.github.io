"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Multi-scale ensemble policy for non-stationary linear contextual bandits.
    
    Instead of a single model with an adaptive forgetting factor, this policy
    maintains a bank of linear models for each arm, each with a fixed, geometrically
    spaced forgetting factor (beta). 
    
    The selection process involves:
    1. For each arm, compute the prediction from the ensemble member that has
       the lowest recent residual variance (best fitting model for current dynamics).
    2. Add a UCB bonus based on the uncertainty of that specific model.
    3. Select the arm with the highest score.
    
    This allows the policy to implicitly switch memory horizons: if the environment
    changes rapidly, models with shorter memory (lower beta) will fit better; if
    stable, models with longer memory (higher beta) will fit better.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        
        # Define the ensemble of forgetting factors
        # Geometric spacing from slow forgetting (0.99) to fast forgetting (0.85)
        # 5 models per arm
        self.betas = [0.99, 0.96, 0.93, 0.90, 0.85]
        self.n_models = len(self.betas)
        
        # Initialize A and b for each arm and each model in the ensemble
        # A[arm][model_idx], b[arm][model_idx]
        self.A = [[np.eye(dim) * self.lam for _ in range(self.n_models)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_models)] for _ in range(n_arms)]
        
        # Track recent residuals for each arm and model to determine which model is best
        # residuals[arm][model_idx] = list of recent squared errors
        self.residuals = [[[] for _ in range(self.n_models)] for _ in range(n_arms)]
        self.res_window = 20
        
        # Exploration parameter
        self.alpha = 1.2
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

    def _get_best_model_index(self, arm):
        """
        Determine which model in the ensemble has the lowest recent residual variance.
        """
        best_idx = 0
        min_var = float('inf')
        
        for m in range(self.n_models):
            res_list = self.residuals[arm][m]
            if len(res_list) < 5:
                # If not enough data, prefer the middle model (moderate forgetting)
                # Or just keep default, but let's say we prefer slower forgetting initially
                # Actually, if no data, variance is 0, so we might pick the first one.
                # Let's add a small penalty for low data count to encourage exploration of models?
                # No, let's just use the variance. If empty, var=0.
                # To break ties, we can prefer the model with more data or a specific index.
                # Let's just compute var if possible.
                if len(res_list) == 0:
                    var = 1.0 # Default high variance to avoid picking untrained models
                else:
                    var = np.var(res_list)
            else:
                var = np.var(res_list)
            
            # Add a tiny regularization to avoid picking a model with 0 variance due to insufficient data
            # if it's the only one with data? No, if var is 0 it's good.
            # But if len < 5, we set var=1.0 above.
            
            if var < min_var:
                min_var = var
                best_idx = m
                
        return best_idx

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback: pick arm with highest mean prediction from the "best" model
            scores = []
            for a in range(self.n):
                best_m = self._get_best_model_index(a)
                try:
                    theta = np.linalg.solve(self.A[a][best_m], self.b[a][best_m])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Find the best model for this arm based on recent residuals
            best_m = self._get_best_model_index(a)
            
            try:
                A = self.A[a][best_m]
                b = self.b[a][best_m]
                
                # Solve A * theta = b
                theta = np.linalg.solve(A, b)
                
                # Prediction
                mu = theta @ context
                
                # Confidence interval
                # var = context^T A^{-1} context
                inv_A_ctx = np.linalg.solve(A, context)
                var = context @ inv_A_ctx
                if var < 0:
                    var = 0.0
                uc = self.alpha * np.sqrt(var)
                
                scores.append(mu + uc)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Check deadline
        if time.time() > self.deadline:
            # Minimal update to avoid crash, but skip heavy residual tracking
            for m in range(self.n_models):
                self.A[arm][m] = self.betas[m] * self.A[arm][m]
                self.b[arm][m] = self.betas[m] * self.b[arm][m]
                self.A[arm][m] += np.outer(context, context)
                self.b[arm][m] += reward * context
            return

        # Update all models in the ensemble for the selected arm
        for m in range(self.n_models):
            beta = self.betas[m]
            
            # Forget old data
            self.A[arm][m] = beta * self.A[arm][m]
            self.b[arm][m] = beta * self.b[arm][m]
            
            # Add new observation
            self.A[arm][m] += np.outer(context, context)
            self.b[arm][m] += reward * context
            
            # Calculate residual for this model
            # We need the prediction BEFORE updating with the new point to get the true residual
            # However, we already updated A and b. 
            # To get the residual, we should have calculated the prediction using the OLD A and b.
            # Let's restructure: calculate prediction first, then update.
            
            # Wait, I updated A and b above. I need to calculate residual using the state BEFORE update.
            # Let's swap the order in the loop.
            pass

        # Let's rewrite the update loop correctly to capture residuals.
        # We need to predict using the current (pre-update) model, calculate error, then update.
        
        # Re-doing the update logic properly
        for m in range(self.n_models):
            beta = self.betas[m]
            A_old = self.A[arm][m]
            b_old = self.b[arm][m]
            
            try:
                theta_old = np.linalg.solve(A_old, b_old)
                pred = theta_old @ context
                residual = reward - pred
                residual_sq = residual * residual
            except:
                residual_sq = 0.0
            
            # Update model
            self.A[arm][m] = beta * A_old
            self.b[arm][m] = beta * b_old
            self.A[arm][m] += np.outer(context, context)
            self.b[arm][m] += reward * context
            
            # Update residual history
            self.residuals[arm][m].append(residual_sq)
            if len(self.residuals[arm][m]) > self.res_window:
                self.residuals[arm][m].pop(0)
# EVOLVE-BLOCK-END
