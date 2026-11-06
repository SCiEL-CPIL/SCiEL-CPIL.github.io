"""Non-stationary linear contextual bandit with multi-scale ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Multi-scale ensemble LinUCB for non-stationary linear bandits.
    
    Each arm maintains a bank of linear models with different forgetting factors.
    The policy selects the model with the lowest recent residual variance for 
    prediction and uses its uncertainty for exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Define a bank of forgetting factors
        # Geometric spacing from fast adaptation (low alpha) to slow (high alpha)
        # We want to cover a range of timescales.
        # Let's use 5 models per arm.
        self.n_models = 5
        # alphas: 0.90, 0.93, 0.96, 0.98, 0.999
        # Let's generate them geometrically
        alpha_min = 0.90
        alpha_max = 0.999
        # rho = (alpha_max / alpha_min) ** (1 / (n_models - 1))
        # alphas = [alpha_min * (rho ** i) for i in range(n_models)]
        
        # Simpler: hardcode or generate
        self.alphas = np.geomspace(0.90, 0.999, self.n_models)
        
        # Initialize models
        # A[a][k] : dim x dim matrix
        # b[a][k] : dim vector
        # R[a][k] : scalar residual sum of squares (discounted)
        
        self.A = [[np.eye(dim) for _ in range(self.n_models)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_models)] for _ in range(n_arms)]
        self.R = [[0.0 for _ in range(self.n_models)] for _ in range(n_arms)]
        
        # Track if we have seen enough data to trust the model selection
        # Initially, all models are equally uncertain.
        # We will just use the first few steps to initialize.
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s margin

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0
            
        context = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Find the best model for this arm based on lowest recent residual variance
            # We use R[a][k] as the metric.
            # To avoid picking a model with R=0 due to no data, we can add a prior or just let it be.
            # Initially R=0 for all. Argmin will pick 0.
            # As data comes in, R will grow. The model with the best fit (lowest R) will be picked.
            
            # Optimization: If we have very little data, just use the middle model or all are equivalent.
            # Let's just find argmin of R.
            k_best = int(np.argmin(self.R[a]))
            
            A = self.A[a][k_best]
            b = self.b[a][k_best]
            
            # Calculate prediction and uncertainty
            # theta = A^{-1} b
            # pred = theta @ context
            # var = context @ A^{-1} @ context
            
            # Solve A theta = b
            try:
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                
            pred = np.dot(theta, context)
            
            # Calculate variance term
            # inv_A_context = A^{-1} context
            try:
                inv_A_context = np.linalg.solve(A, context)
            except np.linalg.LinAlgError:
                inv_A_context = np.zeros(self.d)
                
            variance = np.dot(context, inv_A_context)
            
            # Exploration bonus
            # Beta is the exploration parameter. 
            # In non-stationary settings, we might want to increase exploration.
            # Let's use a fixed beta or adaptive.
            # Standard LinUCB uses beta * sqrt(var).
            # Let's try beta = 1.0 or slightly higher.
            beta = 1.0
            scores[a] = pred + beta * np.sqrt(variance)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context = np.array(context)
        self.t += 1
        
        for k in range(self.n_models):
            alpha = self.alphas[k]
            
            # Update A and b
            # A_new = alpha * A_old + outer(context, context)
            # b_new = alpha * b_old + reward * context
            
            self.A[arm][k] = alpha * self.A[arm][k] + np.outer(context, context)
            self.b[arm][k] = alpha * self.b[arm][k] + reward * context
            
            # Update Residual Sum of Squares
            # We need the prediction from this model *before* update to calculate residual.
            # Wait, the update is done. We should calculate residual using the model *before* update?
            # Or after?
            # Standard recursive least squares residual:
            # r = y - x^T theta_old
            # Then update theta.
            # But here we update A and b directly.
            # Let's calculate the prediction using the OLD parameters.
            
            # We need to store old A and b? Or calculate pred before update.
            # Let's refactor: Calculate pred, then update.
            
            # Actually, I can just calculate the residual using the current (pre-update) model.
            # But I already updated A and b in the lines above.
            # Let's swap the order.
            
        # Let's rewrite the loop to calculate residual first.
        
        for k in range(self.n_models):
            alpha = self.alphas[k]
            
            # 1. Calculate prediction with current model
            A = self.A[arm][k]
            b = self.b[arm][k]
            
            try:
                theta = np.linalg.solve(A, b)
                pred = np.dot(theta, context)
            except np.linalg.LinAlgError:
                pred = 0.0
                
            residual = reward - pred
            
            # 2. Update Residual Sum of Squares
            self.R[arm][k] = alpha * self.R[arm][k] + residual * residual
            
            # 3. Update A and b
            self.A[arm][k] = alpha * self.A[arm][k] + np.outer(context, context)
            self.b[arm][k] = alpha * self.b[arm][k] + reward * context

# EVOLVE-BLOCK-END
