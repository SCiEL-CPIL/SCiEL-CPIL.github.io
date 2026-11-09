"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Linear UCB with Exponential Discounting for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Decay factor: 0.95 means recent data is weighted ~20x more than data 10 steps ago
        self.gamma = 0.95
        # Exploration parameter
        self.beta = 1.5
        # Initialize precision matrices A (inverse of covariance) and reward vectors b
        # Start with identity to provide prior regularization
        self.A = [np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        # Maintain the inverse of A for efficient UCB calculation
        # V = A^{-1}
        self.V = [np.eye(self.d) for _ in range(self.n)]
        
        # Initialize step counter for potential adaptive beta if needed, 
        # but fixed beta is standard for this setup.
        self.t = 0

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Mean estimate: mu = V * b
            mu = self.V[a] @ self.b[a]
            # Predicted reward: mu^T * context
            pred_reward = np.dot(mu, context)
            
            # UCB Bonus: beta * sqrt(context^T V context)
            # V is the covariance matrix of the estimate
            var = np.dot(context, self.V[a] @ context)
            # Ensure non-negative due to numerical errors
            if var < 0:
                var = 0.0
            bonus = self.beta * np.sqrt(var)
            
            scores.append(pred_reward + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # 1. Decay existing statistics
        # A_new = gamma * A_old
        # V_new = (gamma * A_old)^-1 = (1/gamma) * A_old^-1 = (1/gamma) * V_old
        self.V[arm] = self.V[arm] / self.gamma
        self.b[arm] = self.b[arm] * self.gamma
        
        # 2. Update with new observation using Woodbury Identity
        # We want to update V = (gamma*A + x*x^T)^-1
        # Let A_curr = gamma * A_old. Then V_curr = (1/gamma) * V_old.
        # We are adding x*x^T to A_curr.
        # New V = (V_curr^{-1} + x*x^T)^-1
        # Using Woodbury: (M + u v^T)^-1 = M^-1 - M^-1 u (1 + v^T M^-1 u)^-1 v^T M^-1
        # Here M = V_curr^{-1} = gamma * A_old. But we don't have A_old directly, we have V_curr.
        # It is easier to think in terms of A.
        # A_updated = gamma * A_old + x x^T
        # We maintain V = A^{-1}.
        # Let V_old be the inverse of A_old.
        # V_decayed = (1/gamma) * V_old. This is the inverse of (gamma * A_old).
        # Now we add x x^T to the precision matrix (gamma * A_old).
        # New Precision = (gamma * A_old) + x x^T.
        # New Inverse V_new = ( (gamma * A_old) + x x^T )^{-1}.
        # Let M = gamma * A_old. Then M^{-1} = V_decayed.
        # V_new = (M + x x^T)^{-1} = M^{-1} - M^{-1} x (1 + x^T M^{-1} x)^{-1} x^T M^{-1}
        
        V_curr = self.V[arm] # This is M^{-1}
        
        # Compute scalar term: 1 + x^T V_curr x
        x = context
        v = V_curr @ x
        denom = 1.0 + np.dot(x, v)
        
        # Update V
        # V_new = V_curr - (V_curr x x^T V_curr) / denom
        # Note: V_curr x is v. So V_curr x x^T V_curr = v (x^T V_curr) = v v^T? No.
        # v = V_curr x.
        # V_curr x x^T V_curr = v (x^T V_curr) = v (V_curr x)^T = v v^T.
        # So V_new = V_curr - (v v^T) / denom
        
        self.V[arm] = V_curr - (np.outer(v, v) / denom)
        
        # Update b
        # b_updated = gamma * b_old + reward * x
        self.b[arm] = self.b[arm] + reward * x
# EVOLVE-BLOCK-END
