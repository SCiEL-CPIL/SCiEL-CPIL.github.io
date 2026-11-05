"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Recursive Least Squares (RLS) Linear Bandit with Forgetting Factor."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.98      # Forgetting factor (discount)
        self.lambda_reg = 1.0  # Initial regularization / scale for inverse
        self.beta = 1.0        # Exploration parameter (UCB width)
        
        # Initialize state for each arm
        # A_inv[a] is the inverse of the weighted sum of outer products of contexts
        # theta[a] is the weight vector estimate
        self.A_inv = [self.lambda_reg * np.eye(self.d) for _ in range(self.n)]
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        self.t = 0

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            pred = self.theta[a] @ context
            # Variance term for UCB
            var = context @ self.A_inv[a] @ context
            # Clamp to avoid negative due to numerical errors
            var = max(var, 0.0)
            score = pred + self.beta * np.sqrt(var)
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # Recursive Least Squares update with forgetting factor
        # A_new = gamma * A_old + x x^T
        # A_inv_new = (A_old_inv^{-1} + x x^T / gamma)^{-1} 
        # Using Sherman-Morrison-Woodbury:
        # (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here, we want to update A_inv for A_new = gamma * A_old + x x^T
        # Let A_old_inv = self.A_inv[arm]
        # We are effectively computing (gamma * A_old + x x^T)^{-1}
        # = (gamma * (A_old_inv^{-1}) + x x^T)^{-1}
        # = 1/gamma * (A_old_inv^{-1} + x x^T / gamma)^{-1}
        # This is slightly tricky to do directly on A_inv without inverting.
        # Alternative formulation:
        # Let P_k = A_k^{-1}.
        # A_{k+1} = gamma A_k + x_{k+1} x_{k+1}^T
        # P_{k+1} = (gamma A_k + x x^T)^{-1}
        # Using SMW on (gamma A_k + x x^T):
        # Let M = gamma A_k. M^{-1} = (1/gamma) P_k.
        # P_{k+1} = (M + x x^T)^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        # P_{k+1} = (1/gamma)P_k - (1/gamma)P_k x x^T (1/gamma)P_k / (1 + x^T (1/gamma)P_k x)
        # P_{k+1} = (1/gamma)P_k - (1/gamma^2) P_k x x^T P_k / (1 + (1/gamma) x^T P_k x)
        
        P = self.A_inv[arm]
        x = context
        r = reward
        
        # Calculate scalar terms
        Px = P @ x
        xP = x @ P  # same as Px.T
        denom = 1.0 + (x @ Px) / self.gamma
        
        # Update P (A_inv)
        # P_new = (1/gamma) * P - (1/gamma^2) * outer(Px, Px) / denom
        self.A_inv[arm] = (1.0 / self.gamma) * P - (1.0 / (self.gamma ** 2)) * np.outer(Px, Px) / denom
        
        # Update theta
        # theta_{k+1} = theta_k + P_{k+1} x (r - theta_k^T x)
        # Note: P_{k+1} is the inverse of the new A.
        # Standard RLS update: theta_new = theta_old + K * (r - x^T theta_old)
        # where K = P_new * x
        # Let's verify the gain.
        # In standard RLS with discount gamma, the gain K is often defined as:
        # K = P_new * x
        # Let's use the updated P.
        P_new = self.A_inv[arm]
        K = P_new @ x
        pred = self.theta[arm] @ x
        error = r - pred
        self.theta[arm] = self.theta[arm] + K * error
        
        self.t += 1
# EVOLVE-BLOCK-END
