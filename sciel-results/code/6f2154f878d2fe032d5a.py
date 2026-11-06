"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear model with arm-specific bias terms for non-stationary linear bandits.
    
    Structure:
    - A single global ridge regression model A_global, b_global for shared context features
    - Per-arm bias terms c_arm that adapt to non-stationarity
    - Forgetting factor for non-stationary adaptation
    - UCB exploration based on global model uncertainty
    
    This reduces parameter count vs per-arm models and leverages cross-arm correlations.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global linear model
        self.A = np.eye(dim)  # Regularized covariance
        self.b = np.zeros(dim)  # Weight vector
        
        # Arm-specific bias terms (scalar per arm)
        self.arm_biases = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)  # For bias estimation
        
        # Forgetting factor for non-stationarity
        self.forgetting = 0.99
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Track moving average of rewards per arm for bias estimation
        self.arm_reward_sums = np.zeros(n_arms)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute global model prediction
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b
        global_pred = theta @ context
        
        # Compute uncertainty
        variance = context @ inv_A @ context
        exploration_bonus = self.alpha * np.sqrt(variance)
        
        # Score each arm: global prediction + arm bias + exploration
        scores = []
        for a in range(self.n):
            # Arm bias: use running average if we have data, else 0
            if self.arm_counts[a] > 0:
                bias = self.arm_biases[a]
            else:
                bias = 0.0
            
            score = global_pred + bias + exploration_bonus
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Update global model with forgetting
        self.A = self.forgetting * self.A + np.outer(context, context)
        self.b = self.forgetting * self.b + reward * context
        
        # Update arm-specific bias
        # Estimate bias as: reward - global_model_prediction
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b
        global_pred = theta @ context
        residual = reward - global_pred
        
        # Exponentially weighted moving average of residual for this arm
        if self.arm_counts[arm] == 0:
            self.arm_biases[arm] = residual
        else:
            # Weighted update with forgetting
            self.arm_biases[arm] = self.forgetting * self.arm_biases[arm] + (1 - self.forgetting) * residual
        
        self.arm_counts[arm] += 1
        self.arm_reward_sums[arm] = self.forgetting * self.arm_reward_sums[arm] + (1 - self.forgetting) * reward
# EVOLVE-BLOCK-END
