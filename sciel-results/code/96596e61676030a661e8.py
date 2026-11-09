# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# We need to handle the cost-aware acquisition.
# The task states:
# - Minimize three hump camel 2d on [-5, 5]^2.
# - mu, sigma, f_best, cost are passed in.
# - cost has shape (2000,).
# - mu and sigma have shape (2000,).
# - Return 2000 finite scores (higher = evaluate next).
# - The evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one.
# - The cost of evaluating x is 1 + mean(x scaled to [0, 1]).
# - The acquisition function should account for the cost.

# The current program uses Expected Improvement (EI) divided by cost.
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)
# where z = (f_best - mu) / sigma.

# Let's try to improve this.
# One common cost-aware acquisition is the "Expected Improvement per unit cost" (EI/cost), which is what the current program does.
# Another is "Probability of Improvement" (PI) per unit cost.
# PI = Phi(z).

# Let's consider the structure of the problem.
# The three hump camel function is:
# f(x) = 2*x1^2 - 1.05*x1^4 + x1^6/6 + x1*x2 + x2^2
# It has a global minimum at (0,0) with value 0.
# The domain is [-5, 5]^2.

# The GP is Matern-5/2 with length scale 0.2 on coordinates scaled to [0, 1].
# This means the GP is quite local (length scale 0.2 in [0,1] scale is 1.0 in original scale).

# The cost is 1 + mean(x_scaled).
# x_scaled = (x + 5) / 10.
# So cost = 1 + (x1_scaled + x2_scaled) / 2.
# cost = 1 + (x1 + x2 + 10) / 20 = 1.5 + (x1 + x2) / 20.
# The cost ranges from 1.5 + (-10)/20 = 1.0 to 1.5 + 10/20 = 2.0.
# So the cost varies by a factor of 2.

# The current program divides EI by cost. This is a reasonable approach.
# However, we might want to be more aggressive in exploring areas with high uncertainty (high sigma) even if the expected improvement is moderate, especially in the early stages.

# Let's try a different approach: use a combination of EI and a bonus for uncertainty.
# Or, we can use the "Upper Confidence Bound" (UCB) acquisition function, which is:
# UCB = mu + kappa * sigma
# where kappa is a parameter that controls the exploration-exploitation trade-off.
# But UCB is for minimization, so we would use:
# UCB = mu - kappa * sigma
# And then divide by cost.

# However, the task specifies that mu and sigma are in objective units, and the GP is standardized.
# So sigma is the standard deviation of the GP prediction.

# Let's try a hybrid approach:
# score = (EI + lambda * sigma) / cost
# where lambda is a small constant to encourage exploration.

# Alternatively, we can use the "Expected Improvement" with a penalty for high cost.
# The current program already does this by dividing by cost.

# Let's consider the possibility that the current program is already quite good.
# The score is 1.0000, which is the baseline.
# To improve, we need to find the minimum faster or more accurately.

# One idea is to use a more sophisticated acquisition function that takes into account the number of evaluations so far.
# However, the function signature does not include the number of evaluations.
# But we can infer it from the cost.
# The initial cost is 5 points * (1 + mean(x_scaled)).
# The mean of x_scaled for uniform points in [0,1] is 0.5.
# So the initial cost is approximately 5 * (1 + 0.5) = 7.5.
# The total cost budget is 30.
# So we have about 30 - 7.5 = 22.5 cost units left.
# Each evaluation costs between 1.0 and 2.0.
# So we can make about 11-22 more evaluations.

# This is a small number of evaluations, so we need to be efficient.

# Let's try to use a more robust version of EI that handles the case where sigma is very small.
# The current program uses (sigma + 1e-9) to avoid division by zero.
# This is fine.

# Another idea is to use the "Probability of Improvement" (PI) instead of EI.
# PI = Phi(z)
# This is less sensitive to the magnitude of the improvement and might be more robust in some cases.

# Let's try a combination of EI and PI:
# score = (alpha * EI + (1 - alpha) * PI * sigma) / cost
# where alpha is a weight between 0 and 1.

# Or, we can use the "Expected Improvement" with a different formulation.
# The standard EI is:
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)
# This is correct for minimization.

# Let's try to use a "look-ahead" approach where we consider the expected improvement after the next evaluation.
# This is more complex and might not be worth the computational cost.

# Given the time constraint (330s), we need to keep the acquisition function fast.
# The current program is very fast.

# Let's try a simple modification: add a small exploration bonus based on sigma.
# score = (EI + 0.1 * sigma) / cost

# This encourages exploring areas with high uncertainty, which might help find the global minimum faster.

# Let's implement this and see if it improves the score.

# Also, we need to make sure that the scores are finite.
# If sigma is very large, EI might be very large, but that's fine.
# If cost is very small, the score might be very large, but that's also fine.
# We just need to avoid NaNs and infs.

# Let's write the new acquisition function.

def acquisition(mu, sigma, f_best, cost):
    # Ensure cost is not zero or negative.
    cost = np.maximum(cost, 1e-9)
    
    # Calculate z.
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Calculate Expected Improvement.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus.
    exploration_bonus = 0.1 * sigma
    
    # Calculate the score.
    score = (ei + exploration_bonus) / cost
    
    # Ensure the scores are finite.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
