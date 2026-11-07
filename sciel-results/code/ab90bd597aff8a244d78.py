"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to be careful about time. The evaluator kills after 330s.
# The function is called repeatedly. We should add a deadline guard.
# However, the function signature is fixed: acquisition(mu, sigma, f_best, cost).
# It returns 2000 scores.
# The cost of evaluating a point is 1 + mean(x_scaled).
# We want to minimize the function.
# The seed program uses Expected Improvement (EI) divided by cost.
# EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
# This is standard for minimization.
# Dividing by cost makes it cost-aware.
#
# Let's think about potential improvements:
# 1. The current EI might be too greedy. Maybe a more exploration-heavy approach?
# 2. The cost model is 1 + mean(x_scaled). Points near the center (x=0 in [-1,1], so x_scaled=0.5) cost 1.5.
#    Points at the edges (x=1, x_scaled=1) cost 2.0. Points at x=-1 (x_scaled=0) cost 1.0.
#    So points near x=-1 are cheaper.
# 3. The function is cosine8 8d. The minimum is -0.8.
# 4. The instances have shifted minimizers.
#
# Let's try a different acquisition function: Probability of Improvement (PI) or Upper Confidence Bound (UCB).
# UCB: mu - kappa * sigma. This balances exploitation (low mu) and exploration (high sigma).
# Cost-aware UCB: (mu - kappa * sigma) / cost.
#
# Another idea: The current EI is computed as:
# z = (f_best - mu) / sigma
# EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
#
# Let's check the behavior of EI.
# If mu < f_best, z > 0. norm.cdf(z) is close to 1. norm.pdf(z) is small.
# EI is roughly (f_best - mu).
# If mu > f_best, z < 0. norm.cdf(z) is small. norm.pdf(z) is larger.
# EI can be positive even if mu > f_best, due to the uncertainty term.
#
# The division by cost is key.
#
# Let's try to improve the cost-awareness.
# Maybe use a more sophisticated cost model or a different acquisition function.
#
# Let's try UCB with a time-varying kappa.
# But we don't have access to the iteration count directly, only f_best and cost.
# We can infer some progress from f_best.
#
# Let's try a combination:
# Score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z) - lambda * cost
# Or Score = EI / (cost^alpha) for some alpha.
#
# The seed program uses alpha=1.
# Let's try alpha=0.5 or alpha=2.
#
# Also, we can try to add a small random noise to break ties and encourage exploration.
# But the function must be deterministic for the same inputs? No, the task says "return 2000 finite scores".
# It doesn't say it must be deterministic. But the evaluator runs it 5 times with different seeds.
# So adding randomness might help escape local optima.
# However, the task says "Each function runs on five seeded instances".
# This implies the acquisition function itself might be called multiple times in a loop.
#
# Let's stick to a deterministic function first.
#
# Let's try:
# 1. UCB: (mu - kappa * sigma) / cost
# 2. EI with different alpha: EI / (cost^alpha)
# 3. A hybrid.
#
# Let's analyze the cost.
# cost = 1 + mean(x_scaled).
# x is in [-1, 1]. x_scaled = (x + 1) / 2.
# mean(x_scaled) is in [0, 1].
# So cost is in [1, 2].
#
# If we use EI / cost, we are penalizing expensive points.
# Expensive points are those with high mean(x_scaled), i.e., closer to x=1.
# Cheap points are closer to x=-1.
#
# The minimizer is shifted to a uniform point in the inner 80% of each range.
# So the minimizer is in [-0.8, 0.8] for each coordinate.
#
# Let's try to use the fact that the minimizer is in the center.
# We can bias the acquisition function towards the center.
#
# Let's try a UCB with a center bias.
# Score = (mu - kappa * sigma) / cost + bias(x)
# But we don't have x! We only have mu, sigma, f_best, cost.
#
# Wait, the task says: "No coordinates are passed."
# So we cannot use x directly.
#
# We can only use mu, sigma, f_best, cost.
#
# Let's focus on improving the acquisition function using only these inputs.
#
# Let's try:
# Score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
# Score = Score / (cost ** 1.5)  # More aggressive cost penalty
#
# Or maybe:
# Score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
# Score = Score / cost
# Score = Score + 0.1 * sigma  # Add exploration bonus
#
# Let's try a simple change:
# Use a slightly different EI formula or a UCB.
#
# Let's try UCB:
# Score = (mu - 1.0 * sigma) / cost
#
# Let's try a mix:
# Score = 0.5 * EI/cost + 0.5 * (mu - sigma) / cost
#
# Let's try to implement a deadline guard.
# We need to track elapsed time.
# We can use a global variable or a closure.
# Since the function is called repeatedly, we can store the start time in a global variable.
#
# However, the task says "Keep the function names and signatures of the seed program."
# So we can add global variables.
#
# Let's add a deadline guard.
#
# Also, let's consider the possibility that the GP is not well-calibrated.
# We can use a more robust acquisition function.
#
# Let's try:
# 1. Deadline guard.
# 2. UCB with kappa = 1.0.
# 3. Cost-aware: divide by cost.
#
# Let's code this up.

_start_time = None
_DEADLINE = 320.0  # 330s limit, leave 10s margin

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe result, e.g., uniform random or just mu
        # But we must return 2000 finite scores.
        # Let's return -mu / cost as a fallback
        return -mu / (cost + 1e-9)
    
    # Compute EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI
    score = ei / (cost + 1e-9)
    
    # Add a small exploration bonus based on sigma
    # score += 0.1 * sigma / cost
    
    return score
# EVOLVE-BLOCK-END
