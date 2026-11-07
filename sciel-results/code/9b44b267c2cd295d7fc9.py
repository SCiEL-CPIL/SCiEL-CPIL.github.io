"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to keep track of time for the deadline guard
# The evaluator kills after 330s wall-clock. We should aim to finish well before.
# However, the acquisition function is called 25 times per seed (30 total evals - 5 initial).
# 3 seeds * 25 calls = 75 calls total.
# 330s / 75 calls = 4.4 seconds per call.
# The computation inside acquisition is vectorized over 2000 points, so it should be very fast (< 10ms).
# The bottleneck is likely not the acquisition itself but the GP fit in the loop (not part of this function).
# But the prompt says "Each evaluation must finish within 300 s". This usually refers to the whole step.
# Since we only control `acquisition`, we just need to make it fast and robust.

# Let's implement a more robust acquisition function.
# Standard EI is good, but sometimes adding a small exploration term or using a different metric helps.
# For Branin, EI is standard. Let's stick with EI but ensure numerical stability.
# We can also try Expected Improvement with a small kappa (like EIC - Expected Improvement with Constant) 
# or simply standard EI.
# The current code uses standard EI.
# Let's try to improve it by handling the case where sigma is very small or zero more gracefully, 
# or by using a slightly different formulation.

# Actually, the current score is 1.0000, which is the baseline.
# To improve, we need a better acquisition function.
# Common alternatives:
# 1. Probability of Improvement (PI)
# 2. Expected Improvement (EI)
# 3. Upper Confidence Bound (UCB)
# 4. Expected Improvement with noise (EI with kappa)

# For minimization:
# UCB: mu - beta * sigma
# PI: P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma)
# EI: (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z) where z = (f_best - mu) / sigma

# UCB can be more aggressive in exploration. 
# Let's try UCB with a beta parameter.
# A common choice for beta is sqrt(2 * log(t)) where t is the number of evaluations.
# However, we don't have access to t directly in the function signature.
# We can infer t from the number of points? No, mu and sigma are always shape (2000,).
# We don't have the iteration count.

# Without iteration count, a fixed beta might be okay, or we can use a heuristic.
# But EI is generally more robust for BO.

# Let's look at the "INSPIRATIONS" - none provided.
# Let's look at "LESSONS" - none provided.

# Let's try a variant of EI that is slightly more exploratory.
# One common trick is to add a small constant to the improvement, or use a different kernel.
# But we can't change the kernel.

# Another idea: The GP is standardized. 
# Maybe the issue is that EI is too greedy?
# Let's try UCB with a fixed beta. 
# For Branin, a beta around 1.0 to 2.0 often works well.
# Let's try beta = 1.0.

# Wait, the prompt says "Write a new design from scratch".
# Let's consider that the current EI might be stuck in local minima due to the small length scale (0.2).
# A more exploratory acquisition might help.

# Let's implement UCB.
def acquisition(mu, sigma, f_best):
    # Use UCB with a fixed beta
    beta = 1.0
    # Higher score means we want to evaluate next.
    # For minimization, we want low mu and high sigma.
    # Score = mu - beta * sigma
    # But we want higher score to be better.
    # So we should return -mu + beta * sigma ?
    # Or simply return the negative of the UCB value?
    # UCB value = mu - beta * sigma (this is the lower bound of the function value)
    # We want to minimize this, so we want the *lowest* UCB value.
    # But the function returns scores where *higher* is better.
    # So we should return -UCB = -mu + beta * sigma.
    
    # Let's check the current EI code:
    # return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # This is positive when mu < f_best, which is good.
    # So higher EI is better.
    
    # For UCB:
    # We want to find points with low predicted mean and high uncertainty.
    # Score = -mu + beta * sigma
    # This is a valid score.
    
    # However, UCB can be unstable.
    # Let's try a combination or a different EI variant.
    
    # Actually, let's stick with EI but make it more robust.
    # The current EI is standard.
    # Maybe the issue is that the GP is not well-calibrated?
    # Or maybe we should use a different acquisition function altogether.
    
    # Let's try "Expected Improvement with a small exploration bonus"
    # Or simply use UCB with a tuned beta.
    
    # Given the small domain and small length scale, exploration is key.
    # Let's try UCB with beta = 1.5.
    
    # But wait, if I change to UCB, it might be worse.
    # Let's try to improve the EI calculation for numerical stability.
    # The current code uses 1e-9 for sigma.
    # Let's use a slightly larger epsilon to avoid division by zero issues.
    
    # Another idea: Use the "Probability of Improvement" (PI) which is simpler and sometimes more robust.
    # PI = norm.cdf((f_best - mu) / sigma)
    # This is between 0 and 1.
    
    # Let's try PI.
    
    # Or, let's try a hybrid: EI + small * UCB
    # This is getting complicated without tuning.
    
    # Let's look at the observed changes.
    # p15 had a score of 0.015, which is much worse.
    # p3 (current) has 1.0.
    # So the current EI is the best we have.
    
    # To beat 1.0, we need a better acquisition.
    # Common improvements for EI:
    # 1. Use a better approximation of the normal CDF/PDF? No, scipy is fine.
    # 2. Use a different metric like "Expected Simple Regret"?
    # 3. Use "Knowledge Gradient"? Too expensive.
    
    # Let's try a simple modification: 
    # Add a small exploration term to EI.
    # Score = EI + epsilon * sigma
    # This encourages exploring high variance regions.
    
    # Let's try epsilon = 0.1.
    
    # Or, let's try UCB with a beta that adapts to the number of evaluations.
    # We don't have t, but we can estimate it from the number of unique points?
    # No, we only get mu and sigma for 2000 candidates.
    
    # Let's assume the current EI is good, and try to tweak it slightly.
    # Maybe the issue is that f_best is not updated correctly in the loop?
    # No, that's the evaluator's job.
    
    # Let's try a different acquisition: 
    # "Expected Improvement with a constant" (EIC)
    # EIC = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    # where z = (f_best - mu - xi) / sigma
    # xi is a small constant, e.g., 0.01 * (f_max - f_min)
    # This makes the acquisition less greedy.
    
    # But we don't know f_max and f_min.
    # We can approximate f_max - f_min by the range of mu?
    # No, that's not reliable.
    
    # Let's stick with standard EI but ensure it's computed correctly.
    # The current code is correct.
    
    # What if we use a different distribution?
    # No, the GP is fixed.
    
    # Let's try UCB with beta = 1.0.
    # It's a simple change and might work better for exploration.
    
    # Let's implement UCB.
    beta = 1.0
    scores = -mu + beta * sigma
    return scores

# Wait, I should be careful. UCB might be too exploratory.
# Let's try a compromise: 
# Score = EI + 0.1 * sigma
# This adds a small exploration bonus.

def acquisition(mu, sigma, f_best):
    # Standard EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus
    exploration_bonus = 0.1 * sigma
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
