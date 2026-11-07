"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to track time to avoid timeout.
# The evaluator kills after 330s. We have 5 instances.
# Each instance has a loop of cost < 30.
# Initial cost is ~ 5 * (1 + 0.5) = 7.5.
# Remaining cost budget ~ 22.5.
# Each step cost is 1 + mean(x). Mean x is roughly 0.5 if uniform. So ~1.5 per step.
# So ~ 15 steps per instance.
# Total steps ~ 75.
# Each step scores 2000 candidates.
# So ~ 150,000 acquisition evaluations total.
# This is small enough that we don't need to worry about timeout unless the acquisition is very slow.
# However, the prompt asks for an explicit deadline guard.

_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin

def _time_exceeded():
    return (time.time() - _start_time) > TIME_LIMIT

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses a Pareto-optimal cost-normalization strategy:
    Maximize (EI) / (marginal cost).
    
    Marginal cost is approximated by the cost of the candidate relative to the average cost of the current batch,
    or simply the cost itself if we treat each step independently.
    The prompt suggests: "ratio of the acquisition value ... to the marginal cost relative to the current best cost".
    This is a bit ambiguous. Let's interpret "marginal cost" as the cost of the candidate.
    However, the directive says: "treat the search as a multi-objective optimization".
    
    Let's try a different structural change: 
    Instead of just EI/cost, use a UCB-based acquisition with a cost penalty, 
    and add a diversity term to avoid clustering.
    
    Actually, the lesson says: "Combining EI and UCB with a diversity penalty ... Measured: 0.3459 -> 1.699."
    This is a strong signal. Let's implement this.
    
    Acquisition = w1 * EI + w2 * UCB - w3 * Diversity_Penalty / cost?
    Or maybe: (w1 * EI + w2 * UCB) / cost - w3 * Diversity_Penalty?
    
    Let's stick to the high-performing strategy found in the lesson:
    Score = (EI + kappa * sigma) / cost - diversity_penalty
    
    But we don't have coordinates! We only have mu, sigma, cost.
    How to compute diversity penalty without coordinates?
    We can't. The diversity penalty in the lesson likely relied on coordinates or was computed differently.
    Wait, the input to `acquisition` is `mu, sigma, f_best, cost`. No coordinates.
    So we cannot compute distance to other points.
    
    Therefore, we must rely on the GP properties.
    High sigma implies unexplored regions.
    So UCB = mu + kappa * sigma naturally provides exploration.
    
    Let's refine the cost-normalization.
    The prompt suggests: "ratio of the acquisition value ... to the marginal cost".
    Let's use: (EI + kappa * sigma) / (cost + epsilon)
    
    However, the seed program uses EI / cost.
    The lesson says combining EI and UCB worked well.
    Let's try: (EI + kappa * sigma) / cost.
    
    We need to tune kappa.
    Also, we should handle the case where sigma is very small.
    
    Let's add a time guard.
    If time is exceeded, return a default value (e.g., zeros or EI/cost) to finish quickly.
    """
    if _time_exceeded():
        # Fallback to simple EI/cost to be fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Calculate EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB term
    # kappa controls exploration. 
    # In the lesson, the specific combination was not detailed, but "EI and UCB with diversity".
    # Since we can't do diversity, let's blend EI and UCB.
    # UCB = mu + kappa * sigma. 
    # But we are minimizing. So we want low mu.
    # Let's use a weighted sum of EI and a "potential" term.
    # Potential = -mu + kappa * sigma (since we want to minimize, -mu is good).
    # Or just use EI which already balances exploitation (f_best - mu) and exploration (sigma).
    
    # The seed program is EI / cost.
    # The lesson says EI + UCB + Diversity improved score.
    # Let's try: (EI + 0.1 * sigma) / cost.
    # This adds a small exploration bonus.
    
    # Also, consider the "Pareto-optimal cost-normalization" directive.
    # "ratio of the acquisition value ... to the marginal cost relative to the current best cost"
    # This might mean: value / (cost - min_cost) or value / (cost / mean_cost).
    # Let's try normalizing cost by the mean cost of the candidates.
    # This makes the acquisition scale-invariant to the cost distribution.
    
    mean_cost = np.mean(cost)
    if mean_cost > 0:
        normalized_cost = cost / mean_cost
    else:
        normalized_cost = cost + 1e-9
        
    # Acquisition: (EI + kappa * sigma) / normalized_cost
    kappa = 0.5
    acq = (ei + kappa * sigma) / (normalized_cost + 1e-9)
    
    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
