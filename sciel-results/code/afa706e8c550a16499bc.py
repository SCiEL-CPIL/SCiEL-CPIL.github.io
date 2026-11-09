"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement (EI) normalized by cost.
    To improve upon the baseline, we use a slightly more robust formulation:
    1. Handle zero/negative sigma safely.
    2. Use a small epsilon to avoid division by zero in cost.
    3. The core logic is EI / cost, which is standard for cost-aware BO.
    
    However, to differentiate from the 'seed' program (which is exactly EI/cost),
    let's consider if there's a better cost-aware metric. 
    Actually, the seed program IS EI/cost. The directive says "design a new program... from scratch; do not reuse the current designs."
    But the current program is the seed. So I must implement something different.
    
    Alternative: Probability of Improvement (PI) divided by cost?
    Or: Expected Improvement with a penalty for high cost?
    
    Let's try a different approach: Use the UCB (Upper Confidence Bound) formulation but for minimization, which is often equivalent to EI.
    Actually, a common alternative is the "Cost-Adjusted Expected Improvement" where we might weight the sigma term differently.
    
    Let's look at the structure:
    mu: predicted mean
    sigma: predicted std dev
    f_best: best observed value
    cost: cost of evaluating this point
    
    The standard EI for minimization is:
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    The seed program returns EI / cost.
    
    What if we use a different acquisition function?
    Maybe PI (Probability of Improvement) / cost?
    PI = Phi(z)
    
    Or maybe we can use a more sophisticated cost-aware metric like the "Expected Improvement per unit cost" but with a different baseline?
    
    Actually, let's think about the problem: we have a fixed budget (cost < 30). We want to minimize simple regret.
    EI/cost is a very strong baseline.
    
    Let's try a different family: "Thompson Sampling" style? But we only have mu and sigma, not the full posterior.
    
    How about "Knowledge Gradient"? Hard to compute with just mu/sigma.
    
    Let's try a variant of EI that is more robust to high uncertainty.
    Sometimes, EI can be dominated by high sigma. A cost-aware PI might be more robust.
    
    Let's implement PI / cost.
    z = (f_best - mu) / sigma
    PI = Phi(z)
    Score = PI / cost
    
    This is a different design.
    """
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: higher is better
    # We divide by cost to favor cheaper points
    cost_safe = np.maximum(cost, 1e-9)
    
    return pi / cost_safe
# EVOLVE-BLOCK-END
