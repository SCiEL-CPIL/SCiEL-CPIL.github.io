"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB exploration with cost normalization.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    normalized by cost^gamma to strongly favor low-cost evaluations.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard deviation for numerical stability in z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) with adaptive kappa
    # Use a moderate kappa to balance exploration
    kappa = 1.5
    ucb = mu + kappa * sigma_safe
    
    # Combine EI and UCB: use EI when it's positive (exploitation),
    # but add a UCB component to encourage exploration
    # The EI is in objective units (positive means improvement over f_best)
    # UCB is also in objective units (lower is better, so we want high UCB score? No.)
    # Wait, for minimization:
    #   EI = E[max(f_best - f(x), 0)] -> higher is better
    #   UCB for minimization: we want low mu + high sigma. 
    #   Standard UCB for maximization: mu + kappa*sigma
    #   For minimization: mu - kappa*sigma (optimistic lower bound)
    #   But we want a score where HIGHER = evaluate next.
    #   So for minimization, a good candidate has low mu and high sigma.
    #   Score = -mu + kappa*sigma? Or just use EI which already handles this.
    
    # Let's stick to a robust cost-normalized EI, but with a stronger exploration boost.
    # The directive suggests UCB normalized by cost^gamma.
    # For minimization, the "optimistic" value is mu - kappa*sigma (lower is better).
    # We want to maximize a score. So score = -(mu - kappa*sigma) / cost^gamma = (-mu + kappa*sigma) / cost^gamma
    # However, pure UCB can be too aggressive. A blend with EI is often more stable.
    
    # Let's try a cost-normalized EI with an added exploration term.
    # EI is already a good measure of expected improvement.
    # Let's enhance EI by adding a small UCB-like term to handle flat regions.
    
    # Alternative: Pure cost-normalized EI worked at 1.0.
    # Let's try cost-normalized UCB as per directive.
    # UCB_min = mu - kappa * sigma (this is the optimistic lower bound)
    # We want to maximize this? No, we want to minimize the function.
    # So a point with low UCB_min is promising.
    # Score should be high for promising points.
    # Score = -UCB_min / cost^gamma = (-mu + kappa*sigma) / cost^gamma
    
    gamma = 3.0
    kappa = 2.0
    
    # Calculate UCB score for minimization
    ucb_score = (-mu + kappa * sigma_safe) / (cost ** gamma)
    
    # Calculate EI score
    ei_score = ei / (cost ** gamma)
    
    # Combine: Use EI as primary, but add a fraction of UCB to ensure exploration
    # If EI is very low (flat region), UCB will drive exploration.
    # We can blend them. Let's try 0.5 * EI + 0.5 * UCB?
    # Or just use UCB as the directive suggests.
    
    # Let's use a weighted combination to be safe.
    # EI is in [0, inf), UCB can be negative.
    # To make them comparable, we might need scaling.
    # But both are divided by cost^gamma.
    
    # Let's try the directive's approach: UCB normalized by cost^gamma.
    # But pure UCB might ignore known good points if they have low sigma.
    # EI is better at exploiting.
    
    # Let's try: score = (ei + 0.1 * sigma_safe) / cost^gamma
    # This adds a small exploration bonus to EI.
    
    # Actually, let's look at the seed: EI / cost.
    # The directive says: UCB / cost^gamma with high gamma.
    # Let's implement exactly that, but with a twist: use a "soft" UCB that incorporates EI.
    
    # Let's try a different approach: Probability of Improvement (PI) is more stable than EI?
    # PI = norm.cdf(z). Score = PI / cost^gamma.
    
    # Let's try a blend of EI and PI.
    # PI is in [0, 1]. EI can be large.
    
    # Let's go with the directive: UCB.
    # For minimization, the acquisition function is often:
    # alpha(x) = -mu(x) + kappa * sigma(x)  (maximize this)
    # Or: alpha(x) = mu(x) - kappa * sigma(x) (minimize this)
    # We return scores where higher is better.
    # So we return: (-mu + kappa * sigma) / cost^gamma
    
    # To prevent UCB from being too noisy, let's cap the exploration.
    # Or use a larger gamma to make cost the dominant factor initially.
    
    final_score = ucb_score
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
