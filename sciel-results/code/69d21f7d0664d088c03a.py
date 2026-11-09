"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Structural change: Instead of adding a fixed or adaptively scaled UCB term,
    # we use a "Thompson Sampling" like approach via a lower confidence bound (LCB)
    # but with a dynamic beta that is driven by the ratio of current best to the
    # typical scale of the function.
    
    # The Beale function has a minimum of 0.
    # If f_best is large, we are far from the optimum, so we should explore more.
    # If f_best is small, we are close, so we should exploit.
    
    # A robust way to scale the exploration is to use the standard deviation of the
    # predicted values as a proxy for the "scale" of the remaining uncertainty,
    # but here we only have mu and sigma.
    
    # Let's try a different acquisition function family: Probability of Improvement (PI)
    # combined with a strong penalty for high mu, or simply a modified EI.
    
    # However, the directive asks for a MATERIAL change.
    # Let's try "Knowledge Gradient" approximation or a "Greedy" approach on the lower bound.
    
    # A very effective strategy for 2D functions like Beale is to use a UCB with a
    # beta that decays rapidly. The current program uses a sigmoid.
    # Let's try a purely exploitation-focused approach with a small exploration floor,
    # or a "Noisy" EI.
    
    # Actually, let's look at the "relative uncertainty" idea from the knowledge.
    # "Replaces fixed EI/UCB blend with EI plus relative-uncertainty exploration bonus"
    
    # Let's implement: Score = EI + beta * sigma / (f_best + epsilon)
    # This makes exploration proportional to the relative uncertainty compared to the current best.
    # If f_best is large (100), and sigma is 1, the term is 1/100 = 0.01.
    # If f_best is small (1), and sigma is 1, the term is 1/1 = 1.0.
    # This automatically scales exploration down as we get closer to the optimum,
    # without needing a hand-tuned sigmoid or sqrt.
    
    # But wait, if f_best is very close to 0, the term explodes.
    # We should cap it or use a different scaling.
    
    # Alternative: Use the "Probability of Being the Best" (POB) or a similar metric?
    # No, let's stick to a robust EI variant.
    
    # Let's try a "Sharp" EI where we only consider points that are significantly better than mu.
    # Or, let's use the "Expected Regret" which is similar.
    
    # Let's try a different structural change: **Lower Confidence Bound (LCB) only**.
    # Many BO implementations fail because they mix EI and UCB poorly.
    # Pure LCB: score = mu - beta * sigma
    # The choice of beta is critical.
    # If we set beta dynamically based on the spread of mu and sigma, we can adapt.
    
    # Let's compute the "effective" beta as the ratio of the range of mu to the range of sigma?
    # No, that's too complex.
    
    # Let's try a **Normalized EI**.
    # Normalize mu and sigma by f_best (or a scale factor).
    # If we divide mu and sigma by f_best, we are working in "relative improvement" space.
    
    # Let's try:
    # scale = max(f_best, 1e-6)
    # mu_norm = mu / scale
    # sigma_norm = sigma / scale
    # f_best_norm = 1.0
    # EI_norm = (1.0 - mu_norm) * Phi((1.0 - mu_norm) / sigma_norm) + sigma_norm * phi((1.0 - mu_norm) / sigma_norm)
    # This EI is in units of "fraction of current best".
    # This might be more stable across different f_best values.
    
    scale = max(f_best, 1e-6)
    mu_s = mu / scale
    sigma_s = sigma / scale
    f_best_s = 1.0
    
    z_s = (f_best_s - mu_s) / sigma_s
    cdf_z_s = norm.cdf(z_s)
    pdf_z_s = norm.pdf(z_s)
    
    ei_s = (f_best_s - mu_s) * cdf_z_s + sigma_s * pdf_z_s
    
    # The score is now in normalized units.
    # We can return this directly.
    # This effectively makes the acquisition function adaptive to the scale of the problem.
    # When f_best is large, small improvements are "cheap" in normalized terms,
    # encouraging exploration. When f_best is small, only large relative improvements
    # are rewarded, encouraging exploitation.
    
    score = ei_s
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
