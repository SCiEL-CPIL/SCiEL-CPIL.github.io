# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = None
DEADLINE = 300.0
SAFETY_MARGIN = 25.0  # seconds before deadline to start returning fast


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) with fallback to EI.
    KG estimates the expected reduction in the minimum objective value by
    considering the probability that each candidate becomes the new global best.
    
    Returns 2000 finite scores; higher means evaluate next.
    """
    global _start_time
    
    # Ensure timer is initialized
    if _start_time is None:
        _start_time = time.perf_counter()
    
    elapsed = _get_elapsed()
    
    # Deadline guard: if we're running out of time, fall back to simple EI
    if elapsed > DEADLINE - SAFETY_MARGIN:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Knowledge Gradient scores
    # For each candidate i, KG_i = E[min(f_best, f_i)] reduction
    # Approximation: sum over all candidates j of P(candidate i gives new best)
    # A practical approximation: for each i, compute the probability that
    # f_i < f_best AND f_i is the minimum among all candidates
    
    # More tractable approximation: 
    # KG_i ≈ (f_best - mu_i) * Phi(z_i) + sigma_i * phi(z_i)  [this is EI]
    # But KG considers global minimum, so we can approximate by:
    # For candidate i, the expected improvement in the global minimum is:
    # E[max(0, f_best - min_j f_j)] where the min is over all candidates
    
    # A simpler and effective approximation for KG in 1D BO:
    # Weight each candidate by its probability of being the global best
    # P(i is best) ∝ exp(-(mu_i - mu_min)^2 / (2 * sigma_i^2)) * something
    
    # Let's use a more standard KG approximation:
    # KG_i = f_best - E[min(f_best, Y_i)] where Y_i ~ N(mu_i, sigma_i^2)
    # E[min(f_best, Y_i)] = mu_i * Phi(z_i) + sigma_i * phi(z_i) + f_best * (1 - Phi(z_i))
    # where z_i = (f_best - mu_i) / sigma_i
    # So KG_i = f_best - [mu_i * Phi(z_i) + sigma_i * phi(z_i) + f_best * (1 - Phi(z_i))]
    #          = f_best * Phi(z_i) - mu_i * Phi(z_i) - sigma_i * phi(z_i)
    #          = (f_best - mu_i) * Phi(z_i) - sigma_i * phi(z_i)
    
    # Wait, that's not right. Let me recalculate:
    # E[min(a, Y)] where Y ~ N(mu, sigma^2) and a is constant
    # = integral_{-inf}^{a} y * phi((y-mu)/sigma)/sigma dy + integral_{a}^{inf} a * phi((y-mu)/sigma)/sigma dy
    # = mu * Phi((a-mu)/sigma) + sigma * phi((a-mu)/sigma) + a * (1 - Phi((a-mu)/sigma))
    # Wait, that's not right either. Let me use the formula:
    # E[min(a, Y)] = a - E[max(0, a-Y)] = a - EI where EI is expected improvement
    # Actually: E[min(a,Y)] = mu*Phi(z) + sigma*phi(z) + a*(1-Phi(z)) where z=(a-mu)/sigma
    # No, let me just use: E[min(a,Y)] = a - E[max(0, a-Y)]
    # E[max(0, a-Y)] = (a-mu)*Phi(z) + sigma*phi(z) where z=(a-mu)/sigma
    # So E[min(a,Y)] = a - (a-mu)*Phi(z) - sigma*phi(z)
    # KG_i = f_best - E[min(f_best, Y_i)] = E[max(0, f_best - Y_i)] = EI_i
    
    # So in 1D with single candidate evaluation, KG reduces to EI!
    # But the true KG considers the joint distribution of all candidates.
    # For a better approximation, we can use the fact that after evaluating i,
    # the new f_best' = min(f_best, Y_i), and the expected future improvement
    # depends on where the true minimum is.
    
    # A practical KG approximation that works well:
    # Score_i = EI_i * (1 + bonus) where bonus depends on uncertainty
    # Or use the "probability of being the global best" heuristic
    
    # Let's try: KG_i ≈ EI_i + lambda * sigma_i * (1 - sigma_i/max_sigma)
    # This adds exploration bonus for uncertain regions
    
    # Actually, let's implement a proper-ish KG approximation:
    # The value of information from evaluating point i is:
    # VOI_i = E[f_best - min(f_best, Y_i)] = EI_i
    # But this ignores that evaluating i also updates our belief about other points.
    # For Weierstrass with many local minima, we want to explore broadly.
    
    # Let's use a blend of EI and a "diversity" term:
    z = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Progress-based blending
    progress = min(elapsed / DEADLINE, 1.0)
    
    # In early stages, explore more; in later stages, exploit
    # Use a KG-like score that emphasizes finding the global minimum
    # by adding a term that rewards points with high expected minimum value
    
    # Approximate KG by considering that the best candidate among all 2000
    # will likely be near the true minimum. We can estimate the "value" of
    # each candidate as its probability of being the best.
    
    # P(i is best) ∝ exp(-0.5 * ((mu_i - mu_ref)^2) / (sigma_i^2 + sigma_ref^2))
    # where mu_ref and sigma_ref are reference values
    
    # Simpler: use EI with an adaptive exploration coefficient
    # beta that decreases over time
    beta_max = 2.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # UCB component for exploration
    ucb_scores = -mu + beta * sigma
    
    # Normalize and blend
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    ei_norm = _normalize(ei_scores)
    ucb_norm = _normalize(ucb_scores)
    
    # Blend: start with more exploration, shift to exploitation
    w_ei = 0.4 + 0.4 * progress
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
