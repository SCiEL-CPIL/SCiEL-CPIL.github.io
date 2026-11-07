"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural shift:
    instead of a fixed blend of EI and UCB, we use a two-stage
    cost-aware score that adapts its exploration/exploitation
    balance based on the current f_best and the uncertainty landscape.
    
    The key structural change: we compute a "confidence-weighted improvement"
    that uses the ratio of expected improvement to cost, but modulates the
    exploration bonus by how far we are from the known optimum region.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin for 330s limit
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Structural change: Instead of a fixed UCB blend, we use an adaptive
    # exploration term that scales with the uncertainty relative to the
    # improvement potential. This creates a dynamic balance:
    # - When f_best is far from the predicted minimum (large ei), exploit
    # - When f_best is close to predicted minima (small ei), explore more
    
    # Compute the "exploration pressure" based on how much uncertainty
    # exists relative to the current best
    # If sigma is large compared to (f_best - min_mu), we should explore more
    mu_min = np.min(mu)
    improvement_scale = max(f_best - mu_min, 1e-9)
    
    # Adaptive beta: larger when uncertainty is high relative to improvement
    # This makes exploration more aggressive when the GP is uncertain
    # and the current best isn't much better than the predicted minimum
    beta = 1.5 + 1.5 * np.clip(sigma_safe / improvement_scale, 0, 2)
    
    # UCB for minimization
    ucb = -mu + beta * sigma_safe
    
    # Compute a "confidence" term: how confident are we that this point
    # will actually improve? This is based on the probability that
    # f(x) < f_best, which is Phi((f_best - mu) / sigma)
    prob_improve = norm.cdf(z)
    
    # Structural change: Use a product-like combination where the exploration
    # term is gated by the probability of improvement, but with a floor
    # to ensure some exploration always occurs
    # This is different from a linear blend: it creates a multiplicative
    # interaction between exploitation confidence and exploration value
    
    # Gate the UCB by a soft confidence term
    # When prob_improve is high, trust the mean more (exploit)
    # When prob_improve is low, allow more exploration
    confidence_gate = 0.3 + 0.7 * prob_improve
    
    # Combine: EI provides the exploitation signal, UCB provides exploration
    # The confidence gate modulates how much of the UCB is "allowed through"
    base_score = ei + confidence_gate * ucb
    
    # Cost normalization: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = base_score / cost_safe
    
    # Clip negative scores to 0
    scores = np.maximum(scores, 0.0)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
