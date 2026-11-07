"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280

def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function using Knowledge Gradient approximation.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        # Fallback to simple cost-aware EI if we are running out of time
        sigma_fb = np.maximum(np_sigma, 1e-12)
        z_fb = (f_best - mu) / sigma_fb
        cdf_fb = norm.cdf(z_fb)
        pdf_fb = norm.pdf(z_fb)
        ei_fb = (f_best - mu) * cdf_fb + sigma_fb * pdf_fb
        ei_fb = np.maximum(ei_fb, 0.0)
        score_fb = ei_fb / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score_fb), score_fb, 0.0)

    # Sanitize inputs
    sigma = np.maximum(np_sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Normalize mu and sigma for numerical stability in the KG calculation
    # The KG for minimizing is approximated by the expected reduction in the best value
    # E[ max(0, f_best - Y) ] where Y ~ N(mu, sigma^2)
    # This is exactly Expected Improvement (EI).
    # However, standard EI can be overly greedy. 
    # A more sophisticated approach is to consider the "Value of Information" 
    # relative to the cost.
    
    # Let's try a different family: Probability of Improvement (POI) blended with EI,
    # but adjusted for cost via a "soft" penalty that doesn't zero out high cost points
    # too aggressively, allowing exploration of expensive but potentially very good areas.
    
    # 1. Compute Standard EI
    z = (f_best - mu) / sigma
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Probability of Improvement (POI)
    # POI is a smoother signal than EI, less prone to numerical spikes
    poi = cdf_vals
    
    # 3. Cost-Aware Scoring
    # Instead of dividing by cost (which can be very punishing), 
    # we use a cost-efficiency metric that balances improvement per unit cost.
    # We use a geometric mean or a soft harmonic mean to avoid extreme penalties.
    
    # Strategy: Score = EI * (1 / (1 + cost_penalty))
    # Or better: Score = EI / (cost^alpha) where alpha < 1 to soften the penalty.
    # The insight from the 1.6 score was using cost^0.7. Let's try a similar softening
    # but also incorporate POI to guide exploration in regions with low EI but high uncertainty.
    
    # Let's use a weighted sum:
    # Score = w_ei * (EI / cost^0.7) + w_poi * (POI / cost^0.5)
    
    # Normalize costs to [0, 1] range for stable exponentiation
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Map normalized cost to a penalty factor in [1, 2] (mild penalty)
    # Lower cost -> factor closer to 1, Higher cost -> factor closer to 2
    cost_penalty_ei = 1.0 + cost_norm * 1.0 
    cost_penalty_poi = 1.0 + cost_norm * 0.5
    
    # Calculate components
    ei_term = ei / cost_penalty_ei
    poi_term = poi / cost_penalty_poi
    
    # We need to balance the scales. EI is in objective units, POI is in [0,1].
    # Scale POI by the typical magnitude of EI or f_best to make them comparable.
    # If f_best is large, POI needs a larger weight.
    # Let's use the current best value as a scale, but capped.
    scale_factor = max(abs(f_best), 1.0)
    
    # Weighting: 
    # Early on, we want more exploration (POI). Later, more exploitation (EI).
    # We don't have explicit time steps, but we can infer progress from f_best relative to 0.
    # However, a simpler robust approach is a fixed blend that favors EI slightly
    # because EI directly estimates improvement.
    
    # Let's try: Score = ei_term + 0.5 * poi_term * scale_factor
    # The 0.5 factor dampens the POI term so it doesn't dominate if scale is large.
    
    score = ei_term + 0.5 * poi_term * scale_factor
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
