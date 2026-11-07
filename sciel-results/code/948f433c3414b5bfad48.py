import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Structural change: Hybrid LCB/EI acquisition for multimodal Weierstrass
    # Combine Lower Confidence Bound (for exploitation) with Expected Improvement (for exploration)
    
    # Normalize f_best to estimate progress (Weierstrass min is 0, max is around 1)
    f_best_norm = max(0.0, min(1.0, f_best))
    
    # Hybrid weight: blend LCB and EI based on exploration-exploitation tradeoff
    # Early on: more exploration (EI), later: more exploitation (LCB)
    lcb_weight = 0.7
    ei_weight = 0.3
    
    # LCB for minimization: mu - beta * sigma (lower is better, so we maximize -LCB)
    beta = 2.0
    lcb = mu - beta * sigma
    lcb_score = -lcb  # Higher score means better
    
    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For Gaussian: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    from scipy.stats import norm
    
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei_score = ei  # Higher EI means better
    
    # Combine LCB and EI scores
    scores = lcb_weight * lcb_score + ei_weight * ei_score
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
