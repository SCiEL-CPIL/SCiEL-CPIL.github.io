import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive Beta for LCB
    # If sigma is high everywhere, we are early in the process -> explore more.
    # If sigma is low, we are late -> exploit more.
    # Use the ratio of max sigma to median sigma as a proxy for "exploration needed"
    med_sigma = np.median(sigma)
    if med_sigma > 1e-12:
        # Scale beta based on uncertainty spread
        # A high ratio means some regions are very uncertain -> explore them.
        # A low ratio means uncertainty is uniform or low -> exploit.
        ratio = np.max(sigma) / med_sigma
        # Map ratio to beta. 
        # If ratio is 1, beta is small. If ratio is 10, beta is large.
        # Let's use a soft cap.
        beta = np.clip(0.5 * np.log1p(ratio), 0.5, 5.0)
    else:
        beta = 0.5
    
    # LCB for minimization: We want to minimize f.
    # The "improvement" is f_best - f.
    # LCB on improvement: (f_best - mu) - beta * sigma.
    # If this is negative, we don't expect improvement.
    lcb_improvement = (f_best - mu) - beta * sigma_safe
    
    # We only care if LCB is positive? 
    # Actually, LCB can be negative. If it is, the score should be low.
    # But EI is always >= 0.
    # To maintain a similar scale, we can use max(0, LCB_improvement).
    # However, LCB is a lower bound. If LCB > 0, we are *certain* of improvement.
    # If LCB < 0, it's possible but not certain.
    # EI captures the expectation. LCB captures the pessimistic case.
    # For exploration, we want to pick points where LCB is high.
    
    # Let's use a hybrid: Score = max(0, LCB_improvement) / Cost
    # This strongly favors points where we are *confident* about improvement, 
    # adjusted by the uncertainty term beta*sigma.
    
    # Wait, standard LCB for minimization is mu - beta*sigma.
    # We want to minimize mu - beta*sigma.
    # So we want to MAXIMIZE -mu + beta*sigma.
    # But we also need to compare to f_best.
    # The "Improvement LCB" is (f_best - mu) - beta*sigma.
    
    # Let's try: Score = max(0, (f_best - mu) - beta * sigma) / Cost
    # This is different from EI. EI is smooth. This is piecewise linear.
    
    # Alternative: Use "Expected Improvement" but with a "Penalty" for high cost that is 
    # not just division, but a "Budget Penalty".
    # Given the directive to make a structural change, LCB is a distinct family.
    
    lcb_scores = np.maximum(0.0, lcb_improvement)
    
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = lcb_scores / cost_safe
    
    # Add a small exploration term for points with high sigma but low LCB?
    # No, LCB already includes sigma.
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
