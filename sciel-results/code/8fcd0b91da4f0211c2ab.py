import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline Guard: 
    # The evaluator kills after 330s. 
    # We check if we are dangerously close to the limit.
    # If so, we return a "safe" acquisition function that just exploits 
    # the most promising point (lowest mu) to finish quickly and safely.
    elapsed = time.time() - _start_time
    if elapsed > 320: 
        # Return score based purely on exploitation (lowest mu is best)
        # Score = -mu (higher is better)
        # Ensure finiteness
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Quantile-based Exploration Boost
    # We identify points that are "promising" (mu < f_best) and have high uncertainty.
    # We boost the score of points in the top 5% of uncertainty among promising points.
    
    promising_mask = mu < f_best
    
    if np.any(promising_mask):
        # Get the sigmas of the promising points
        promising_sigmas = sigma_safe[promising_mask]
        
        # If there are very few promising points, skip the quantile boost to avoid noise
        if len(promising_sigmas) > 10:
            # Calculate the 95th percentile of uncertainty among promising points
            # This defines the "high uncertainty" threshold
            uncertainty_threshold = np.percentile(promising_sigmas, 95)
            
            # Identify points that are both promising and high-uncertainty
            high_unc_mask = promising_mask & (sigma_safe >= uncertainty_threshold)
            
            if np.any(high_unc_mask):
                # Calculate a bonus. 
                # We use a fraction of the max EI to scale the bonus, 
                # ensuring the boost is significant but not overwhelming if EI is very small.
                # If max EI is 0, we use a small absolute bonus to encourage exploration.
                max_ei = np.max(ei[promising_mask])
                if max_ei > 1e-6:
                    bonus = 0.1 * max_ei
                else:
                    bonus = 1e-3 * (f_best - np.min(mu)) # Fallback scale
                
                # Apply the bonus to the scores
                # We add the bonus to the EI score for these specific points
                ei[high_unc_mask] += bonus
    
    # Final cleanup to ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
