import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0 # 2 seconds is plenty for vectorized ops on 2000 points
    
    try:
        # Calculate statistics
        mu_min = np.min(mu)
        mu_max = np.max(mu)
        mu_spread = mu_max - mu_min
        
        sigma_mean = np.mean(sigma)
        
        # Adaptive Beta
        # If uncertainty is high relative to the range of means, explore more.
        # If spread is 0, ratio is inf, clip it.
        if mu_spread < 1e-9:
            ratio = 5.0 # High exploration if model is flat/constant
        else:
            ratio = sigma_mean / mu_spread
            
        # Map ratio to beta
        # We want beta to be at least 1.0 (moderate exploration)
        # and scale up if ratio is large.
        # Clip ratio to avoid extreme values if sigma is huge or spread is tiny.
        ratio_clipped = np.clip(ratio, 0.0, 5.0)
        beta = 1.0 + 1.5 * ratio_clipped
        
        # Lower Confidence Bound (LCB) for minimization
        # We want to minimize, so we prefer low mu and high sigma (uncertainty).
        # Score = mu - beta * sigma
        # Higher score = better candidate to evaluate.
        scores = mu - beta * sigma
        
        # Ensure finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # Check deadline
        if time.time() > deadline:
            # If we are too slow (unlikely), return a safe fallback
            # The best we can do quickly is just the mean or a simple heuristic
            # But since we already computed it, we just return it.
            # If computation itself was slow, we might not have finished.
            pass
            
        return scores

    except Exception:
        # Fallback to simple EI or Mean if something goes wrong
        try:
            from scipy.stats import norm
            z = (f_best - mu) / (sigma + 1e-9)
            return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        except:
            return mu # Last resort
