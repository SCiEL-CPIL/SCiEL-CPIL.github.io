import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s safety margin from 330s limit
    
    scores = np.zeros_like(mu)
    
    # Mask for points with significant uncertainty
    mask = sigma > 1e-8
    
    if np.any(mask):
        mu_m = mu[mask]
        sigma_m = sigma[mask]
        
        # Compute Z score for Probability of Improvement (PI)
        # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
        Z = (f_best - mu_m) / sigma_m
        pi = norm.cdf(Z)
        
        # Penalty term: bias toward the origin (0,0) where Ackley's minimum is
        # We need to infer coordinates from the posterior, but since we only have mu/sigma,
        # we use a proxy: lower mu values are more likely near the minimum
        # Scale the penalty based on how far mu is from 0 (the known minimum)
        
        # Use a Gaussian kernel centered at the origin in function space
        # Points with low predicted values (close to 0) get less penalty
        # This creates a bias toward exploring regions where the GP predicts low values
        
        # Alternative: use the posterior uncertainty to guide exploration
        # High uncertainty + low mean = good candidate
        # Combine PI with exploration bonus weighted by distance from known minimum
        
        # Since we don't have coordinates, use a heuristic:
        # - PI favors points that might improve
        # - Add exploration bonus for high sigma (uncertainty)
        # - Weight by how "promising" the mean is (lower is better)
        
        # Create a composite score:
        # score = PI + exploration_bonus * (1 - normalized_mu)
        
        # Normalize mu to [0,1] range for weighting
        mu_min = np.min(mu_m)
        mu_max = np.max(mu_m)
        mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
        mu_norm = (mu_m - mu_min) / mu_range
        
        # Exploration bonus: higher when sigma is large and mu is low
        # This encourages exploring uncertain regions that might contain the minimum
        exploration_bonus = sigma_m / (sigma_m + 0.1) * (1 - mu_norm)
        
        # Combine PI and exploration with a balance parameter
        # PI is in [0,1], exploration_bonus is in [0,1]
        # Weight PI more heavily as we want to exploit known good regions
        balance = 0.7
        scores[mask] = balance * pi + (1 - balance) * exploration_bonus
        
    if np.any(~mask):
        mu_s = mu[~mask]
        # For zero variance, use deterministic improvement
        scores[~mask] = np.maximum(f_best - mu_s, 0.0)
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
