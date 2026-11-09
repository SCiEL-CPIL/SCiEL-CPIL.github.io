import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining UCB exploration with Pareto-frontier
    guided geometric scoring for Bayesian optimization of Ackley 2D.
    
    Strategy:
    1. Compute UCB scores for strong exploration pressure (linear in sigma).
    2. Identify the exploitation anchor (lowest predicted mean).
    3. Compute geometric distance from anchor in prediction space.
    4. Combine UCB with a geometric exploration term that prioritizes 
       high-uncertainty regions far from the current best prediction.
    
    This structural change moves away from probability-saturated EI toward
    a hybrid that maintains strong exploration pressure while using the
    geometric structure of the prediction space to guide search.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    if time.time() - _START_TIME > 320:
        return -mu  # Greedy fallback: minimize mu
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # --- Component 1: UCB for strong exploration ---
    # UCB for minimization: maximize -mu + beta * sigma
    # This provides linear exploration pressure, unlike EI's saturation
    beta = 1.5
    ucb_scores = -mu + beta * sigma_safe
    
    # --- Component 2: Pareto-frontier guided geometric exploration ---
    # Find the exploitation anchor: candidate with lowest predicted mean
    mu_min_idx = np.argmin(mu)
    mu_anchor = mu[mu_min_idx]
    
    # Compute normalized distance from anchor in prediction space
    # Distance in (mu, sigma) space, normalized by sigma scale
    # This creates a geometric prior that favors regions far from the 
    # current best prediction but with high uncertainty
    mu_diff = mu - mu_anchor
    # Normalize by typical sigma scale to make distance dimensionless
    sigma_scale = np.median(sigma_safe[sigma_safe > 1e-10]) if np.any(sigma_safe > 1e-10) else 1.0
    sigma_scale = max(sigma_scale, 1e-10)
    
    # Geometric distance: combines mu deviation and sigma
    # Points with high sigma AND far from anchor in mu get boosted
    geo_distance = np.sqrt((mu_diff / sigma_scale)**2 + (sigma_safe / sigma_scale)**2)
    
    # Normalize geo_distance to [0, 1] range for stable combination
    geo_max = np.max(geo_distance) if np.max(geo_distance) > 0 else 1.0
    geo_norm = geo_distance / geo_max
    
    # --- Combination: Weighted blend of UCB and geometric exploration ---
    # The UCB provides the base exploration signal
    # The geometric term adds a structural bias toward unexplored regions
    # that are far from the current best prediction
    # 
    # Use a smooth combination: UCB dominates when it's strong,
    # geometric term modulates to avoid local optima
    w_geo = 0.3  # Weight for geometric exploration component
    
    # Scale UCB to comparable range
    ucb_range = np.ptp(ucb_scores) if np.ptp(ucb_scores) > 0 else 1.0
    ucb_norm = (ucb_scores - np.min(ucb_scores)) / ucb_range
    
    # Final score: blend of normalized UCB and geometric distance
    # Higher score = more promising
    scores = ucb_norm + w_geo * geo_norm
    
    # Add a tiny exploitation term to break ties when sigma is very small
    # This ensures that among points with near-zero sigma, the one with
    # the lowest mu gets a slightly higher score
    eps = 1e-6
    scores = scores + eps * (-mu)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
