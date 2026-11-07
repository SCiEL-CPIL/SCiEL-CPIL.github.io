"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Safety margin: 300s limit, leave 5s for return overhead
    deadline = start_time + 295.0
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI)
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    z_ei = (f_best - mu) / sigma
    ei_score = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # 2. Upper Confidence Bound (UCB)
    # UCB: mu + kappa * sigma
    # We want to maximize UCB, but since we are minimizing f, 
    # high UCB means high uncertainty + high potential for low f (if mu is low).
    # Actually, for minimization, we want low mu. 
    # UCB for minimization is often: mu - kappa * sigma (Lower Confidence Bound, LCB)
    # OR we can just use the standard UCB logic where we add uncertainty to the "goodness".
    # Let's use LCB (Lower Confidence Bound) which is standard for minimization:
    # LCB = mu - kappa * sigma. Lower is better.
    # So we want to MAXIMIZE -LCB = -mu + kappa * sigma.
    # Let's define exploration_score = -mu + kappa * sigma.
    
    # We need to scale these to be comparable.
    # EI is in units of f. LCB is in units of f.
    # However, EI can be very small if f_best is close to mu.
    # Let's use a dynamic kappa or a fixed one.
    # A common heuristic is kappa ~ 2.0.
    # To make them comparable, we can just take the max of the two scores.
    # But EI is non-negative. LCB score can be negative.
    # Let's normalize or just rely on the fact that EI dominates when improvement is certain,
    # and LCB dominates when uncertainty is high.
    
    # Let's try a weighted sum or max.
    # Max is more robust against scale issues if they are in similar ranges.
    # Let's use a coefficient for the exploration term to balance it.
    # If f_best is -3.8, and mu is -3.5, sigma is 0.1.
    # EI ~ small.
    # LCB score = -(-3.5) + 2*0.1 = 3.7.
    # This might be too high compared to EI.
    
    # Alternative: Use the "Pareto" approach directly on the candidates.
    # Since we don't have coordinates, we can't do local search.
    # We must return scores for all 2000 points.
    
    # Let's stick to a robust combination.
    # The issue with pure EI is that it can be very flat in regions of high sigma.
    # UCB/LCB provides a stronger gradient towards high sigma regions.
    
    # Let's compute a "Composite Score"
    # score = max(EI, alpha * (-mu + beta * sigma))
    
    # To avoid scaling issues, let's normalize the exploration term relative to the current scale.
    # However, simple addition or max is often sufficient if the scales are similar.
    # In Bayesian Optimization, EI and UCB are often comparable in magnitude.
    
    # Let's use a fixed kappa for UCB.
    kappa = 2.0
    
    # Exploration score (higher is better)
    # We want to go where mu is low (good) and sigma is high (uncertain).
    # So exploration_score = -mu + kappa * sigma
    
    exploration_score = -mu + kappa * sigma
    
    # We need to ensure that if EI is clearly better, we pick it.
    # If exploration is clearly better, we pick it.
    # Taking the element-wise max is a valid Pareto-frontier strategy.
    
    # However, EI is always >= 0.
    # Exploration score can be anything.
    # If mu is very high (bad) and sigma is small, exploration score is very negative.
    # If mu is very low (good) and sigma is small, exploration score is very positive.
    # This seems reasonable.
    
    # Let's add a small regularization to prevent numerical issues
    composite_score = np.maximum(ei_score, exploration_score)
    
    # Check time
    if time.time() > deadline:
        # If we are running out of time, just return the EI score which is already computed
        # or the composite if it's fast. The computation above is vectorized and fast.
        # The main cost is in the caller's GP evaluation, not here.
        pass
        
    # Ensure finite values
    composite_score = np.nan_to_num(composite_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return composite_score
# EVOLVE-BLOCK-END
