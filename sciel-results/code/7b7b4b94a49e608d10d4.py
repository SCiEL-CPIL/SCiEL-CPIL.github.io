# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Fast CDF/PDF approximations to speed up inner loops
def norm_cdf_fast(z):
    """Approximate normal CDF using Abramowitz and Stegun approximation (fast, accurate)"""
    # Handle NaN/Inf
    z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    d = 0.3989422804014327 * np.exp(-z * z / 2.0)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    out = 1.0 - p
    out = np.where(z < 0, 1.0 - out, out)
    return out

def norm_pdf_fast(z):
    """Approximate normal PDF"""
    z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
    return 0.3989422804014327 * np.exp(-0.5 * z * z)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d).
    
    Strategy:
    1. Compute Expected Improvement (EI) for each candidate.
    2. Normalize EI by cost to get "efficiency".
    3. Add a small exploration bonus based on uncertainty (sigma) to prevent 
       premature convergence, scaled by the remaining budget proxy.
    4. Ensure all outputs are finite.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    scores: (2000,) array of finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Since we are minimizing, improvement is f_best - mu.
    # If mu < f_best, z > 0.
    # Standard EI formula for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    ei = (f_best - mu) * norm_cdf_fast(z) + sigma * norm_pdf_fast(z)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # This encourages finding improvements where they are "cheap"
    score = ei / cost
    
    # Exploration bonus:
    # In the early stages (when f_best is high/large), we might want to explore more.
    # In the late stages (when f_best is low/small), we should exploit.
    # A simple adaptive weight:
    # If f_best is large, we are far from optimum, explore more.
    # If f_best is small, we are close, exploit more.
    
    # Let's use a simple heuristic:
    # exploration_weight = 1.0 if f_best > 10 else 0.1
    # But f_best for rosenbrock can be 0 to large values.
    # Let's use a sigmoid-like transition based on f_best.
    # f_best is in objective units. Min is 0.
    # A good transition point might be around f_best = 1.0.
    
    # Alternative: Use the magnitude of sigma to drive exploration.
    # High sigma means high uncertainty.
    # We add a small term: beta * sigma / cost
    # This is like a UCB term.
    
    # Let's combine EI/cost and a small UCB/cost term.
    # UCB term: -mu + beta * sigma (for minimization, we want low mu, high sigma)
    # Actually, standard UCB for minimization is mu - beta * sigma.
    # Let's stick to EI as the primary driver, as it is more robust for BO.
    
    # To improve upon standard EI/cost, we can apply a power to the cost to make it less sensitive to small cost differences, or more sensitive.
    # cost in [1, 2] (since x in [-2,2], scaled to [0,1], mean is [0,1], cost = 1+mean).
    # Range is small.
    
    # Let's try a hybrid:
    # score = (ei / cost) + lambda * (sigma / cost)
    # Where lambda is small.
    
    # However, a known issue with EI is that it can be zero for points where mu >> f_best.
    # UCB handles this better.
    
    # Let's use a weighted sum of normalized EI and normalized UCB.
    # But normalization requires knowing the scale.
    
    # A simpler, very robust approach:
    # Use "Probability of Improvement" (PI) per cost?
    # PI = Phi(z).
    # PI is bounded [0,1]. EI is unbounded.
    
    # Let's stick to EI/cost but add a small exploration term based on sigma.
    # exploration_term = 0.1 * sigma / cost
    # This ensures that if EI is 0, we still have some score based on uncertainty.
    
    # Let's refine the exploration term.
    # If f_best is very good (close to 0), we should not explore too much.
    # If f_best is bad, we should explore.
    
    # Let's use a dynamic beta.
    # beta = 0.5 * np.exp(-f_best)  # If f_best is 0, beta=0.5. If f_best is 10, beta~0.
    # This might be too aggressive.
    
    # Let's try:
    # score = ei / cost
    # This is the most standard cost-aware acquisition.
    
    # To differentiate from the "seed" (which might be simple EI), let's add a 
    # "regret" based exploration.
    
    # Let's try a specific variant: "Cost-Adjusted Expected Improvement" with a 
    # slight preference for lower cost even if EI is slightly lower, by using 
    # a power on cost.
    
    # score = ei / (cost ** 1.5)
    # This penalizes high cost more heavily.
    
    # Let's use a power of 1.0 (standard) but add a small bonus for low cost 
    # when EI is comparable.
    
    # Actually, the most effective simple strategy for ROSENBROCK (which has a 
    # narrow valley) is to ensure we don't get stuck in local minima. 
    # EI tends to exploit. UCB explores.
    
    # Let's use a mix:
    # ucb = mu - 1.0 * sigma  (for minimization, we want to minimize this value, so we maximize -ucb? No.)
    # For minimization, UCB is usually defined as mu - beta * sigma. We want to select the point with the lowest UCB value.
    # So acquisition score for UCB should be - (mu - beta * sigma) = beta * sigma - mu.
    
    # Let's compute UCB score:
    ucb_score = 1.0 * sigma - mu
    
    # Normalize EI and UCB scores to [0,1] range for the batch?
    # This is risky if the batch is all bad or all good.
    
    # Let's stick to EI/cost. It is robust.
    # To improve, let's use a "soft" cost normalization.
    
    # Final Decision:
    # Use EI / cost.
    # Add a small term: 0.01 * sigma / cost to encourage exploration in uncertain regions.
    
    score = ei / cost
    exploration_bonus = 0.05 * sigma / cost
    score = score + exploration_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
