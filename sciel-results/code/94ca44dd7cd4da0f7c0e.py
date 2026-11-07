import numpy as np
import time
import math

# Global deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # 300s limit per evaluation, but we want to be safe.
# The task says "Each evaluation must finish within 300 s".
# We are inside the acquisition function. We should return quickly.
# However, the inspiration mentions a deadline guard. 
# The acquisition function is called 2000 times per step? No, "evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one".
# So acquisition is called ONCE with 2000 candidates.
# It needs to return 2000 scores.
# It should be fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Expected Improvement (EI)
    and Kriging Distance (KD).
    
    Strategy:
    - Exploitation: Expected Improvement (EI) based on the GP posterior.
    - Exploration: Kriging Distance (KD), which is essentially the normalized
      predictive standard deviation. This encourages visiting regions where
      the model is uncertain.
    - Cost Awareness: Divide the combined score by the evaluation cost to
      prioritize cheap evaluations that offer high potential gain.
    
    The blend is dynamic: early on, we might explore more; later, exploit more.
    However, since we don't have explicit step count, we can use the magnitude
    of f_best or just a fixed robust blend. Given the "stagnation" signal,
    a robust exploration term is key.
    
    We use a weighted sum: Score = w_ei * EI + w_kd * KD
    Then normalize by cost.
    """
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For Gaussian: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Where Phi is CDF and phi is PDF of standard normal.
    
    # Optimization: Use vectorized operations
    # Handle sigma == 0 case
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Use scipy.stats.norm for accuracy, or approximate?
    # SciPy is available. Let's use it for robustness.
    from scipy.stats import norm
    
    # EI calculation
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Kriging Distance (KD)
    # KD is typically the normalized sigma. 
    # To make it comparable to EI (which is in objective units), we might need scaling.
    # However, EI can be large if f_best is large. 
    # A common approach is to add a small multiple of sigma directly to the UCB,
    # or use a separate exploration term.
    # Let's use a weighted sum. 
    # We need to scale KD so it's comparable to EI.
    # A heuristic: scale KD by the current range of f_best or just use a fixed weight.
    # Since f_best starts high and decreases, EI decreases.
    # Let's try: Score = EI + beta * sigma
    # This is effectively UCB. The current program uses UCB (beta=1.0) divided by cost.
    # The inspiration suggests "blends EI with KD". 
    # Maybe the current UCB is too aggressive or not enough exploration?
    # Or maybe the cost normalization is hurting?
    
    # Let's try a different structure: 
    # The "Kriging Distance" is often defined as sigma / max(sigma). 
    # Or just sigma.
    # Let's stick to EI + exploration_bonus.
    
    # Dynamic Beta: 
    # If we are early, f_best is likely far from min. 
    # Let's use a slightly higher beta to encourage exploration more than the current 1.0.
    # Or, use the "Expected Regret" which is similar.
    
    # Let's try a specific structure from the literature for cost-aware BO:
    # Maximize (EI(x)) / Cost(x)
    # But add an exploration term that is NOT divided by cost? 
    # No, cost matters everywhere.
    
    # Let's look at the failure of the previous attempt (score 0.39 vs 0.92).
    # The previous attempt (p9) used KD. It scored lower. 
    # The current program (p6) uses UCB/cost. Score 0.92.
    # Inspiration 1 (score 1.22) used "hybrid acquisition that blends EI with KD".
    
    # Let's implement the hybrid carefully.
    # Score = (EI + lambda * sigma) / cost
    # The current program is effectively (f_best - mu + 1.0*sigma) / cost.
    # Note: EI <= f_best - mu + sigma (roughly, when mu is close to f_best).
    # Actually, EI is more precise. 
    # Let's use EI instead of the linear UCB approximation.
    # And let's adjust the exploration weight.
    
    # If p9 (pure KD or heavy KD) failed, maybe KD needs to be secondary.
    # Let's use EI as the primary term and add a small KD term.
    
    # Scale EI and KD to be comparable.
    # EI is in objective units. Sigma is in objective units.
    # They are directly comparable.
    
    # Let's try: Score = (EI + 0.5 * sigma) / cost
    # The current program uses 1.0 * sigma (in UCB form).
    # EI is generally smaller than UCB's exploration part when uncertainty is high?
    # Actually, UCB = f_best - (mu - beta*sigma) = f_best - mu + beta*sigma.
    # EI is the expected max improvement.
    # For large sigma, EI ~ sigma * 0.8 (approx).
    # So EI is roughly 0.8 * sigma.
    # The current program uses 1.0 * sigma.
    # So the current program is slightly more exploratory than pure EI.
    
    # To improve, we might need to:
    # 1. Use the exact EI (more accurate exploitation).
    # 2. Add a separate exploration term that doesn't depend on f_best?
    # 3. Or adjust the cost sensitivity.
    
    # Let's try: Score = EI / cost + gamma * (sigma / cost)
    # This is equivalent to (EI + gamma * sigma) / cost.
    # If we set gamma = 0.5, we get (EI + 0.5 sigma) / cost.
    # Since EI ~ 0.8 sigma, total exploration weight is ~ 1.3 sigma.
    # This is MORE exploratory than the current 1.0 sigma.
    # Is more exploration better? The inspiration 1 (1.22) suggests yes, or at least that hybrid is better.
    
    # Let's implement EI + 0.5 * sigma.
    
    # Optimization for speed:
    # We call norm.cdf and norm.pdf on 2000 elements. This is fast.
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Exploration Term
    # Use a slightly higher exploration weight to encourage finding new basins.
    # Let's use 0.5 * sigma added to EI.
    exploration_weight = 0.5
    combined_score = ei + exploration_weight * sigma
    
    # Cost Aware
    score = combined_score / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
