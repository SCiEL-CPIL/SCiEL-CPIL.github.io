"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (CA-EI) acquisition function.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) for minimization.
       EI(x) = E[max(f_best - f(x), 0)]
       For GP posterior N(mu, sigma):
       EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
       
    2. Incorporate Cost Awareness.
       The problem states cost = 1 + mean(x_scaled). Since we don't have x, 
       we are given `cost` directly. 
       We want to favor points that have high EI but low cost.
       A simple multiplicative penalty or additive penalty can be used.
       
       Given the budget is tight (30 units) and initial points are 5, 
       we have ~25 evaluations left. 
       Average cost is ~1.5 (if uniform in [0,1]^4, mean(x) is 0.5, cost is 1.5).
       Total budget 30 / 1.5 cost = 20 evals. We start with 5, so we have 15 left.
       
       We use a penalty factor that reduces the acquisition score for expensive points.
       Let normalized_cost = cost / max(cost). 
       Score = EI * (1 - lambda * normalized_cost)
       
       However, a more robust approach for cost-aware BO is to divide by cost 
       or use a penalty that reflects the "opportunity cost" of using budget.
       
       Let's try: Score = EI / (1 + cost)
       This heavily penalizes high cost.
       
       Alternatively, since cost is in [1, 2], we can just subtract a scaled cost.
       Score = EI - beta * cost
       
       Let's analyze the scale of EI. 
       f_best starts around the minimum of 5 random points. Shekel 4d min is -10.5.
       Random points in [0,10]^4 might have f ~ -5 to -10.
       So f_best ~ -8.
       mu can be lower, say -10.
       EI can be on the order of 1-2.
       Cost is 1-2.
       
       If we use EI / cost, the scale is preserved.
       If we use EI - cost, we might over-penalize.
       
       Let's use a hybrid: EI * exp(-alpha * cost) or EI / (1 + cost).
       EI / (1 + cost) seems reasonable. Cost range [1, 2] -> denominator [2, 3].
       This reduces score by factor of 2 to 3.
       
       To make it more sensitive, we can use:
       Score = EI / cost^gamma
       
       Let's try gamma = 1.0 first.
       
    3. Handle edge cases: sigma ~ 0.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive for log/phi calculations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0. We want to improve (lower f).
    # EI = (f_best - mu) * P(f < f_best) + sigma * pdf(z)
    # P(f < f_best) = Phi(z) where z = (f_best - mu) / sigma
    # Note: norm.cdf takes (x - mean)/std. 
    # Here f ~ N(mu, sigma). We want P(f < f_best).
    # Standardize: (f_best - mu) / sigma.
    
    z = (f_best - mu) / sigma_safe
    
    # Avoid overflow in exp for large z, but norm.cdf and norm.pdf handle it.
    # For very large z, Phi(z) ~ 1, phi(z) ~ 0.
    # For very small z (mu >> f_best), EI ~ 0.
    
    # Compute EI
    # term1 = (f_best - mu) * Phi(z)
    # term2 = sigma * phi(z)
    
    # Use scipy.stats.norm.cdf and norm.pdf
    # To ensure numerical stability and speed, we can vectorize.
    
    # Note: if f_best is very high (early on), and mu is low, z is large.
    # Phi(z) -> 1. EI -> f_best - mu.
    
    try:
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        
        ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        
        # Ensure EI is non-negative (numerical errors might make it slightly negative)
        ei = np.maximum(ei, 0.0)
        
        # Cost-aware adjustment
        # We want to maximize EI per unit of cost.
        # Score = EI / cost
        # This is a simple and effective heuristic for cost-aware BO.
        
        # Ensure cost is positive
        cost_safe = np.maximum(cost, 1e-9)
        
        scores = ei / cost_safe
        
        # Alternative: Penalize cost more aggressively?
        # scores = ei / (cost_safe ** 1.5)
        
    except Exception:
        # Fallback to pure EI if scipy fails or numerical issues arise
        z_clipped = np.clip(z, -10, 10)
        cdf_z = norm.cdf(z_clipped)
        pdf_z = norm.pdf(z_clipped)
        ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        scores = ei / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
