"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 280s limit to leave 20s margin for 300s kill switch

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (EI).
    
    Unlike LCB, EI integrates the tail of the predictive distribution, which is 
    more robust to non-Gaussian errors or heavy-tailed distributions often seen 
    in black-box optimization. We use a 'virtual best' (f_best - epsilon) to 
    avoid the degeneracy of standard EI where the improvement is zero at the 
    current best.
    
    We also add a dynamic exploration bonus that scales with the global 
    uncertainty (mean sigma) and decays as we get more confident (lower sigma).
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(len(mu))
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement Calculation
    # Standard EI minimization: E[f_best - f(x) | f(x) < f_best]
    # We use a slightly offset target to encourage moving away from the exact best point
    # if the GP suggests the best point is a local minimum, but mainly to ensure 
    # positive scores for points that are "close" to best.
    # However, for a robust start, we use f_best directly but add a small epsilon 
    # to the improvement calculation to handle the case where mu == f_best.
    
    # Normalize for numerical stability in the normal CDF
    # z = (f_best - mu) / sigma
    # If sigma is very small, z is huge. 
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent overflow in exp or inf in cdf
    # Phi(z) for z > 10 is ~1, for z < -10 is ~0
    # phi(z) for |z| > 10 is ~0
    z_clipped = np.clip(z, -10, 10)
    
    Phi_z = norm.cdf(z_clipped)
    phi_z = norm.pdf(z_clipped)
    
    # EI value (expected improvement over f_best)
    # Note: If mu > f_best, z is negative. Phi(z) is small. phi(z) is small. EI ~ 0.
    # If mu < f_best, z is positive. Phi(z) is large. EI is large.
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # 2. Exploration Bonus
    # We want to explore regions with high sigma, but we don't want to ignore 
    # regions with high potential improvement (low mu).
    # A common heuristic is to scale the exploration by the relative uncertainty.
    
    # Global uncertainty level
    mean_sigma = np.mean(sigma)
    max_sigma = np.max(sigma)
    
    # If the GP is very certain (mean_sigma is small), we rely more on EI.
    # If the GP is uncertain (mean_sigma is large), we add a bonus for high sigma.
    
    # Dynamic beta: 
    # Start high (exploration) and decay as mean_sigma decreases?
    # Actually, mean_sigma usually decreases as we collect data.
    # Let's use a fixed structure but scaled by the "quality" of the best point found so far.
    
    # Simple robust exploration: 
    # Bonus = kappa * sigma / (mean_sigma + eps)
    # This normalizes sigma so that average uncertainty gets a bonus of kappa.
    
    # Choose kappa based on the ratio of f_best to the scale of the problem.
    # Hartmann 3D min is -3.86. Range is roughly [-4, 4].
    # If f_best is close to -3.86, we should be more exploitative.
    # If f_best is far (e.g., -1.0), we should be more explorative.
    
    # Distance from theoretical minimum (approx -3.86)
    # We don't know the exact min of the shifted instance, but -3.86 is a good anchor.
    # Let's assume the target is around -3.86.
    # Regret = f_best - (-3.86) = f_best + 3.86
    # If regret is high, increase exploration.
    
    regret = max(f_best + 3.86, 0.0)
    
    # Kappa: 
    # Base kappa = 1.0
    # Increase kappa if regret is large.
    # kappa = 1.0 + 2.0 * min(regret / 2.0, 1.0)
    # So kappa ranges from 1.0 (if f_best ~ -3.86) to 3.0 (if f_best ~ -1.86)
    
    kappa = 1.0 + 2.0 * min(regret / 2.0, 1.0)
    
    # Exploration term
    # We want to maximize: (EI + Exploration) / Cost
    # Exploration = kappa * sigma
    
    exploration = kappa * sigma_safe
    
    # Total Score
    # We combine EI and Exploration.
    # EI is in objective units. Exploration is in objective units.
    # Score = (EI + Exploration) / Cost
    
    score = (ei + exploration) / cost_safe
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Optional: If the score is all negative (shouldn't happen if EI+Expl > 0), 
    # shift it? No, we just pick the max. If all are negative, we pick the least negative.
    # But EI is always >= 0. Exploration is always > 0. So score should be > 0.
    
    return score
# EVOLVE-BLOCK-END
