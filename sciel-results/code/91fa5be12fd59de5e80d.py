import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Knowledge Gradient (KG) approximation
    # 
    # The KG value of evaluating x is:
    # E[ max(f_best, f(x)) - min(f_best, f(x)) ] = E[ f_best - min(f_best, f(x)) ]
    # = E[ f_best - f(x) * I(f(x) < f_best) ]
    # 
    # A common approximation for minimization:
    # KG ≈ (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # This is actually the same as Expected Improvement (EI).
    # 
    # However, the directive asks for a KG that decouples "discovery probability" 
    # from "value magnitude". A better approximation for multi-modal landscapes:
    # 
    # KG ≈ P(f(x) < f_best) * E[f_best - f(x) | f(x) < f_best]
    #    = Phi(z) * (f_best - mu)   [when f(x) ~ N(mu, sigma^2)]
    # 
    # But this ignores the sigma contribution. A more complete form:
    # The exact EI is: (f_best - mu)*Phi(z) + sigma*phi(z)
    # 
    # For KG, we can use a variant that emphasizes exploration in high-uncertainty 
    # regions where the probability of finding a new minimum is high.
    # 
    # Let's use a hybrid:
    # 1. Compute EI (exploitation + some exploration)
    # 2. Add an exploration bonus based on uncertainty relative to the current best
    # 3. Normalize by cost
    
    # Standard EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of improvement (discovery probability)
    p_improve = norm.cdf(z)
    
    # Expected reduction if improved (value magnitude)
    # E[f_best - f(x) | f(x) < f_best] = (f_best - mu) + sigma * phi(z)/Phi(z)
    # But we need to handle Phi(z) -> 0
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    cdf_z_safe = np.where(cdf_z > 1e-10, cdf_z, 1e-10)
    expected_reduction = (f_best - mu) + sigma * phi_z / cdf_z_safe
    
    # KG approximation: probability of improvement * expected reduction
    kg = p_improve * expected_reduction
    
    # The standard EI is actually equal to:
    # (f_best - mu)*Phi(z) + sigma*phi(z)
    # = Phi(z) * [(f_best - mu) + sigma*phi(z)/Phi(z)]
    # = p_improve * expected_reduction
    # So KG ≈ EI in this Gaussian case.
    # 
    # To make it structurally different and better for multi-modal landscapes,
    # let's add an explicit exploration term that rewards high uncertainty
    # when the expected improvement is low (i.e., we're not clearly better).
    
    # Exploration bonus: reward regions with high sigma where we're uncertain
    # about whether there's a new minimum
    # Scale by 1/(1 + |f_best - mu|/sigma) to dampen when we're far from f_best
    exploration_scale = sigma / (1.0 + np.abs(f_best - mu) / sigma)
    
    # Combine KG (exploitation) with exploration bonus
    # The weight on exploration should be moderate to avoid pure random search
    exploration_weight = 0.3
    score = kg + exploration_weight * exploration_scale
    
    # Cost normalization
    # Cost ranges from 1 to 2 (since x is in [0,1]^2, mean(x) in [0,1], cost = 1 + mean(x))
    # We want to slightly prefer cheaper evaluations, but not too aggressively
    # because the minimum is in the inner 80% box where cost is moderate
    
    # Normalize cost to [0, 1] range for better scaling
    cost_norm = (cost - 1.0) / 1.0  # cost - 1, ranges [0, 1]
    
    # Apply cost penalty: divide by (1 + penalty * cost_norm)
    # penalty = 0.5 means cost of 2 reduces score by ~33%
    cost_penalty = 0.5
    score = score / (1.0 + cost_penalty * cost_norm)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
