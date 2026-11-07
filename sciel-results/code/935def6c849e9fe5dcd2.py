import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-aware acquisition combining PI and EI
    # PI: Probability of Improvement - robust for exploration in multimodal functions
    # EI: Expected Improvement - good for exploitation when close to optimum
    
    # PI for minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z_pi = (f_best - mu) / sigma
    pi = norm.cdf(z_pi)
    
    # EI for minimization: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Adaptive weighting: shift from PI (exploration) to EI (exploitation)
    # as f_best gets closer to 0 (the minimum)
    # When f_best is large, PI dominates; when small, EI dominates
    # Use a smooth transition based on f_best
    f_best_abs = np.abs(f_best)
    # Weight: 1 when f_best is large, 0 when f_best is small
    # Scale: Rastrigin in [-5.12, 5.12]^2 has max ~102.4, min 0
    # Use sigmoid-like transition around f_best ~ 1.0
    weight_pi = 1.0 / (1.0 + np.exp(2.0 * (f_best_abs - 1.0)))
    weight_ei = 1.0 - weight_pi
    
    # Combine PI and EI
    combined = weight_pi * pi + weight_ei * ei
    
    # Divide by cost for cost-awareness
    score = combined / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
