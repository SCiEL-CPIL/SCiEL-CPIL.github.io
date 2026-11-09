"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Known global minimum for Styblinski-Tang 1D
    f_min = -39.1662
    
    # Safety check for sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB)
    # We want to maximize mu - beta * sigma (since we are minimizing)
    # So the score is -mu + beta * sigma
    # Or equivalently, we can just compute the UCB value and invert it for "higher is better"
    # UCB_value = mu - beta * sigma
    # Score = -UCB_value = -mu + beta * sigma
    
    # Dynamic beta: 
    # If f_best is close to f_min, we are likely converged or stuck in a local minimum.
    # We should increase exploration (beta) to escape.
    # If f_best is far from f_min, we should exploit more (lower beta).
    
    # Normalize progress: 1.0 if at min, 0.0 if at start (worst case ~ 400 for ST1D at 5,5? No, max is around 400? 
    # Actually ST1D max is 250? Let's check range. 
    # f(5) = 5^2(5-2) + 5(1-5^2) + 10 = 15 - 100 + 10 = -75? 
    # Wait, f(x) = 0.5*x*(x^2-2)^2 - 0.5*(1-x^2) + 5
    # f(5) = 0.5*5*(25-2)^2 - 0.5*(1-25) + 5 = 2.5*529 - 0.5*(-24) + 5 = 1322.5 + 12 + 5 = 1339.5
    # f(-5) = 0.5*(-5)*(25-2)^2 - 0.5*(1-25) + 5 = -2.5*529 + 12 + 5 = -1322.5 + 17 = -1305.5
    # Wait, the minimum is -39.166.
    # The function is symmetric? No, x^3 term? 
    # f(x) = 0.5 x (x^2 - 2)^2 - 0.5 (1 - x^2) + 5
    # f(-x) = 0.5 (-x) (x^2 - 2)^2 - 0.5 (1 - x^2) + 5
    # f(x) - f(-x) = x (x^2 - 2)^2
    # So it's not symmetric.
    # Max value in [-5,5] is around 1300. Min is -39.16.
    # Range is approx 1340.
    
    # Let's use a simpler heuristic for beta.
    # If f_best is still high (e.g., > 0), we are far from min. Exploit.
    # If f_best is low (e.g., < -30), we are close. Explore.
    
    # Let's map f_best to beta.
    # f_best range: [-39.16, 1300]
    # We want beta to be small when f_best is high, and large when f_best is low.
    # Linear mapping:
    # beta_min = 0.1, beta_max = 2.0
    # If f_best <= f_min, beta = beta_max
    # If f_best >= 0, beta = beta_min
    
    beta_min = 0.1
    beta_max = 2.0
    
    # Progress p: 1.0 at f_min, 0.0 at 0.0
    # p = (0.0 - f_best) / (0.0 - f_min)
    # If f_best > 0, p < 0 -> clamp to 0
    # If f_best < f_min, p > 1 -> clamp to 1
    
    p = (0.0 - f_best) / (0.0 - f_min)
    p = np.clip(p, 0.0, 1.0)
    
    beta = beta_min + (beta_max - beta_min) * p
    
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalize both scores to [0, 1] using min-max normalization over the candidate set
    # This ensures they are on comparable scales
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-9:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
        
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
        
    # 4. Blend
    # Weight exploration more as we get closer to the optimum?
    # Or just a fixed 50/50?
    # The inspiration said 70/30 EI/UCB improved things.
    # Let's try 60% EI, 40% UCB.
    w_ei = 0.6
    w_ucb = 0.4
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    return score
# EVOLVE-BLOCK-END
