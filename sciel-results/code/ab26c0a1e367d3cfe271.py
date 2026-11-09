"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Explicit wall-clock deadline guard (evaluator kills at 330 s; keep margin).
_DEADLINE = time.time() + 325.0

def acquisition(mu, sigma, f_best):
    # Time guard: if we are too close to the deadline, fall back to pure exploitation
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=float)
        return -mu
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Robustify sigma: clip tiny values and drop non-finite entries
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.clip(sigma, 1e-9, None)
    
    # Strategy: Adaptive UCB with Diversity Penalty
    
    # 1. Exploitation Component:
    # We want to find points where the mean is significantly lower than f_best.
    # However, simple -mu is too greedy. 
    # Let's use a normalized improvement score.
    # If mu is much lower than f_best, the score should be high.
    # Let's define a "promising" score based on how far below f_best the mean is.
    # But we need to be careful with scale.
    
    # 2. Exploration Component:
    # Standard UCB: alpha * sigma.
    # However, for Ackley, which is multimodal, we want to explore *different* regions.
    # A common trick is to use the posterior variance to identify "unknown" regions.
    # But since we don't have coordinates, we can't do spatial diversity directly.
    # We can, however, use the distribution of sigma itself to modulate exploration.
    
    # Let's try a different structural change: 
    # Instead of a fixed beta, let's use a "Bayesian Confidence Bound" that adapts to the 
    # uncertainty relative to the best found value.
    
    # A key insight for Ackley: The function has many local minima. 
    # Pure exploitation gets stuck. Pure exploration is slow.
    # 
    # Let's implement a "Weighted EI" where the weight depends on the confidence.
    # Or better: Let's use a "Mean + Lambda * Std" approach but with a dynamic Lambda.
    
    # Dynamic Lambda:
    # If sigma is large everywhere, we are in early stages -> explore more.
    # If sigma is small everywhere, we are in late stages -> exploit more.
    
    # Calculate a global uncertainty measure
    mean_sigma = np.mean(sigma)
    
    # Normalize sigma to get a relative uncertainty
    # If mean_sigma is very small, we should reduce exploration.
    # Let's scale sigma by a factor that decays as we get more confident.
    
    # A simple adaptive beta:
    # beta = 1.0 + 2.0 * exp(-mean_sigma * 100) 
    # This isn't quite right because we don't know the scale of f.
    
    # Let's look at the ratio of sigma to the range of mu.
    mu_range = np.ptp(mu)
    if mu_range < 1e-9:
        mu_range = 1e-9
        
    # Relative uncertainty
    rel_uncertainty = mean_sigma / (mu_range + 1e-9)
    
    # If relative uncertainty is high, explore more.
    # If relative uncertainty is low, exploit more.
    
    # Let's use a UCB with a dynamic beta.
    # beta = 1.5 if we are exploring, 0.5 if we are exploiting.
    
    # We can estimate the "stage" by looking at how much f_best has improved? 
    # We don't have history. 
    # But we have f_best.
    
    # Let's try a different structural approach: 
    # **Expected Improvement with a "Lookahead" approximation**
    # Instead of just EI, let's consider the probability of finding a point that is 
    # significantly better than f_best.
    
    # P(Y < f_best - delta) where delta is a small improvement.
    # This is equivalent to the CDF of the improvement.
    
    # Let's stick to a robust UCB variant but with a twist:
    # Use the *median* of the sigma distribution to scale the exploration term,
    # making it robust to outliers in the posterior variance.
    
    median_sigma = np.median(sigma)
    
    # If the median sigma is high, the model is uncertain -> explore.
    # If the median sigma is low, the model is confident -> exploit.
    
    # We define the score as:
    # Score = -mu + beta * sigma
    
    # To make it adaptive, let beta depend on the "quality" of the current best.
    # If f_best is very low (close to 0 for Ackley), we should be cautious.
    # If f_best is high, we can afford more exploration.
    
    # However, without knowing the scale of Ackley (which is roughly [0, ~10]),
    # we can use a heuristic.
    
    # Let's try a **Thompson Sampling** approximation?
    # No, we return scores, not samples.
    
    # Let's try **Probability of Improvement (POI)**?
    # POI = P(Y < f_best) = Phi((f_best - mu) / sigma)
    # POI is very greedy and often gets stuck in local minima.
    
    # Let's try **Expected Improvement (EI)** again, but with a "floor".
    # The previous program used Quantile EI.
    # Standard EI: (f_best - mu) * Phi(d) + sigma * phi(d)
    
    # The problem with EI is that it can be dominated by a single point with very high 
    # uncertainty if the mean is not good.
    
    # Let's use a **Hybrid Score**:
    # Score = w1 * EI + w2 * UCB
    
    # Or, simpler: **Adaptive UCB**
    # Let's set beta based on the fraction of points that have sigma > some threshold.
    
    # Actually, let's look at the inspiration: "diversity-aware acquisition that combines 
    # a UCB exploration term with a negative density penalty".
    # We don't have coordinates, so we can't do spatial density.
    # BUT, we can use the *indices* of the candidates? No, they are random.
    
    # What if we use the **rank** of the mu values?
    # Points with low mu (good) should be prioritized.
    # Points with high sigma (uncertain) should be prioritized.
    
    # Let's try a **Normalized UCB**:
    # Normalize mu and sigma to [0, 1] range.
    
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    if mu_max - mu_min < 1e-9:
        mu_norm = np.zeros_like(mu)
    else:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
        
    if sigma_max - sigma_min < 1e-9:
        sigma_norm = np.zeros_like(sigma)
    else:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
        
    # Score = (1 - mu_norm) + beta * sigma_norm
    # (1 - mu_norm) is high when mu is low (good).
    # sigma_norm is high when sigma is high (uncertain).
    
    # This normalizes the scales, making the choice of beta more stable.
    
    # Let's choose beta = 1.0 for a balanced start.
    # But we can make it adaptive.
    
    # If the best mu is very close to the worst mu, the model is confident -> lower beta.
    # If the spread is large, the model is uncertain -> higher beta.
    
    mu_spread = mu_max - mu_min
    # If mu_spread is small, the model has converged on a region -> exploit.
    # If mu_spread is large, the model is unsure -> explore.
    
    # Let's map mu_spread to beta.
    # Assume a "typical" spread for Ackley is around 1.0 (scaled).
    # If mu_spread < 0.1, beta = 0.5
    # If mu_spread > 1.0, beta = 2.0
    
    # Linear interpolation
    # beta = 0.5 + 1.5 * min(1.0, mu_spread)
    
    # This is a structural change: **Normalized Adaptive UCB**.
    
    beta = 0.5 + 1.5 * min(1.0, mu_spread)
    
    score = (1.0 - mu_norm) + beta * sigma_norm
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
