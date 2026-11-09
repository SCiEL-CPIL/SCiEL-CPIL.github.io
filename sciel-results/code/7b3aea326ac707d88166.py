import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Use a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    # to better balance exploration and exploitation for the 1D sphere function.
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # UCB = mu - kappa * sigma, where kappa controls the exploration-exploitation tradeoff
    # For 1D sphere, a moderate kappa works well
    kappa = 1.5
    ucb = mu - kappa * sigma
    
    # Normalize both EI and UCB to be on a similar scale
    # EI is already in objective units, UCB is in objective units but we want to combine them
    # Use a weighted combination: higher weight on EI when close to f_best, higher on UCB when far
    
    # Compute a normalized version of UCB that's comparable to EI
    # When f_best - mu is large (far from best), UCB should dominate
    # When f_best - mu is small (close to best), EI should dominate
    
    # Simple approach: weighted sum with fixed weights
    # EI tends to be small when mu is close to f_best, UCB tends to be negative (good) when mu is low
    # We want higher scores to mean "evaluate next"
    
    # Alternative: use the maximum of normalized EI and UCB
    # Or: use a blend that adapts based on the distance from f_best
    
    # Let's use a different approach: compute EI and UCB, then take a convex combination
    # where the weight on UCB increases as we move away from f_best
    
    # Distance from f_best in standard deviation units
    distance_std = np.abs(f_best - mu) / sigma
    
    # Weight for UCB: higher when distance is large
    # Use a sigmoid-like function: w_ucb = 1 / (1 + exp(-distance_std + threshold))
    # When distance_std is large, w_ucb -> 1; when small, w_ucb -> 0
    threshold = 1.0
    w_ucb = 1.0 / (1.0 + np.exp(-distance_std + threshold))
    w_ei = 1.0 - w_ucb
    
    # Combine: score = w_ei * EI + w_ucb * (-UCB) since we want higher to be better
    # Note: UCB = mu - kappa*sigma, so -UCB = -mu + kappa*sigma
    # But we want higher score to mean "more promising", so we should use -UCB or flip it
    # Actually, for minimization, lower UCB is better, so we want to maximize -UCB
    # But EI is already "improvement" so higher EI is better
    
    # Let's normalize both to be in a similar range
    # EI is typically in [0, f_best - min_mu]
    # -UCB = -mu + kappa*sigma, which can be negative or positive
    
    # Simpler approach: just use EI with a small exploration bonus
    # Or: use the standard EI which already balances exploration and exploitation
    
    # Actually, let's try a different structural change: use Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is more exploratory than EI
    
    pi = norm.cdf(z)
    
    # Or use a hybrid: max(EI, alpha * PI) or EI + beta * PI
    # Let's try EI with a slight boost for uncertain regions
    
    # Final approach: use EI but with a modified z that accounts for the 1D sphere structure
    # The sphere function is x^2, so we know the minimum is at 0
    # We can bias the acquisition towards the center of the domain
    
    # Since we don't have coordinates, we can't directly bias towards center
    # But we can use the fact that for x^2, the function is symmetric and convex
    
    # Let's stick with a well-tuned EI or try a different acquisition function
    # Use Knowledge Gradient (KG) approximation or just a tuned EI
    
    # For now, let's use a combination of EI and UCB with adaptive weights
    score = w_ei * ei + w_ucb * (f_best - ucb)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
