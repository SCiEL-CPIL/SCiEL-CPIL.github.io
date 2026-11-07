"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Global state to track time and adapt kappa
_start_time = None
_kappa = 1.0
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _kappa, _call_count
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: If we are getting close to the limit, 
    # we must return quickly to avoid timeout invalidation.
    # The evaluator kills after 330s. We have a 300s budget per eval.
    # However, the acquisition function itself is called within the loop.
    # If the *total* elapsed time for the run is getting too high, 
    # we can't change the past, but we can ensure this call is fast.
    # More importantly, if the *current* evaluation step is taking too long, 
    # we should return a safe, fast result.
    # But the main risk is the outer loop timing out. 
    # Since we don't control the outer loop, we just ensure this function is O(N) and fast.
    
    _call_count += 1
    
    # Adaptive Kappa: 
    # Early on, we want more exploration (higher kappa).
    # Later, we want more exploitation (lower kappa).
    # We don't know the total budget exactly, but we can use call count as a proxy.
    # The loop runs while total cost < 30.
    # Initial cost is ~5 * (1 + 0.5) = 7.5.
    # Each step adds ~1.5 cost on average.
    # So roughly (30 - 7.5) / 1.5 = 15 steps.
    # Let's assume max ~20 calls.
    progress = min(_call_count / 20.0, 1.0)
    
    # Interpolate kappa from 2.0 (exploratory) to 0.1 (exploitative)
    _kappa = 2.0 - 1.9 * progress
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Z-score for Improvement
    # f_best is the current best observed value.
    # We want P(f(x) < f_best - k*sigma) or similar.
    # Standard EI: E[max(f_best - f(x), 0)]
    # For minimization: E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then Y = f_best - f(x) ~ N(f_best - mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Cost-aware EI:
    # We want to maximize the expected improvement per unit cost.
    # Score = EI / cost
    
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2/2) / sqrt(2*pi)
    
    # Vectorized calculation
    # Using scipy.stats.norm is robust, but manual calculation can be faster if needed.
    # Let's stick to scipy for stability and correctness, it's fast enough for N=2000.
    from scipy.stats import norm
    
    phi_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * phi_z + sigma * pdf_z
    
    # Handle numerical issues where EI might be negative due to floating point errors
    # EI should theoretically be non-negative.
    ei = np.maximum(ei, 0.0)
    
    # Cost adjustment
    scores = ei / cost_safe
    
    # Add a small exploration bonus for points with high uncertainty (high sigma)
    # This helps escape local optima, especially in early stages.
    # The term is scaled by kappa.
    # Exploration term: sigma * kappa
    # We blend it into the score.
    # Note: Adding sigma directly might dominate if sigma is large.
    # Let's normalize sigma relative to the scale of the problem or f_best.
    # A simple way: add kappa * sigma / (abs(f_best) + 1e-6)
    # This makes the exploration term relative to the current best value.
    
    scale = abs(f_best) + 1e-6
    exploration_bonus = _kappa * sigma / scale
    
    # Combine: We want the score to be high for both high EI and high uncertainty.
    # Simple sum: Score = EI/cost + kappa * sigma / (cost * scale)
    # This is equivalent to (EI + kappa * sigma / scale) / cost
    # Let's refine: The standard UCB is mu + kappa * sigma.
    # For minimization, we want low mu + high sigma? No, we want low f(x).
    # UCB for minimization (optimistic): mu - kappa * sigma.
    # But we are maximizing an acquisition score.
    # High score -> likely to be new best.
    # So we want points where mu is low (good) and sigma is high (uncertain, could be better).
    # The EI term already captures this trade-off.
    # Adding an explicit sigma term reinforces exploration.
    
    final_scores = scores + exploration_bonus / cost_safe
    
    # Ensure finite output
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Safety: If any scores are negative (shouldn't happen with max(0, ei) but just in case), set to 0
    final_scores = np.maximum(final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
