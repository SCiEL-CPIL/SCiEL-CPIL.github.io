import numpy as np
from scipy.stats import norm
import time

# Global state to track time and potentially adapt parameters if needed across calls
# However, the task says "Write acquisition... for a loop", implying the function is called repeatedly.
# We can use a global variable to track the start time of the entire optimization run if we assume 
# the module is loaded once per run. But to be safe and stateless per call, we rely on the inputs.
# Actually, the prompt says "Each function runs on five seeded instances...". 
# The function `acquisition` is called in a loop. 
# We can estimate the "stage" of optimization by looking at f_best relative to the known bounds of Griewank.
# Griewank 2D min is 0. Max is roughly 1.
# If f_best is close to 0, we are in exploitation mode. If high, exploration.

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s buffer
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Vectorized calculation
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted EI
    # The cost structure is 1 + mean(x_scaled). 
    # We divide EI by cost to get "improvement per unit cost".
    cei = ei / cost_safe
    
    # Adaptive Exploration/Exploitation Balance:
    # Instead of a fixed beta or max-of, we modulate the effective EI.
    # A simple adaptive strategy: 
    # If f_best is still high (e.g., > 0.1), we might want to encourage exploration more.
    # However, CEI already handles this partially because high uncertainty (sigma) increases EI.
    # 
    # A more structural change: Use a "Lookahead" or "Confidence" based adjustment.
    # Let's try a simple adaptive temperature for the EI calculation itself?
    # Or better: Combine CEI with a "Probability of Best" (PoB) term that is weighted by progress.
    
    # Progress metric: 
    # Griewank 2D on [-10,10]^2. 
    # Initial random points often yield f ~ 0.1-0.5.
    # Optimum is 0.
    # Let's define a progress p in [0, 1].
    # If f_best > 0.5, p = 0 (early, explore).
    # If f_best < 0.01, p = 1 (late, exploit).
    # Linear interpolation: p = clip((0.5 - f_best) / (0.5 - 0.01), 0, 1)
    
    f_best_clipped = np.clip(f_best, 0.0, 0.5)
    progress = (0.5 - f_best_clipped) / 0.49  # Range 0 to 1
    
    # Early on (low progress), we trust EI less and might want to push towards lower cost or higher uncertainty?
    # Actually, CEI is generally robust. 
    # The previous strategy failed by taking the MAX of normalized scores. This creates a "winner-takes-all" 
    # that can be noisy. 
    # Let's stick to CEI but add a small, adaptive "novelty" or "uncertainty" bonus that decays as we progress.
    
    # Uncertainty bonus: sigma / cost
    # We want to explore where sigma is high, but pay less for it.
    unc_bonus = sigma_safe / cost_safe
    
    # Weight the uncertainty bonus. 
    # Early: weight = 0.5 (significant exploration)
    # Late: weight = 0.0 (pure exploitation via EI)
    w_unc = 0.5 * (1.0 - progress)
    
    final_scores = cei + w_unc * unc_bonus
    
    # Check time
    if time.time() > deadline:
         # Return a safe, deterministic score based on CEI only if we ran out of time?
         # Actually, we are well within time for 2000 points.
         pass
        
    # Normalize? 
    # The task says "Return 2000 finite scores (higher = evaluate next)".
    # It does NOT require normalization. The evaluator picks the argmax.
    # Normalizing can sometimes distort the relative magnitudes if the range is weird, 
    # but usually, raw scores are fine as long as they are finite.
    # However, if CEI values are very small (e.g., 1e-6), floating point precision might matter.
    # Let's ensure they are finite.
    
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # One more structural check:
    # Is there a better acquisition? 
    # "Cost-aware" implies we should be very sensitive to cost.
    # The cost function is linear in x. 
    # Points with lower x (in scaled coords) are cheaper.
    # My `cost` input is the cost.
    # 
    # Let's try a different family: **Thompson Sampling** approximation?
    # We don't have samples, only mu and sigma.
    # We can approximate the probability of being the best.
    # P(x is best) ~ Phi((f_best - mu) / sigma) is not quite right for continuous distributions.
    # 
    # Let's stick to the refined CEI + Adaptive Uncertainty.
    # The key structural difference from the parent is:
    # 1. No min-max normalization (preserves magnitude information).
    # 2. No max-of ensemble (avoids noisy "outliers" winning).
    # 3. Adaptive weighting of exploration based on optimization progress.
    
    return final_scores
