# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard to ensure we don't exceed time limits
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        return np.zeros_like(mu, dtype=float)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    f_best = float(f_best)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Theoretical GP-UCB exploration parameter
    # beta_t = sqrt(2 * t * log(1 + t))
    # We scale this to match the magnitude of the fixed beta=5.0 that worked well.
    # t is the number of acquisition calls (proxy for iterations).
    t = max(_call_count, 1)
    
    # The raw theoretical beta grows slowly. 
    # For t=1, sqrt(2*1*log(2)) ~ 1.17
    # For t=25, sqrt(2*25*log(26)) ~ sqrt(50*3.25) ~ 12.7
    # We want to maintain a strong exploration pressure similar to beta=5.0 
    # but modulated by the theoretical form.
    # Let's normalize or scale it. 
    # If we just use the raw value, it starts at ~1.17 which might be too low 
    # compared to the successful 5.0.
    # Let's scale it by a factor. 5.0 / 1.17 ~ 4.27.
    # Or simply: beta_t = 5.0 * (theoretical_t / theoretical_1)
    # theoretical_1 = sqrt(2*log(2))
    
    beta_theoretical_1 = np.sqrt(2.0 * np.log(2.0))
    beta_theoretical_t = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # Scale so that at t=1, beta is 5.0
    beta_t = 5.0 * (beta_theoretical_t / beta_theoretical_1)
    
    # Compute UCB score: -mu + beta * sigma
    ucb_score = -mu + beta_t * sigma_safe
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be on a similar scale to UCB or just add it?
    # The directive suggests combining additively.
    # EI is in objective units. UCB is in objective units (since mu, sigma are).
    # However, EI is bounded by (f_best - min_possible) which is f_best.
    # UCB can be negative or positive.
    # To avoid EI dominating if f_best is large, we might need to normalize.
    # But f_best decreases. 
    # Let's just add them directly as per "combine it additively".
    # To prevent EI from being too strong early on (when f_best is large), 
    # we can scale EI by a small factor or rely on the fact that 
    # UCB's exploration term (beta*sigma) is also significant.
    
    # Let's try a weighted sum. 
    # If we just do ucb + ei, the EI term might pull towards exploitation 
    # too hard if f_best is large.
    # However, the "strongest idea" from inspirations was the pure LCB/UCB with high beta.
    # The directive asks to combine with normalized EI.
    
    # Normalize EI by the current sigma or f_best?
    # "Normalized Expected Improvement" usually means dividing by sigma or similar.
    # Let's use EI / sigma_safe which is roughly the standardized improvement.
    # This makes the term scale-invariant w.r.t local uncertainty.
    # ei_normalized = ei / sigma_safe
    # This term is bounded roughly by max(z*Phi(z) + phi(z)) which is around 0.8.
    # So it's a small additive term.
    # Let's use a weight of 1.0 for the normalized EI.
    
    ei_normalized = ei / sigma_safe
    
    score = ucb_score + ei_normalized
    
    # Ensure the output is finite to prevent invalid results
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
