"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return best observed (greedy) or zeros
        # Greedy is safe
        return -mu

    # Hybrid acquisition: Expected Improvement (EI) + Probability of Improvement (PI)
    # For minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # PI = Phi(z)
    # We want to maximize these.
    
    # Handle sigma == 0
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI and PI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    pi = norm.cdf(z)
    
    # Hybrid: 70% PI weight, 30% EI weight
    # This biases towards exploitation (PI is binary-like, EI is continuous)
    # But EI provides gradient information.
    # Let's try a blend.
    
    # Normalize EI to [0, 1] roughly? Or just mix?
    # PI is in [0, 1]. EI can be large.
    # Let's use a weighted sum.
    # If we just use EI, it's standard.
    # If we just use PI, it's greedy-ish but respects uncertainty.
    # The inspiration says "Hybrid EI+PI acquisition with 70% PI weight".
    # How to combine scales?
    # Maybe: score = w_pi * pi + w_ei * (ei / (f_best - min(mu)))?
    # Or simpler: score = pi * (1 - w) + ei * w? No, scales differ.
    
    # Let's try: score = pi + 0.1 * ei
    # Or: score = 0.7 * pi + 0.3 * (ei / (f_best + 1e-9)) ?
    
    # Actually, let's look at the "inspiration" more closely.
    # "Hybrid EI+PI acquisition with 70% PI weight biases towards exploitation"
    # This implies PI is the dominant term.
    # Let's try: score = 0.7 * pi + 0.3 * (ei / (abs(f_best) + 1e-9))
    # This normalizes EI somewhat.
    
    # Alternatively, just use PI?
    # PI is effectively: "Is this point likely to be better than f_best?"
    # For a unimodal function, this is very strong.
    
    # Let's try a simple blend:
    # score = pi + 0.01 * ei
    # This makes PI the primary driver, EI the tie-breaker.
    
    # Let's try:
    # score = 0.7 * pi + 0.3 * (ei / (f_best - mu + 1e-9)) # No, that's just pdf.
    
    # Let's stick to a standard hybrid form found in literature or intuition:
    # Often, people use EI. But if PI is better, maybe it's because EI over-explores?
    # No, EI is usually good.
    
    # Let's try the specific "70% PI" hint.
    # If we assume the "score" in the inspiration was the result, and it jumped to 2.151.
    # Let's try: score = 0.7 * pi + 0.3 * (ei / (f_best + 1e-6))
    
    # Actually, let's just try PI first? It's simpler.
    # But the hint says "Hybrid EI+PI".
    
    # Let's use:
    # score = pi + 0.1 * ei
    # This keeps PI as the main driver.
    
    scores = 0.7 * pi + 0.3 * (ei / (f_best + 1e-9))
    
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return scores
# EVOLVE-BLOCK-END
