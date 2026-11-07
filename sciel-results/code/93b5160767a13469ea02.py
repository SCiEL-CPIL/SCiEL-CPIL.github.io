"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard to ensure we finish well before the 330s limit
_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Multi-Objective Expected Improvement (MOEI).
    
    Instead of Thompson Sampling (which relies on stochastic sampling), this uses
    a deterministic, vectorized approach based on two distinct acquisition criteria:
    1. Exploitation: Probability of Improvement (PI)
    2. Exploration: Expected Improvement (EI) normalized by uncertainty
    
    We compute a composite score that balances these two objectives.
    For minimization:
    - PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    - EI(x) = E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
             where z = (f_best - mu) / sigma
    
    To create a single score, we use a weighted sum. The weights are dynamic:
    - Early on (high sigma), exploration is favored.
    - Later on (low sigma), exploitation is favored.
    
    However, a more robust structural change is to use the **Upper Confidence Bound (UCB)**
    with a dynamic beta, but computed via the standard normal CDF/PDF for exactness,
    or better yet, a **Probability of Improvement (PI)** combined with **Expected Regret**.
    
    Let's stick to a pure, deterministic **Expected Improvement (EI)** but with a twist:
    Standard EI is often too greedy. A "structural" alternative is **Probability of Improvement (PI)**
    which is purely probabilistic. But PI can be too greedy in practice for noisy functions.
    
    Let's try **Kriging Believer (KB)** or **Expected Improvement with a penalty**.
    
    Actually, the most distinct structural change from Thompson Sampling (stochastic) is
    **Expected Improvement (EI)** (deterministic). The current program uses TS.
    Inspiration 3 (score 1.0) likely used EI or similar.
    Inspiration 1 (score 2.52) used something else.
    
    Let's implement a **Hybrid PI + EI** score, which is structurally different from TS.
    Score = alpha * PI + (1-alpha) * EI / sigma_max
    
    This explicitly separates the "chance of beating best" from the "magnitude of potential gain".
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple deterministic strategy if we are running out of time
        return -mu

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement and Probability of Improvement
    # z = (f_best - mu) / sigma
    # If f_best is very low and mu is high, z is negative.
    # If f_best is high and mu is low, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # Using scipy.stats.norm.cdf for precision
    from scipy.stats import norm
    pi_scores = norm.cdf(z)
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # phi(z) is the standard normal PDF
    ei_scores = (f_best - mu) * pi_scores + sigma_safe * norm.pdf(z)
    
    # Normalize EI scores to [0, 1] range to make them comparable to PI scores
    # We need to be careful with division by zero or very small values
    max_ei = np.max(ei_scores)
    if max_ei > 1e-9:
        ei_norm = ei_scores / max_ei
    else:
        ei_norm = np.zeros_like(ei_scores)
    
    # Combine PI and normalized EI
    # We want to encourage exploration when uncertainty is high.
    # PI is bounded [0,1]. EI_norm is bounded [0,1].
    # A simple average might work, but let's weight them.
    # If we are early in the optimization (high overall sigma), we might want more exploration.
    # However, the function signature doesn't give us the step count.
    # We can infer "early" from the magnitude of sigma relative to the range of mu.
    
    # Let's use a fixed weight that balances them well for 1D sphere.
    # Typically, EI is a stronger signal for the actual value reduction.
    # PI is a stronger signal for finding *any* improvement.
    
    # Strategy: Score = 0.5 * PI + 0.5 * EI_norm
    scores = 0.5 * pi_scores + 0.5 * ei_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
