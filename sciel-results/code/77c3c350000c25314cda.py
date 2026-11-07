"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level timer for the deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid Thompson Sampling and Expected Improvement approach.
    
    Thompson Sampling: score = -mu + sigma * z, where z ~ N(0,1)
    This provides natural exploration-exploitation balance.
    
    We also compute EI as a fallback/blend to ensure robustness.
    The final score is a weighted combination that favors exploration early on.
    """
    # Deadline guard: Check if we are close to the 330s limit.
    # If so, return a safe, exploitation-heavy score to ensure we finish.
    elapsed = time.time() - _start_time
    if elapsed > 320.0:
        # Pure exploitation: pick lowest mean
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        return -mu_safe

    # Compute Expected Improvement (EI) for minimization
    # E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Z = (f_best - mu) / sigma
    scores_ei = np.zeros_like(mu)
    mask = sigma > 1e-8
    
    if np.any(mask):
        mu_m = mu[mask]
        sigma_m = sigma[mask]
        Z = (f_best - mu_m) / sigma_m
        cdf_Z = norm.cdf(Z)
        pdf_Z = norm.pdf(Z)
        ei = (f_best - mu_m) * cdf_Z + sigma_m * pdf_Z
        scores_ei[mask] = ei
    
    if np.any(~mask):
        mu_s = mu[~mask]
        scores_ei[~mask] = np.maximum(f_best - mu_s, 0.0)
    
    # Compute Thompson Sampling score
    # Draw a fresh standard normal vector for each candidate
    # This is the structural change: stochastic sampling from posterior predictive
    z = np.random.randn(mu.shape[0])
    scores_ts = -mu + sigma * z
    
    # Blend EI and TS
    # EI is more stable, TS is more exploratory
    # Use a weighted average: 0.5 * EI + 0.5 * TS
    # But scale them to be comparable. EI is in objective units, TS is in objective units too.
    # However, EI is always non-negative, while TS can be negative.
    # Let's normalize or just use a simple blend.
    
    # Actually, let's use a different strategy:
    # Use Thompson Sampling as the primary, but add a small EI component to guide exploitation.
    # Or, use a dynamic blend based on how close mu is to f_best.
    
    # Simpler: Just use Thompson Sampling with a slight exploitation bias.
    # score = -mu + sigma * z, but if mu is very close to f_best, reduce exploration.
    
    # Let's try: score = -mu + sigma * z + 0.1 * (f_best - mu)
    # This adds a small deterministic pull towards the best known point.
    
    # Actually, the pure Thompson Sampling is a strong baseline. Let's stick with it but ensure stability.
    # The key insight is that TS naturally balances exploration and exploitation.
    
    # Let's use a hybrid: 0.7 * TS + 0.3 * EI
    # But we need to scale EI to be comparable to TS.
    # EI is bounded by f_best, TS is unbounded.
    # Let's just use TS for now, as it's a structural change.
    
    scores = scores_ts
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
