# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a hybrid of Probability of Improvement (PI) and Expected Improvement (EI)
    with an adaptive exploration weight that decays as f_best improves.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Probability of Improvement (PI)
    pi = norm.cdf(z)

    # Expected Improvement (EI)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Adaptive weight for PI: more exploration when f_best is high (early), less when low (late)
    # f_best ranges from ~25 down to 0
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    # alpha starts at 1.0 (equal weight) and decays to 0.1
    alpha = 0.1 + 0.9 * ratio

    # Hybrid score: blend EI and PI
    # EI is unbounded, PI is [0,1]. We need to scale them.
    # A simple approach: score = ei + alpha * pi * (ei + 1)
    # Or just: score = ei + alpha * pi
    # Let's try: score = ei + alpha * pi * (1 + ei) to make PI more impactful when EI is high
    # Actually, simpler: score = ei + alpha * pi
    # But EI can be large. Let's normalize EI by f_best? No, f_best changes.
    # Let's stick to the inspiration 1 structure but with adaptive alpha.
    
    # Alternative: Use UCB as the base, but the directive says PI.
    # Let's combine PI and EI as suggested in inspiration 1, but with adaptive weight.
    
    score = ei + alpha * pi
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple exploitation score (negative mean)
        return -mu
    
    return score
# EVOLVE-BLOCK-END
