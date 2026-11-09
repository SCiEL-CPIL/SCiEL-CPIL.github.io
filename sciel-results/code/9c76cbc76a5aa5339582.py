import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 300s limit, start timer
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    score_ei = ei / cost_safe

    # Structural change: Use probability of significant improvement as exploration signal
    # P(improvement > threshold) where threshold scales with current regret
    # This is more principled than raw uncertainty because it's tied to actual improvement potential
    threshold = 0.1 * max(f_best, 0.1)  # threshold for "significant" improvement
    z_thresh = (f_best - threshold - mu) / sigma_safe
    p_sig = norm.cdf(z_thresh)  # P(f(x) < f_best - threshold)
    
    # Combine: cost-aware EI + weighted probability of significant improvement
    # The weight decays as f_best approaches 0 (less need for aggressive exploration)
    w = 1.0 / (1.0 + np.exp((f_best - 2.0) / 2.0))
    
    # Scale p_sig to be comparable to score_ei
    ei_max = np.max(score_ei)
    if ei_max < 1e-12:
        ei_max = 1.0
    p_sig_scaled = w * 0.5 * ei_max * p_sig / cost_safe

    score = score_ei + p_sig_scaled

    # Time check
    if time.time() > deadline:
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
