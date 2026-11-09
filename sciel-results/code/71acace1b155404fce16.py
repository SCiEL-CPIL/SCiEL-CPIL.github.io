"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state
_acq_call_count = 0
_start_time = time.time()
TIME_LIMIT = 300.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    global _acq_call_count, _start_time, TIME_LIMIT
    
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Return pure exploitation (minimize mu) as a safe fallback
        return -mu.copy()
    
    _acq_call_count += 1
    
    # The evaluator runs 25 optimization steps per seed (30 total evaluations, 5 initial + 25 acquisition steps)
    # We need to determine which seed we're in and what step within that seed
    # Total calls so far: _acq_call_count (1-indexed after increment)
    # Each seed has 25 acquisition calls
    seed_idx = (_acq_call_count - 1) // 25
    step_in_seed = (_acq_call_count - 1) % 25
    
    # Progress within the current seed (0 to 1)
    progress = (step_in_seed + 1) / 25.0  # 1/25, 2/25, ..., 1
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw a single sample from the posterior predictive
    # This naturally balances exploration and exploitation
    sample = mu - sigma_safe * np.random.standard_normal(len(mu))
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Time-adaptive blending:
    # Early in optimization: rely more on Thompson Sampling (exploration)
    # Late in optimization: rely more on EI (exploitation)
    
    # Thompson score: lower sample value is better, so negate for "higher means evaluate next"
    ts_score = -sample
    
    # Normalize both to have comparable scales
    ts_min, ts_max = np.min(ts_score), np.max(ts_score)
    ei_min, ei_max = np.min(ei), np.max(ei)
    
    ts_range = ts_max - ts_min + 1e-9
    ei_range = ei_max - ei_min + 1e-9
    
    ts_norm = (ts_score - ts_min) / ts_range
    ei_norm = (ei - ei_min) / ei_range
    
    # Blend weights: start with more Thompson (exploration), end with more EI (exploitation)
    # progress goes from 1/25 to 1
    w_ts = 0.7 * (1.0 - progress) + 0.3  # Starts at ~0.69, ends at 0.3
    w_ei = 1.0 - w_ts  # Starts at ~0.31, ends at 0.7
    
    score = w_ts * ts_norm + w_ei * ei_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
