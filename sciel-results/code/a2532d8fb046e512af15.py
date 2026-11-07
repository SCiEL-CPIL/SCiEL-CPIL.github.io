"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time
    global _call_count
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are extremely close to the limit, return a safe fallback
    # In practice, this loop runs in < 1ms, so this is just a safety net.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback: return -mu (exploit current best) to ensure finite output
        return -np.asarray(mu)

    _call_count += 1
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robustness: ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Hybrid acquisition: Combine LCB (exploration) and EI (exploitation)
    # LCB scores: -mu + kappa * sigma
    # EI scores: (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Use an adaptive kappa that decreases over time to shift from exploration to exploitation
    # Start with higher exploration (kappa=2.0) and decay to lower (kappa=0.5)
    # The decay is based on the call count, which corresponds to the number of evaluations
    # Since there are 30 total evaluations, we can map _call_count to a range.
    # However, _call_count starts at 1.
    # Let's use a simple linear decay or a fixed value that worked well.
    # The current program with kappa=1.0 scored 1.42.
    # Let's try a combination.
    
    # Compute LCB component
    # A common adaptive strategy is kappa = c * sqrt(log(n)) or similar.
    # For simplicity and robustness, let's try a blend.
    # score = alpha * LCB + (1 - alpha) * EI
    
    # Let's try to dynamically adjust the weight.
    # Early on, we want more exploration (LCB). Later, more exploitation (EI).
    # But Branin is multi-modal, so we might need sustained exploration.
    # The inspiration suggests LCB alone was better (1.42 vs 1.0).
    # Let's stick closer to LCB but refine the kappa.
    # Maybe a slightly higher kappa helps find the other minima?
    # Or perhaps a hybrid where EI handles the local search better once we are close?
    
    # Let's try: score = -mu + kappa * sigma
    # with kappa = 1.5?
    # Or let's try the hybrid idea properly.
    
    # Compute EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB (higher is better for exploration)
    # LCB value is mu - kappa*sigma. We want to minimize it, so we maximize -LCB = -mu + kappa*sigma.
    # Let's use kappa = 1.0 as in the parent, but maybe blend with EI?
    # If we just use LCB, we got 1.42.
    # If we use EI, we got 1.0.
    # Let's try a weighted sum.
    # Normalize them? No, they have different scales.
    # Let's try: score = -mu + 1.0 * sigma_safe  (This is the parent)
    # Let's try: score = ei + 0.5 * sigma_safe  (Add some extra exploration on top of EI?)
    # Or: score = 0.5 * (-mu + 1.0 * sigma_safe) + 0.5 * ei
    
    # Actually, the LCB score of 1.42 is quite good.
    # Let's investigate if a slightly different kappa helps.
    # Branin has 3 global minima.
    # Maybe kappa = 1.2?
    
    # Let's try a simple refinement: Adaptive Kappa.
    # kappa = 2.0 * (1 - _call_count / 30.0) + 0.5
    # At call 1: kappa ~ 1.93
    # At call 30: kappa ~ 0.5
    # This encourages broad search early, local search later.
    
    # Let's test this hypothesis.
    # If _call_count is small, kappa is high.
    # Note: _call_count increments every time acquisition is called.
    # The loop calls it once per evaluation step (after drawing 2000 candidates).
    # So _call_count goes from 1 to 25 (30 total evals - 5 initial = 25 acquisition calls).
    
    # Let's use a decay from 2.0 to 1.0.
    # kappa = 2.0 - (1.0 * (_call_count - 1) / 24.0)
    
    # Or simpler: just use kappa = 1.5 constant?
    # The parent used 1.0.
    # Let's try 1.5.
    
    kappa = 1.5
    scores = -mu + kappa * sigma_safe
    
    # Ensure all values are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
