# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to persist across calls within the same seed run
_acq_state = {
    'call_count': 0,
    'last_f_best': None,
    'f_best_history': [],
    'rng_seed_counter': 0
}

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    deadline = t_start + 295.0  # Safety margin for 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is safe
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Update state
    _acq_state['call_count'] += 1
    calls = _acq_state['call_count']
    
    # Determine the exploration-exploitation trade-off parameter (kappa)
    # For Thompson Sampling on the objective (minimization), we sample:
    # y_i ~ N(mu_i, sigma_i^2)
    # We want to minimize y_i.
    # However, standard TS for minimization often uses:
    # score_i = mu_i - kappa * sigma_i  (This is LCB, not TS)
    # True TS samples the function value.
    # y_i = mu_i + sigma_i * z_i, where z_i ~ N(0, 1)
    # We pick the argmin of y_i.
    
    # To implement this via a "score" vector where higher is better:
    # We want the index of argmin(y) to have the highest score.
    # So, score_i = -y_i = -mu_i - sigma_i * z_i
    
    # We need to generate z_i for all 2000 candidates.
    # This is computationally cheap (2000 draws).
    
    # The key insight for TS in BO:
    # If we simply sample y_i and pick the min, we are doing pure exploration 
    # early on (high sigma dominates) and exploitation later (sigma shrinks).
    # However, TS can be "too" exploratory if not tuned.
    # Often, a "discounted" TS or a mix is used.
    
    # Let's stick to pure Thompson Sampling first, as it is a distinct structural change
    # from the deterministic EI/LCB blend.
    
    # Generate standard normal random variables for each candidate
    # We use a deterministic seed based on call count to ensure reproducibility 
    # if the evaluator re-runs, though the evaluator likely provides fresh randoms 
    # or expects stochastic behavior. 
    # Wait, the task says "draws 2000 fresh uniform candidates". It doesn't say the 
    # acquisition function itself must be stochastic. But TS *is* stochastic.
    # If we use a fixed seed, it becomes deterministic. 
    # Using np.random.default_rng() without a seed uses the global state or OS entropy.
    # This is fine.
    
    rng = np.random.default_rng()
    z = rng.standard_normal(n)
    
    # Sampled objective values (lower is better)
    sampled_y = mu + sigma_safe * z
    
    # Score: higher is better. So we invert the sampled objective.
    scores = -sampled_y
    
    # Optional: Add a small regularization or bias towards f_best?
    # Pure TS minimizes the sampled value. 
    # If f_best is very low, and mu is high, sigma might be low, so -sampled_y is very negative.
    # If mu is close to f_best, and sigma is small, -sampled_y is close to -f_best.
    # This naturally prioritizes points where the sampled value is low.
    
    # One potential issue: TS can get stuck if the GP posterior is overconfident 
    # in a bad region. But the GP is updated by the evaluator.
    
    # Let's add a very small amount of "optimism" or "pessimism" if we want to 
    # guide it slightly, but pure TS is the cleanest structural change.
    # Actually, for minimization, pure TS (minimize sampled value) is equivalent to 
    # maximizing -sampled value.
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    # If any scores are -inf, they won't be picked (good).
    # If any are inf, they will be picked (bad). 
    # Since mu and sigma are finite, and z is finite, scores should be finite.
    # But just in case of numerical overflow:
    if not np.all(np.isfinite(scores)):
        # Fallback to EI-like score if something went wrong
        z_ei = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
        scores = ei
        
    # Time check
    if time.time() > deadline:
        pass # Return what we have
        
    return scores

# EVOLVE-BLOCK-END
