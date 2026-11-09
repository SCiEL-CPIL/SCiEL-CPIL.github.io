# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to persist across calls within the same seed run
# We keep it minimal to avoid state leakage issues across different seeds if the module is reloaded or if state persists incorrectly.
# However, the task implies a loop, so state persistence is expected.
_acq_state = {
    'call_count': 0,
    'last_f_best': None,
    'rng': None
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
    
    # Ensure sigma is safe for division
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Initialize RNG if needed
    if _acq_state['rng'] is None:
        # Use a fixed seed for reproducibility within the same process run if possible, 
        # but since we don't know the external seed, we rely on the system RNG or a fixed one.
        # The task says "For each seed 0, 1 and 2", implying the evaluator manages seeds.
        # We should not hardcode a seed that makes all runs identical if we want to explore.
        # However, for Thompson Sampling, the randomness is part of the strategy.
        # We'll use a dedicated RNG instance.
        _acq_state['rng'] = np.random.default_rng()
        
    rng = _acq_state['rng']
    _acq_state['call_count'] += 1
    calls = _acq_state['call_count']
    
    # 1. Compute base acquisition functions
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) with a high beta for exploration
    # We use a time-adaptive beta for the exploration component
    # Early: high beta (explore), Late: low beta (exploit)
    # Total calls ~25. 
    # Let's define a schedule for the exploration weight.
    
    # Strategy: Two-Stage Selection
    # Stage 1: Coarse Ranking using a Randomized Blend (Thompson Sampling on Acquisition)
    # We sample a random mixing parameter alpha from a Beta distribution or Uniform.
    # Let's sample alpha ~ Uniform(0, 1) but biased towards exploitation as we progress.
    # Early on, we want more exploration. Late on, we want more exploitation.
    # Progress p = calls / 25.
    # If p < 0.5, alpha is biased towards exploration (high weight on LCB).
    # If p > 0.5, alpha is biased towards exploitation (high weight on EI).
    
    # Actually, a cleaner TS approach is to sample the parameters of the acquisition function.
    # For EI, the parameter is f_best (which is fixed).
    # For LCB, the parameter is beta.
    # Let's sample beta from a distribution that shifts over time.
    # Early: Beta ~ Uniform(1, 5)
    # Late: Beta ~ Uniform(0.1, 1.0)
    
    p = min(calls / 25.0, 1.0)
    
    # Interpolate the range of beta
    beta_min = 0.1
    beta_max_early = 5.0
    beta_max_late = 1.0
    
    # Linear interpolation of the max beta
    beta_max = beta_max_early + (beta_max_late - beta_max_early) * p
    
    # Sample beta for this step
    beta_sample = rng.uniform(beta_min, beta_max)
    
    # Compute LCB score with this sampled beta
    lcb = beta_sample * sigma_safe - mu
    
    # Normalize both scores to [0, 1] to make them comparable for ranking
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_norm = normalize(ei)
    lcb_norm = normalize(lcb)
    
    # Stage 1: Select top K candidates based on a random blend
    # The blend weight w is also sampled, or we can just pick the max of the two normalized scores?
    # A common TS approach for multiple acq functions is to sample the function index.
    # Here, we can sample a weight w ~ Uniform(0,1) and compute score = w*EI + (1-w)*LCB.
    # This effectively samples from the Pareto front of the two acq functions.
    
    w = rng.uniform(0.0, 1.0)
    # Bias w towards exploitation later?
    # If p is high, we want w to be close to 1.
    # Let's use a Beta distribution for w.
    # alpha_w = 1 + 5*p, beta_w = 1
    # This makes w concentrate near 1 as p increases.
    alpha_w = 1.0 + 5.0 * p
    beta_w = 1.0
    w = rng.beta(alpha_w, beta_w)
    
    scores_coarse = w * ei_norm + (1.0 - w) * lcb_norm
    
    # Identify top K candidates
    K = 50
    if n > K:
        # Get indices of top K
        top_k_indices = np.argpartition(scores_coarse, -K)[-K:]
    else:
        top_k_indices = np.arange(n)
        
    # Stage 2: Deterministic Local Refinement on Top K
    # For these top K candidates, we don't use the random sample anymore.
    # We use a deterministic, high-quality acquisition to rank them.
    # We can use the original EI (which is theoretically sound for minimization) 
    # or a very tight LCB (beta=0.5) to exploit the uncertainty in this small set.
    
    # Let's use a hybrid: 0.5 * EI + 0.5 * LCB(beta=0.5) for the top K.
    # This balances the "promise" of EI with the "potential" of LCB in the final selection.
    
    # Compute LCB with a fixed, low beta for exploitation
    beta_exploit = 0.5
    lcb_exploit = beta_exploit * sigma_safe - mu
    lcb_exploit_norm = normalize(lcb_exploit)
    
    # Final scores for top K
    # We re-normalize EI and LCB_exploit within the top K set? No, global normalization is fine.
    # But to ensure we pick the best among the top K, we can just sum the normalized scores.
    
    scores_refined = 0.5 * ei_norm + 0.5 * lcb_exploit_norm
    
    # Create the final score array, filled with 0 (or -inf) for non-selected candidates
    # We must return 2000 finite scores.
    # We set non-top-K scores to a very low value so they are never picked.
    final_scores = np.full(n, -1e9, dtype=np.float64)
    
    # Assign the refined scores to the top K indices
    # Note: The evaluator picks the argmax of final_scores.
    # We want the argmax to be the best one in top_k_indices.
    # The values in scores_refined are relative.
    # We should scale them to be positive and larger than the -1e9 fill.
    # Since scores_refined is in [0, 1] (roughly), we can just use it.
    # But to be safe, let's shift them up.
    
    # Actually, just assigning the normalized scores is fine as long as they are > -1e9.
    # scores_refined is in [0, 1].
    final_scores[top_k_indices] = scores_refined[top_k_indices]
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=-1e9, posinf=1.0, neginf=-1e9)
    
    # Time check
    if time.time() > deadline:
        # If we are running out of time, just return what we have.
        # The computation is O(N log N) due to argpartition, which is fast for N=2000.
        pass
        
    return final_scores

# EVOLVE-BLOCK-END
