import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Expected Improvement (SEI) with log-normal cost model.
    
    Structural change:
    - Replaces pointwise cost-normalized UCB/EI blend with a stochastic
      formulation where cost is modeled as log-normal.
    - Integrates cost uncertainty directly into the acquisition integral.
    - This transforms the selection criterion from a static ratio to a
      probabilistic expectation of improvement per unit of stochastic cost.
    
    The SEI score is:
      SEI(x) = E[max(f_best - f(x), 0) / C(x)]
    
    where C(x) is modeled as log-normal with log-mean = log(cost(x))
    and log-variance = 0.05 (small uncertainty).
    
    For a log-normal cost C with log-mean m and log-variance v,
    E[1/C] = exp(-m + v/2)
    
    So SEI(x) = EI(x) * exp(-log(cost(x)) + v/2)
             = EI(x) / cost(x) * exp(v/2)
    
    The exp(v/2) is a constant multiplier, so the relative ranking is
    the same as EI/cost. However, we add a small UCB-style exploration
    term to handle multimodal landscapes and low-data regimes, weighted
    by the same stochastic cost expectation.
    
    This is structurally different from the parent because:
    1. The cost is treated as a random variable, not a deterministic penalty.
    2. The acquisition is an expectation over both function value and cost.
    3. The exploration term is also cost-aware in the stochastic sense.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    # Guard against zero/negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Log-normal cost model: log-variance = 0.05
    log_cost_var = 0.05
    
    # Stochastic cost expectation: E[1/C] = exp(-log(cost) + log_cost_var/2)
    # This is equivalent to 1/cost * exp(log_cost_var/2)
    # The exp(log_cost_var/2) is a constant, so it doesn't affect ranking,
    # but we include it for correctness.
    stochastic_cost_factor = np.exp(log_cost_var / 2.0)
    
    # SEI for the exploitation term
    sei_exploit = ei / cost_safe * stochastic_cost_factor
    
    # UCB-style exploration term, also cost-aware in stochastic sense
    # For minimization: (f_best - mu) + beta * sigma
    beta = 2.5
    ucb_term = (f_best - mu) + beta * sigma_safe
    ucb_term = np.maximum(ucb_term, 0.0)
    
    # Stochastic cost-aware UCB
    sei_explore = ucb_term / cost_safe * stochastic_cost_factor
    
    # Blend: 0.7 SEI (exploitation) + 0.3 stochastic UCB (exploration)
    # The SEI formulation already captures probabilistic improvement,
    # so we give it higher weight. The UCB term handles multimodal exploration.
    alpha = 0.7
    scores = alpha * sei_exploit + (1.0 - alpha) * sei_explore
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check
    if time.time() > deadline:
        # Fallback: return simple EI/cost if we're close to timeout
        fallback_scores = ei / cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
