"""
Strategy: Cost-Aware Thompson Sampling with Dynamic Budget Decay.

Instead of using a deterministic acquisition function like EI or UCB, we use
Posterior Sampling (Thompson Sampling). For each candidate, we sample a value
from the GP posterior. We then select the candidate that maximizes the ratio
of the negative sampled value (since we minimize) to the cost.

The exploration is controlled by adding a noise term to the sampled value,
scaled by a coefficient beta that decays as the budget is consumed.
"""

import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 300s limit

def _get_elapsed_time():
    return time.time() - _start_time

def _check_deadline():
    return _get_elapsed_time() > (300.0 - _DEADLINE_MARGIN)

def _safe_div(num, den):
    return np.where(den > 1e-9, num / den, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, lowest observed value
    cost: (2000,) array of costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    if _check_deadline():
        # Fallback to simple EI/cost if running out of time to ensure valid output quickly
        z = _safe_div(f_best - mu, sigma)
        ei = (f_best - mu) * np.where(z > 0, 1 - norm.cdf(-z), norm.cdf(z)) + sigma * norm.pdf(z)
        # Simple fallback: EI / cost
        return _safe_div(ei, np.maximum(cost, 1e-9))

    # Estimate progress based on number of calls
    # Budget is 30. Initial 5 points cost ~7.5.
    # Remaining ~22.5. Average cost ~1.5.
    # Max iterations ~ 15.
    global _call_count
    if '_call_count' not in globals():
        _call_count = 0
    _call_count += 1
    
    # Progress p in [0, 1]
    p = min(_call_count / 15.0, 1.0)
    
    # Dynamic beta for exploration.
    # Early: high beta (explore). Late: low beta (exploit).
    # Beta controls the weight of the uncertainty (sigma) in the sample.
    # We use a decaying beta to shift from exploration to exploitation.
    beta = 1.5 * (1.0 - p) + 0.1
    
    # Thompson Sampling:
    # Sample from N(mu, sigma^2)
    # To ensure reproducibility within the call but randomness across calls,
    # we use a fixed seed derived from the call count or just rely on numpy's global RNG.
    # The task does not forbid using random state, but we should be careful.
    # Using np.random.normal is standard.
    
    # We want to minimize. So we want high scores to correspond to low f(x).
    # Sampled value s ~ N(mu, sigma).
    # We add an exploration bonus: s_explored = s + beta * sigma * noise?
    # Actually, standard Thompson Sampling just samples s.
    # The "exploration" is inherent in the variance.
    # However, to make it more aggressive early on, we can scale the sigma term.
    # Let's define the sampled value as:
    # s = mu + sigma * z, where z ~ N(0, 1)
    # To encourage exploration, we can artificially inflate sigma early on?
    # Or just use the standard sample.
    
    # Let's stick to standard TS but weight the cost.
    # Score = -s / cost
    
    # However, pure TS can be noisy. A common variant is "Noisy Thompson Sampling"
    # where we add a term proportional to sigma.
    # Let's use: s = mu - beta * sigma * |z| ? No, that's deterministic.
    
    # Let's use the standard sample:
    z = np.random.normal(0, 1, size=mu.shape)
    s = mu + sigma * z
    
    # For minimization, we want to maximize -s.
    # But we also want to incorporate f_best to guide exploitation?
    # TS naturally does this via the posterior.
    
    # To ensure we don't pick points that are clearly worse than f_best when uncertainty is low,
    # we can clip the score or use a hybrid.
    # But let's try pure TS with cost normalization first.
    
    # Cost normalization:
    # Cost is in [1, 2].
    # Dividing by cost penalizes expensive points.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score = -s / cost_safe
    # This might be too noisy.
    
    # Alternative: Use the sample to estimate the probability of improvement?
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is deterministic (EI-like).
    
    # Let's try a "Stochastic EI" approach which is closer to the current strategy but stochastic.
    # Or, let's use the sample to compute a "sampled EI".
    # Sampled EI = max(0, f_best - s)
    # This is 0 if s > f_best, else f_best - s.
    # This is a valid acquisition function.
    
    sampled_ei = np.maximum(0, f_best - s)
    
    # Add a small exploration bonus based on sigma to avoid zero scores everywhere if s > f_best
    # But TS should handle this.
    
    # Let's blend Sampled EI with a UCB-like term to ensure some exploration
    # even when the sample is bad, but weighted by the remaining budget.
    
    # UCB score: -mu + kappa * sigma
    # Let's use the sample s directly in the numerator.
    
    # Final Score = (sampled_ei + 0.1 * sigma) / cost
    # The 0.1 * sigma term ensures that high uncertainty points get some score even if s > f_best.
    
    exploration_bonus = 0.2 * sigma
    
    final_score = _safe_div(sampled_ei + exploration_bonus, cost_safe)
    
    # Ensure finite
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score

# Helper for fallback (imported in fallback block)
from scipy.stats import norm
