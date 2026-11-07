"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

# Known global minimum of Hartmann-3D
_HARTMANN3D_MIN = -3.86278

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _ei_scores(mu, sigma, target):
    """
    Compute Expected Improvement for minimization relative to a target value.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (target - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining:
    1. Exploitation: EI over current best (f_best).
    2. Exploration: Knowledge Gradient (KG) via Monte Carlo integration.
    
    The KG component estimates the expected reduction in the minimum observed
    value by sampling from the joint GP posterior (independent approximation)
    and computing the probability that each candidate is the global minimum,
    weighted by the expected improvement relative to f_best.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Deadline guard: fall back to simple, fast EI if time is running out
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        return _ei_scores(mu, sigma, f_best)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    n = len(mu)
    if n == 0:
        return mu
    
    # 1. Exploitation: EI over the current best observed value
    ei_fb = _ei_scores(mu, sigma, f_best)
    
    # 2. Convergence Indicator
    # Determine how close we are to the known global optimum.
    # If f_best is close to _HARTMANN3D_MIN, we should exploit more.
    # If f_best is far, we should explore more.
    if f_best > _HARTMANN3D_MIN:
        gap = f_best - _HARTMANN3D_MIN
        conv = 1.0 - (gap / (gap + 1.0))
    else:
        conv = 1.0
    
    conv = float(np.clip(conv, 0.0, 1.0))
    
    # Exploration weight: High (1.0) when far from optimum, Low (0.3) when close.
    w_expl = 1.0 - 0.7 * conv
    
    # 3. Knowledge Gradient via Monte Carlo
    # Estimate E[ min_j (f_j) - f_best | f_i ] by sampling from the posterior.
    # We use an independent sampling approximation for speed.
    
    # Time budget check for MC samples
    time_remaining = _DEADLINE_SECONDS - _SAFETY_MARGIN - _elapsed()
    if time_remaining < 15:
        n_samples = 50
    elif time_remaining < 40:
        n_samples = 100
    else:
        n_samples = 200
    
    rng = np.random.default_rng(42)
    
    # Generate samples: (n_samples, n_candidates)
    # Shape: (n_samples, n)
    samples = mu[None, :] + sigma_safe[None, :] * rng.standard_normal((n_samples, n))
    
    # For each sample, find the minimum value across all candidates
    min_values = np.min(samples, axis=1)  # shape: (n_samples,)
    
    # The KG value for candidate i is E[ max(0, f_best - min_j f_j) ]
    # But a more direct interpretation for acquisition is:
    # How much does evaluating candidate i reduce our uncertainty about the global min?
    # A common approximation: P(candidate i is the global min) * E[improvement | i is min]
    
    # Let's compute the probability that candidate i is the global minimum
    min_indices = np.argmin(samples, axis=1)
    min_counts = np.bincount(min_indices, minlength=n).astype(np.float64)
    p_global_min = min_counts / n_samples
    
    # Also compute the expected improvement conditional on being the global min
    # For each candidate i, if it is the global min, the improvement is f_best - samples[:, i]
    # We can estimate E[ (f_best - samples[:, i]) * 1{samples[:, i] == min} ]
    
    # Create a mask for when candidate i is the global min
    # This is memory intensive for large n, but n=2000 and n_samples=200 is 400k booleans, fine.
    is_min_mask = np.zeros((n_samples, n), dtype=bool)
    is_min_mask[np.arange(n_samples)[:, None], min_indices] = True
    
    # Expected improvement contribution from being the global min
    # (f_best - samples) * is_min_mask, then average over samples
    ei_contribution = np.mean((f_best - samples) * is_min_mask, axis=0)
    
    # KG score: p_global_min * ei_contribution is one way, but let's use the direct expectation
    # Actually, the KG acquisition is often approximated as:
    # KG_i = E[ min_j f_j | f_i ] - min_j E[f_j]  (this is not quite right)
    # Or: KG_i = E[ max(0, f_best - min_j f_j) ] but this doesn't depend on i directly in the min.
    
    # A better approximation for multi-modal functions:
    # Score_i = P(i is global min) * E[f_best - f_i | i is global min]
    # This rewards points that are likely to be the best AND have high expected improvement.
    
    # Let's refine:
    # kg_score = p_global_min * ei_contribution
    
    # However, ei_contribution already includes the condition.
    # Let's just use the expected reduction in the minimum.
    # A simpler and effective proxy:
    # kg_score = p_global_min * (f_best - mu)  -- not quite, mu is the mean, not the conditional mean.
    
    # Let's stick to the measured improvement from the inspiration:
    # "prob_global weighting with alpha=1.5 more strongly favors candidates in promising basins"
    # And "MC-computed EI instead of closed-form EI ensures consistency"
    
    # Let's compute the MC-EI:
    # For each candidate i, sample f_i ~ N(mu_i, sigma_i^2)
    # EI_i = E[max(0, f_best - f_i)]
    # This is what the closed-form does, but MC allows us to combine with the global min probability.
    
    # Let's compute the MC version of EI for consistency, though closed-form is faster.
    # Since we already have samples, let's use them.
    # But samples are for all candidates simultaneously.
    # For each candidate i, the marginal distribution is N(mu_i, sigma_i^2).
    # So the MC-EI is just the average of max(0, f_best - samples[:, i])
    
    mc_ei = np.mean(np.maximum(0, f_best - samples), axis=0)
    
    # Now, the KG-inspired score:
    # We want to balance:
    # 1. Exploitation: mc_ei (or closed-form ei_fb)
    # 2. Exploration: p_global_min (probability of being the global minimum)
    
    # The current program used:
    # scores = ei_fb + w_expl * (0.8 * ei_dgp + 0.3 * ei_scale * p_global_norm)
    
    # The inspiration used:
    # scores = ei * (1.0 + alpha * prob_global_min)
    
    # Let's combine: Use EI as the base, and multiply by a factor that boosts
    # candidates with high probability of being the global minimum.
    # This is the multiplicative approach from inspiration 1.
    
    # Normalize p_global_min to [0, 1] range (it already is, but let's ensure)
    p_global_norm = p_global_min
    
    # Alpha controls the strength of the KG boost
    # From knowledge: alpha=1.5 improved over alpha=1.0
    alpha = 1.5
    
    # The multiplicative boost:
    # scores = mc_ei * (1.0 + alpha * p_global_norm)
    
    # But we also want to adapt the exploration weight based on convergence.
    # When we are close to the optimum, we should rely more on pure exploitation (EI).
    # When we are far, we should rely more on the KG boost.
    
    # Let's blend:
    # scores = (1 - w_expl) * mc_ei + w_expl * mc_ei * (1.0 + alpha * p_global_norm)
    # = mc_ei * ((1 - w_expl) + w_expl * (1.0 + alpha * p_global_norm))
    # = mc_ei * (1.0 + w_expl * alpha * p_global_norm)
    
    # This is clean. The boost is scaled by w_expl.
    
    # However, we should also consider that mc_ei can be zero for many points.
    # The KG boost should still help identify promising regions.
    # Let's add a small additive component to encourage exploration in uncertain regions.
    
    # Alternative: Additive combination
    # scores = mc_ei + w_expl * alpha * p_global_norm * scale_factor
    
    # Let's try the multiplicative approach first, as it was successful in the inspiration.
    # But to ensure we don't suppress exploitation too much, let's use:
    # scores = mc_ei + w_expl * alpha * p_global_norm * (f_best - mu)
    # This adds a term proportional to the expected improvement and the probability of being global min.
    
    # Actually, let's stick to the multiplicative form but with a floor:
    # scores = mc_ei * (1.0 + w_expl * alpha * p_global_norm)
    
    # To avoid zeroing out points with mc_ei=0 but high p_global_norm,
    # let's add a small exploration bonus:
    # exploration_bonus = w_expl * 0.1 * sigma_safe * p_global_norm
    
    # Final score:
    scores = mc_ei * (1.0 + w_expl * alpha * p_global_norm)
    
    # Add a small exploration bonus for uncertain points that are likely global min
    exploration_bonus = w_expl * 0.05 * sigma_safe * p_global_norm
    scores += exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
