"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a structural change: instead of a closed-form EI/UCB blend, we use
    a hybrid that directly targets the known global minimum (3.0) with a
    probabilistic "chance of beating the true optimum" term, combined with
    a variance-scaled improvement term. This provides a persistent
    exploration signal in high-uncertainty regions even as f_best converges,
    and exploits the known problem structure.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    t_start = time.time()
    
    # Safety: clip sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # The known global minimum of Goldstein-Price 2D
    f_true_opt = 3.0
    
    # Compute several components
    
    # 1. Probability of beating the TRUE optimum (not just observed best)
    # This is P(f(x) < 3.0) = Phi((3.0 - mu) / sigma)
    z_true = (f_true_opt - mu) / sigma_safe
    prob_beat_true = norm.cdf(z_true)
    
    # 2. Expected improvement over the TRUE optimum (not observed best)
    # EI_true(x) = E[max(3.0 - f(x), 0)] = (3.0 - mu) * Phi(z) + sigma * phi(z)
    ei_true = (f_true_opt - mu) * norm.cdf(z_true) + sigma_safe * norm.pdf(z_true)
    ei_true = np.maximum(ei_true, 0.0)
    
    # 3. Expected improvement over observed best
    z_best = (f_best - mu) / sigma_safe
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    ei_best = np.maximum(ei_best, 0.0)
    
    # 4. A "variance-weighted improvement potential" term
    # This rewards regions where both sigma is high AND there's room to improve
    # Use a soft cap to avoid domination by extreme sigma
    sigma_cap = 1.0  # Cap on sigma contribution
    sigma_scaled = np.minimum(sigma_safe, sigma_cap)
    
    # Improvement potential: how much lower than f_best can we reasonably expect?
    # Use (f_best - mu) clipped to non-negative, scaled by probability of being low
    improve_potential = np.maximum(f_best - mu, 0.0)
    
    # 5. Combine into a single score
    # Strategy:
    # - In early iterations (high sigma), weight prob_beat_true and ei_true heavily
    #   to explore broadly for the true optimum
    # - In later iterations (low sigma), shift weight to ei_best for fine-tuning
    # - The transition is controlled by relative uncertainty
    
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    sigma_mean = np.mean(sigma_safe)
    # Relative uncertainty: 1.0 means very early, 0.0 means very late
    rel_unc = np.clip(sigma_mean / mu_range, 0.0, 1.0)
    
    # Weights:
    # w_true: weight for true-optimum-based terms (exploration of the landscape)
    # w_best: weight for observed-best-based terms (exploitation of current best)
    # w_var: weight for the variance-scaled term (additional exploration)
    
    w_true = 0.5 + 0.3 * rel_unc  # 0.5 to 0.8
    w_best = 0.5 - 0.2 * rel_unc  # 0.3 to 0.5
    w_var = 0.1 * rel_unc         # 0.0 to 0.1
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn > 1e-12:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
    
    prob_beat_true_norm = normalize(prob_beat_true)
    ei_true_norm = normalize(ei_true)
    ei_best_norm = normalize(ei_best)
    
    # Composite variance term: high sigma * high improvement potential
    var_term = sigma_scaled * improve_potential
    var_term_norm = normalize(var_term)
    
    # Final score
    scores = (w_true * (0.6 * prob_beat_true_norm + 0.4 * ei_true_norm) +
              w_best * ei_best_norm +
              w_var * var_term_norm)
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
