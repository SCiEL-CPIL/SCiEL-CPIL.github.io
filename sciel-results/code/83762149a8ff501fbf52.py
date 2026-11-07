"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t
import time as _time

_DEADLINE_S = 245.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (np.maximum(cost, 1.0) + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, 1.0)

    # --- Student-t Expected Improvement (t-EI) for minimization ---
    # Using a Student-t distribution for the posterior predictive is more
    # robust than Gaussian EI, especially when the GP is uncertain or
    # the objective is non-Gaussian.
    # nu=4 is a common choice for robustness without being too heavy-tailed.
    nu = 4.0
    
    delta = f_best - mu
    z = delta / sigma_safe
    # Clip z to avoid numerical issues in the CDF/PDF for extreme values
    z = np.clip(z, -50.0, 50.0)
    
    # Calculate t-CDF and t-PDF
    t_cdf = student_t.cdf(z, nu)
    t_pdf = student_t.pdf(z, nu)
    
    # The expected improvement for a Student-t variable Y ~ t_nu(mu, sigma)
    # E[max(f_best - Y, 0)] = delta * (1 - F_t(z)) + sigma * f_t(z) * (nu + z^2) / (nu - 1)
    # Note: The standard t-EI formula often includes a correction factor for the variance
    # if sigma is interpreted as standard deviation vs scale parameter. 
    # Here we assume sigma is the standard deviation of the predictive distribution.
    # The PDF of a t-distribution with scale sigma is f(y) = (1/sigma) * t_pdf((y-mu)/sigma, nu).
    # The term (nu + z^2)/(nu - 1) arises from the specific integral of the t-distribution tail.
    
    correction_factor = (nu + z**2) / (nu - 1)
    ei_t = delta * (1.0 - t_cdf) + sigma_safe * t_pdf * correction_factor
    
    # Ensure EI is non-negative (numerical safety)
    ei_t = np.maximum(ei_t, 0.0)
    ei_t = np.nan_to_num(ei_t, nan=0.0, posinf=1e6, neginf=0.0)

    # --- Cost-aware Scaling ---
    # Instead of linear division (1/cost), use a power law to soften the penalty.
    # cost is in [1, 2]. 
    # alpha = 0.7 provides a moderate penalty: a point with cost 2 is penalized by 2^0.7 ~ 1.62,
    # rather than 2.0. This allows higher-cost points to be selected if they offer significantly
    # higher expected improvement.
    alpha = 0.7
    cost_weight = cost_safe ** alpha
    
    score_ei = ei_t / cost_weight

    # --- Adaptive Exploration Bonus ---
    # Add a small bonus for high uncertainty to encourage global exploration,
    # especially in early stages or when the model is uncertain.
    # This helps escape local optima more effectively than pure EI.
    # We scale the bonus by the inverse cost to maintain cost-awareness.
    
    # Normalize sigma to a 0-1 range based on typical GP output scales
    # The GP is standardized, so sigma is in objective units.
    # We use a simple heuristic: if sigma is large relative to the current best improvement potential, boost it.
    # A fixed small weight is safer to avoid destabilizing the search.
    exploration_weight = 0.1
    exploration_score = sigma_safe / cost_weight
    
    # Blend: primarily EI, with a small exploration component
    # We can also modulate the exploration weight based on the 'age' of the search,
    # but since we don't have explicit step count, we rely on the cost/uncertainty balance.
    score = score_ei + exploration_weight * exploration_score

    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
