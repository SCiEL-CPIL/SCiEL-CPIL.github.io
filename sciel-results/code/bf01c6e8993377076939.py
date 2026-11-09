"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t
import time


def acquisition(mu, sigma, f_best):
    """
    Student-t Process (STP) inspired acquisition for Beale 2D minimization.
    
    Uses a scaled Student-t distribution (df=3) to model the predictive
    distribution, yielding heavier tails for epistemic uncertainty.
    Combines Expected Improvement under the t-distribution with an
    uncertainty-aware exploration term.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()
    
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() - t0 > 320.0:
        mu_arr = np.asarray(mu, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        fallback = -(mu_arr + 0.1 * sigma_arr)
        return np.nan_to_num(fallback, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best + 1.0)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-10)

    # Student-t degrees of freedom
    df = 3.0

    # For a Student-t distribution with df degrees of freedom, the variance is df/(df-2).
    # To match the GP's sigma (standard deviation), we scale: sigma_t = sigma * sqrt((df-2)/df)
    # This way, the t-distribution has the same standard deviation as the GP posterior.
    scale_t = sigma_safe * np.sqrt((df - 2.0) / df)

    # --- Expected Improvement under Student-t ---
    # For minimization, EI = E[max(f_best - Y, 0)] where Y ~ t(df, mu, scale_t)
    # We can compute this numerically or use a closed-form approximation.
    # Using numerical integration via the t-distribution's CDF and PDF.
    
    # Shift and scale: let Z = (Y - mu) / scale_t, so Z ~ t(df)
    # EI = E[max(f_best - (mu + scale_t * Z), 0)]
    #    = E[max(f_best - mu - scale_t * Z, 0)]
    # Let a = f_best - mu, so EI = E[max(a - scale_t * Z, 0)]
    
    a = f_best - mu
    
    # For the t-distribution, we can use the relationship:
    # E[max(a - b*Z, 0)] where Z ~ t(df)
    # This can be computed as:
    # If a - b*z > 0, i.e., z < a/b (when b > 0)
    # = integral from -inf to a/b of (a - b*z) * t_pdf(z; df) dz
    
    # We'll use a numerical approach with the t-distribution
    # For efficiency, we'll use a closed-form approximation based on the normal case
    # but adjusted for heavier tails.
    
    # Actually, let's use a simpler approach: compute EI using the t-distribution
    # by leveraging the fact that for large df it approaches normal, but for df=3
    # we can use a correction factor.
    
    # Standard EI for normal: EI = a * Phi(a/sigma) + sigma * phi(a/sigma)
    # For t-distribution, we use a similar form but with t CDF and PDF.
    
    # z_t = a / scale_t
    z_t = a / scale_t
    
    # t CDF and PDF at z_t
    t_cdf_val = student_t.cdf(z_t, df)
    t_pdf_val = student_t.pdf(z_t, df)
    
    # EI approximation for t-distribution:
    # EI = a * t_cdf(z_t) + scale_t * t_pdf(z_t) * (df / (df - 2)) * (1 - z_t * t_cdf(z_t) / t_pdf(z_t))
    # This is getting complex. Let's use a simpler heuristic:
    # The t-distribution has heavier tails, so EI should be larger than normal EI
    # when a/sigma is moderate, but similar when a/sigma is large.
    
    # Use the normal EI as a base and apply a t-correction factor
    z_norm = a / sigma_safe
    from scipy.stats import norm as _norm
    ei_normal = a * _norm.cdf(z_norm) + sigma_safe * _norm.pdf(z_norm)
    
    # T-correction: for df=3, the t-distribution has heavier tails.
    # The ratio of t-pdf to normal-pdf at z is larger for moderate |z|.
    # We can use the ratio of t to normal pdf as a correction.
    normal_pdf_val = _norm.pdf(z_norm)
    if np.any(normal_pdf_val > 1e-12):
        correction = t_pdf_val / normal_pdf_val
        correction = np.clip(correction, 0.5, 3.0)  # Bound the correction
    else:
        correction = np.ones_like(z_norm)
    
    ei_t = ei_normal * correction
    ei_t = np.clip(ei_t, -1e6, 1e6)
    
    # --- Uncertainty exploration term ---
    # For STP, the uncertainty is better captured by the t-distribution's spread.
    # Use the t-distribution's 95th percentile as a measure of uncertainty.
    t_95 = student_t.ppf(0.975, df)  # ~4.54 for df=3
    uncertainty = scale_t * t_95
    
    # Exploration score: higher uncertainty is more promising
    # But we need to balance with exploitation
    exploration = uncertainty
    
    # --- Combine exploitation and exploration ---
    # Normalize both to [0, 1] for fair combination
    ei_min, ei_max = np.min(ei_t), np.max(ei_t)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei_t - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei_t)
    
    exp_min, exp_max = np.min(exploration), np.max(exploration)
    exp_range = exp_max - exp_min
    if exp_range > 1e-12:
        exp_norm = (exploration - exp_min) / exp_range
    else:
        exp_norm = np.zeros_like(exploration)
    
    # Adaptive weighting: weight exploitation more when EI has high spread
    ei_spread = np.std(ei_t)
    exp_spread = np.std(exploration)
    
    if ei_spread > 1e-12 and exp_spread > 1e-12:
        ei_dominance = ei_spread / (ei_spread + exp_spread)
        w_ei = 0.60 + 0.25 * ei_dominance  # Range [0.60, 0.85]
    else:
        w_ei = 0.70
    
    w_exp = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_exp * exp_norm
    
    # --- Penalty for obviously poor regions ---
    scale = max(abs(f_best), 1.0)
    mu_penalty = np.maximum(0.0, (mu - f_best) / (scale + 1e-12))
    mu_penalty = np.minimum(mu_penalty, 3.0)
    scores = scores - 0.03 * mu_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
