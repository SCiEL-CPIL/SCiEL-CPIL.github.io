# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    True Expected Regret (ER) acquisition:
    ER = E[max(f_best - f, 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    This rewards both the probability of improvement AND the expected
    magnitude of improvement, unlike POI which only uses probability.
    
    We add a small smooth uncertainty bonus to encourage exploration
    in early stages when the posterior is highly uncertain.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against degenerate / negative std
    s = np.clip(sigma, 1e-9, None)

    z = (f_best - mu) / s
    z = np.clip(z, -38.0, 38.0)

    # True Expected Regret: (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    er = (f_best - mu) * Phi + s * phi
    
    # Ensure non-negative (numerical safety)
    er = np.maximum(er, 0.0)
    
    # Smooth exploration bonus: encourage exploration when uncertainty is high
    # relative to the scale of the problem. Use a soft normalization.
    s_max = np.max(s)
    if s_max > 1e-12:
        s_norm = s / (s_max + 1e-12)
        # Gentle bonus that saturates, contributing at most ~10% of typical ER scale
        # Scale the bonus relative to the max ER to keep it proportionally small
        er_max = np.max(er) if np.max(er) > 1e-12 else 1.0
        bonus_scale = 0.1 * er_max
        s_bonus = np.tanh(s_norm)
        er += bonus_scale * s_bonus

    # Ensure all scores are finite
    er = np.where(np.isfinite(er), er, 0.0)

    return er
# EVOLVE-BLOCK-END
