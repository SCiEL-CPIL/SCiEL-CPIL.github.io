# EVOLVE-BLOCK-START
import numpy as np
import math


def acquisition(mu, sigma, f_best):
    sigma_safe = np.clip(sigma, 1e-8, None)
    diff = f_best - mu
    z = diff / sigma_safe
    half_z = 0.5 * z
    cdf_val = 0.5 * (1.0 + np.vectorize(math.erf)(half_z / math.sqrt(2.0)))
    pdf_val = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    base = diff * cdf_val + sigma_safe * pdf_val
    bonus = sigma_safe / (np.abs(diff) + sigma_safe + 1e-6)
    result = base + 0.05 * bonus
    result[~np.isfinite(result)] = -1e10
    return result
# EVOLVE-BLOCK-END
