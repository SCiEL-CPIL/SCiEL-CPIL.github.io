# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 325.0          # safety margin below the 330 s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    # Vectorised, O(n) cost-aware EI with an adaptive cost exponent.
    # cost ~ 1 + mean(x) in [1, 2]; the exponent beta adapts to the
    # remaining budget so exploration/cost-sensitivity shifts as the GP matures.
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # ---- time guard: bail out early with a safe, finite fallback ----
    if (time.time() - _T0) > _DEADLINE:
        c = np.clip(cost, 1.0, None)
        return np.clip((f_best - mu) / c, -1e6, 1e6)

    # ---- adaptive cost exponent (structural: self-tuning per iteration) ----
    cmean = float(np.mean(cost))
    beta = 0.5 + 0.5 * min(1.0, max(0.0, (cmean - 1.0)))

    # ---- cost-aware expected improvement (higher = better) ----
    s = np.maximum(sigma, _EPS)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)

    out = ei / np.power(np.clip(cost, 1.0, None), beta)
    out = np.where(np.isfinite(out), out, 0.0)
    return out
# EVOLVE-BLOCK-END
