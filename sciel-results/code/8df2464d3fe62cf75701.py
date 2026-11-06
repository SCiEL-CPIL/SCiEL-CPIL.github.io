"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X.copy())

    # --- Prior mean and anomalies ---
    m = X.mean(axis=0)
    A = X - m[None, :]                       # (n, p) anomalies
    
    # Sample covariance (unbiased)
    P = (A.T @ A) / (n - 1)                   # (p, p)

    # --- Localization: Gaussian taper on the periodic ring ---
    idx = np.arange(p)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, p - dist)         # periodic distance
    
    # Gaussian localization with radius r_loc
    r_loc = 4.0
    loc = np.exp(-0.5 * (dist / r_loc) ** 2)
    
    P = P * loc

    # --- Observation operator: identity on obs indices ---
    k = len(obs)
    Po = P[np.ix_(obs, obs)]                  # (k, k)
    Ho = P[obs, :].T                          # (p, k)   = P[:, obs]
    
    # Stable solve for gain
    try:
        K = Ho @ np.linalg.solve(Po + R_var * np.eye(k), np.eye(k))
    except np.linalg.LinAlgError:
        K = Ho @ np.linalg.pinv(Po + R_var * np.eye(k))

    # --- EAKF Update ---
    # 1. Update mean
    m_new = m + K @ (y - m[obs])
    
    # 2. Update anomalies
    # A_new = A - 0.5 * K @ (H @ A)
    # H @ A is A[:, obs] (n, k)
    # K @ (H @ A).T is (p, k) @ (k, n) = (p, n)
    # Transpose to (n, p)
    HA = A[:, obs] # (n, k)
    term = (K @ HA.T).T # (n, p)
    A_new = A - 0.5 * term

    X_new = m_new[None, :] + A_new

    # --- Adaptive Inflation ---
    if k > 1:
        obs_var_ens = np.var(A_new[:, obs], axis=0, ddof=1)
        obs_var_exp = np.diag(Po) + R_var
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        alpha = np.clip(1.0 + 0.5 * (1.0 - ratio), 0.8, 1.5)
    else:
        alpha = 1.0
        
    A_new = alpha * A_new
    X_new = m_new[None, :] + A_new

    out = _finite_clip(X_new)
    if (time.time() - _START) > _DEADLINE - 5.0:
        return out
    return out
# EVOLVE-BLOCK-END
