"""Lorenz-96 assimilation: ensemble Kalman filter with localization + inflation."""
import time
import math
import numpy as np

_N = 20          # state dimension
_M = 16          # ensemble size
_OBS = np.arange(0, _N, 2)  # even indices observed


def _localization_weight(i, j, sigma):
    """Gaussian localization weight between state indices i and j (periodic)."""
    d = (i - j + _N // 2) % _N - _N // 2
    return math.exp(-0.5 * (d / sigma) ** 2)


def _localize_cov(cov, sigma):
    """Apply Gaussian localization to a covariance matrix (periodic indices)."""
    W = np.empty((_N, _N))
    for i in range(_N):
        for j in range(_N):
            W[i, j] = _localization_weight(i, j, sigma)
    return cov * W


def _ekf_update(ens, obs, obs_idx, R, sigma_loc, inflate):
    """One localized ensemble Kalman filter analysis step (deterministic, centered)."""
    n = ens.shape[1]
    m = len(obs_idx)
    X = ens.T.copy()                       # (n, M)
    mean = X.mean(axis=1)
    Xc = X - mean[:, None]                 # centered anomalies
    cov = (Xc @ Xc.T) / (ens.shape[0] - 1)  # (n, n) sample covariance
    cov = _localize_cov(cov, sigma_loc)

    Pxx = cov[np.ix_(obs_idx, obs_idx)]
    Rm = R * np.eye(m)
    S = Pxx + Rm
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        invS = np.linalg.pinv(S)
    K = cov[:, obs_idx] @ invS             # (n, m) gain

    innov = obs - mean[obs_idx]            # (m,)
    mean = mean + K @ innov
    X = X - Xc[:, obs_idx] @ K.T           # deterministic ensemble update

    # Adaptive multiplicative inflation around the analysis mean.
    if inflate > 0.0:
        X = mean[:, None] + (1.0 + inflate) * (X - mean[:, None])

    out = X.T.copy()
    np.clip(out, -100.0, 100.0, out=out)
    return out


def analyze(ensemble, observations, observed_indices, observation_variance):
    t0 = time.time()
    DEADLINE = 120.0 - 8.0   # safety margin before the 120 s wall-clock kill

    obs_idx = np.asarray(observed_indices)
    R = float(observation_variance)

    # Deterministic localized EKF with mild adaptive inflation.
    updated = _ekf_update(
        ensemble, observations, obs_idx, R,
        sigma_loc=2.8, inflate=0.005,
    )

    if not np.all(np.isfinite(updated)):
        updated = ensemble.copy()
        np.clip(updated, -100.0, 100.0, out=updated)

    if time.time() - t0 > DEADLINE:
        return updated

    return updated
