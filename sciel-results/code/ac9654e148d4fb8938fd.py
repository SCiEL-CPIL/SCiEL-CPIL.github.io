import time

import numpy as np

_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _enkf_update(ensemble, obs_vec, obs_idx, obs_var, D, loc_width):
    """
    Standard EnKF update with Gaussian localization.
    Returns updated ensemble.
    """
    N = ensemble.shape[0]
    M = len(obs_idx)

    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)

    # Observation-space anomalies
    obs_anoms = anomalies[:, obs_idx]  # (N, M)

    # Covariance in observation space
    P_obs = (obs_anoms.T @ obs_anoms) / (N - 1)  # (M, M)
    R = obs_var * np.eye(M)
    S = P_obs + R

    # Regularize S
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-8:
        S += (1e-8 - min_eig) * np.eye(M)
    else:
        S += 1e-10 * np.eye(M)

    # Cross-covariance: P H^T = (1/(N-1)) A^T obs_anoms
    PHt = (anomalies.T @ obs_anoms) / (N - 1)  # (D, M)

    # Localization: Gaussian taper
    idx = np.arange(D)
    loc_mat = np.ones((D, M))
    for m_idx, obs_i in enumerate(obs_idx):
        dist = np.abs(idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_mat[:, m_idx] = np.exp(-0.5 * (dist / loc_width) ** 2)

    PHt_loc = PHt * loc_mat

    # Gain
    try:
        K = PHt_loc @ np.linalg.solve(S, np.eye(M))
    except np.linalg.LinAlgError:
        K = PHt_loc @ np.linalg.pinv(S)

    # Update
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]  # (N, M)
    deltas = innovations @ K.T  # (N, D)
    updated = ensemble + deltas

    return updated


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X.copy())

    # Ensure finite input
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    X = np.clip(X, -100.0, 100.0)
    y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=-100.0)

    # Handle observations shape
    if y.ndim == 2:
        y = y[0]
    else:
        y = y.flatten()

    # --- Two-pass EnKF with different localization widths ---
    # Pass 1: narrow localization for local accuracy
    X_pass1 = _enkf_update(X, y, obs, R_var, p, loc_width=2.5)

    # Pass 2: wider localization to smooth spurious correlations
    X_pass2 = _enkf_update(X_pass1, y, obs, R_var, p, loc_width=4.0)

    # --- Adaptive variance inflation ---
    m2 = X_pass2.mean(axis=0)
    An = X_pass2 - m2[None, :]

    # Compare observed-dimension ensemble spread to expected innovation std
    if len(obs) > 1:
        obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
        # Expected observation variance from ensemble + noise
        A_orig = X - X.mean(axis=0)[None, :]
        obs_anoms_orig = A_orig[:, obs]
        P_obs_orig = (obs_anoms_orig.T @ obs_anoms_orig) / (n - 1)
        obs_var_exp = np.diag(P_obs_orig) + R_var
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        alpha = np.clip(1.0 + 0.3 * (1.0 - ratio), 0.85, 1.4)
    else:
        alpha = 1.0
    X_pass2 = m2[None, :] + alpha * An

    # --- Small deterministic nudge on observed coords ---
    w = 0.08
    X_pass2[:, obs] += w * (y[None, :] - X_pass2[:, obs])

    out = _finite_clip(X_pass2)
    if (time.time() - _START) > _DEADLINE - 5.0:
        return out
    return out
