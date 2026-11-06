"""Two-pass localized EnKF for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaussian_loc_matrix(n_state, observed_indices, width):
    """Gaussian localization weights (n_state x n_obs) on a ring."""
    M = len(observed_indices)
    D = n_state
    idx = np.arange(D)
    obs_idx = np.asarray(observed_indices)
    dist = np.abs(idx[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, D - dist)
    return np.exp(-0.5 * (dist / width) ** 2)


def _enkf_pass(ensemble, observations, observed_indices, obs_var, loc_width):
    """One deterministic EnKF update with Gaussian localization."""
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_idx = np.asarray(observed_indices, dtype=np.int64)

    ens_mean = ensemble.mean(axis=0)
    anomalies = ensemble - ens_mean  # (N, D)

    obs_anoms = anomalies[:, obs_idx]  # (N, M)

    # Localized cross-covariance: PHt_loc[d, m] = (1/(N-1)) sum_n A[n,d] * A[n, obs_idx[m]] * loc[d, m]
    loc_mat = _gaussian_loc_matrix(D, obs_idx, loc_width)  # (D, M)

    PHt = (anomalies.T @ obs_anoms) / (N - 1.0)  # (D, M)
    PHt_loc = PHt * loc_mat  # (D, M)

    # Observation-space covariance
    P_obs = (obs_anoms.T @ obs_anoms) / (N - 1.0)  # (M, M)
    R = obs_var * np.eye(M)
    S = P_obs + R + 1e-8 * np.eye(M)

    # Gain: K = PHt_loc @ S^{-1}  (D x M)
    try:
        K = np.linalg.solve(S.T, PHt_loc.T).T
    except np.linalg.LinAlgError:
        K = np.linalg.lstsq(S.T, PHt_loc.T, rcond=None)[0].T

    # Per-member innovations: y - H x_n  (N, M)
    innovations = observations[None, :] - ensemble[:, obs_idx]
    deltas = innovations @ K.T  # (N, D)

    return ensemble + deltas


def analyze(ensemble, observations, observed_indices, observation_variance):
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)

    if observations.ndim == 2:
        observations = observations[0]

    N, D = ensemble.shape
    M = len(observed_indices)
    obs_var = float(observation_variance)

    # Guard against degenerate inputs
    if N < 2 or D < 3 or M == 0:
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        return np.clip(np.nan_to_num(result), -100.0, 100.0)

    deadline = time.time() + 110.0  # safety margin before 120 s limit

    # --- Pass 1: narrow localization to fix local errors --------------------
    width1 = 1.5  # narrow: only very close observations influence
    result = _enkf_pass(ensemble, observations, observed_indices, obs_var, width1)

    # --- Pass 2: wide localization to propagate information -----------------
    # Re-center: treat pass-1 result as new background
    width2 = 3.5  # wide: allows information to spread further around the ring
    result = _enkf_pass(result, observations, observed_indices, obs_var, width2)

    # --- Adaptive inflation to prevent collapse ------------------------------
    mean_bg = ensemble.mean(axis=0)
    mean_a = result.mean(axis=0)

    spread_bg = np.std(ensemble[:, observed_indices], axis=0)
    spread_a = np.std(result[:, observed_indices], axis=0)
    ratio = np.mean(spread_a / (spread_bg + 1e-8))

    # Target ratio ~0.55 (typical for a good KF update); inflate if too low
    target = 0.55
    if ratio < target:
        infl = 1.0 + 0.6 * (target - ratio) / target
        infl = min(infl, 1.8)
    elif ratio > 0.85:
        # Deflate slightly if spread is too large (over-dispersion)
        infl = 1.0 - 0.15 * (ratio - 0.85) / 0.15
        infl = max(infl, 0.90)
    else:
        infl = 1.0

    result = mean_a + infl * (result - mean_a)

    # --- Safety: finite and bounded ------------------------------------------
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)

    return result

# EVOLVE-BLOCK-END
