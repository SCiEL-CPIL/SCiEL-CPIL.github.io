"""Lorenz-96 ensemble assimilation: LESRF structure with blended adaptive inflation
and tight Gaussian localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Deterministic square-root (LESRF-style) ensemble analysis update for Lorenz-96.

    Combines:
      * LESRF structure (deterministic, mean-preserving, square-root gain),
      * blended adaptive inflation (innovation-based + spread-based),
      * tight Gaussian localization for the Lorenz-96 correlation length.

    Parameters
    ----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.

    Returns
    -------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)

    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean

    # --- Blended adaptive multiplicative inflation -------------------------
    # Innovation-based component: how far the mean is from the observations
    # relative to the expected innovation std (spread + observation noise).
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    innovation = obs - ens_mean[obs_idx]
    innov_std = np.sqrt(np.maximum(ens_var_obs, 1e-12) + R_val)
    innov_ratio = np.mean(np.abs(innovation) / np.maximum(innov_std, 1e-12))
    innov_inflation = np.clip(1.0 + 0.15 * (innov_ratio - 1.0), 0.98, 1.18)

    # Spread-based component: push ensemble spread toward a target near sqrt(R).
    target_spread = np.sqrt(R_val) * 1.35
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    spread_ratio = np.mean(target_spread / np.maximum(current_spread, 1e-12))
    spread_inflation = np.clip(spread_ratio, 0.95, 1.18)

    # Blend (0.6 innovation, 0.4 spread) with a small stabilising base.
    inflation = 0.6 * innov_inflation + 0.4 * spread_inflation
    inflation = 0.8 * inflation + 0.2 * 1.02
    inflation = np.clip(inflation, 1.0, 1.15)

    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean

    # --- Localization (tight Gaussian, circular distance) ------------------
    sigma = 2.2
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)

    # --- Square-root (LESRF) analysis --------------------------------------
    A_full = ens - ens_mean            # (n_members, n_vars)
    A_obs = ens[:, obs_idx] - ens_mean[obs_idx]  # (n_members, n_obs)

    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs) + 1e-8 * np.eye(n_obs)

    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(S + 1e-6 * np.eye(n_obs))

    innovation = obs - ens_mean[obs_idx]  # (n_obs,)

    P_xo = (A_full.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    P_xo_loc = P_xo * loc_weights

    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)

    delta_mean = K @ innovation  # (n_vars,)
    ens_mean_new = ens_mean + delta_mean

    # Update anomalies (square-root form preserves covariance structure).
    A_obs_Kt = A_obs @ K.T  # (n_members, n_vars)
    A_new = A_full - A_obs_Kt

    ens = ens_mean_new[None, :] + A_new

    # --- Post-processing ---------------------------------------------------
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)

    if time.time() > deadline:
        return ens.astype(ensemble.dtype)

    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
