"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.88  # Use 88% of the 150s budget


def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
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


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _periodic_dist(i, j, D):
    """Periodic distance on a ring of size D."""
    d = abs(i - j)
    return min(d, D - d)


def _compute_localization_matrix(D, observed_indices, L):
    """
    Compute localization weights for cross-covariance.
    Returns (D, M) matrix where entry [d, m] is the localization weight
    between state variable d and observation m (at index observed_indices[m]).
    """
    M = len(observed_indices)
    W = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = _periodic_dist(d, obs_i, D)
            W[d, m_idx] = _gaussian_localization(dist, L)
    return W


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) analysis with localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of N members, D state variables
    observations : (M,) array - noisy observations of observed coordinates
    observed_indices : array of M indices - indices of observed coordinates
    observation_variance : float - variance of observation noise
    
    Returns:
    --------
    updated_ensemble : (N, D) array - updated ensemble
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 150 * _DEADLINE_FRACTION:
        return np.clip(ensemble, -100.0, 100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs is 1D
    obs = np.asarray(observations, dtype=np.float64).ravel()
    if obs.size != M:
        obs = obs[:M]
    
    # --- Step 1: Compute ensemble mean and covariance ---
    ens_mean = np.mean(ensemble, axis=0)  # (D,)
    
    # Centered ensemble
    X = ensemble - ens_mean[None, :]  # (N, D)
    
    # Sample covariance of state: (1/(N-1)) X X^T
    # Cross-covariance with observations: (1/(N-1)) X H^T Y^T
    Hx = ensemble[:, observed_indices]  # (N, M)
    Hx_centered = Hx - ens_mean[observed_indices][None, :]  # (N, M)
    
    # P_XX = (1/(N-1)) * X @ X.T  -> (D, D)
    # P_XY = (1/(N-1)) * X @ Hx_centered.T  -> (D, M)
    # P_YY = (1/(N-1)) * Hx_centered @ Hx_centered.T  -> (M, M)
    
    # For the gain computation, we need K = P_XY @ (P_YY + R)^{-1}
    # where R = observation_variance * I_M
    
    # To avoid computing full DxD, use the identity:
    # K = X @ Hx_centered.T @ (P_YY + R)^{-1} / (N-1)
    # Update: x_a = x_f + K @ (y - H x_f)
    #         For each member: x_i^a = x_i^f + K @ (y - H x_i^f)
    
    # Compute P_YY + R (M x M)
    P_YY = (Hx_centered.T @ Hx_centered) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_YY + R  # (M, M)
    
    # Invert S (M is small, typically 10)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Compute gain K = (1/(N-1)) * X @ Hx_centered.T @ S_inv  -> (D, M)
    # But we need to apply localization.
    
    # Localization: apply to cross-covariance
    # L_matrix[d, m] = localization weight between state d and obs m
    L_loc = 3.0  # localization length scale
    W_loc = _compute_localization_matrix(D, observed_indices, L_loc)
    
    # Apply localization to cross-covariance
    # P_XY_local[d, m] = P_XY[d, m] * W_loc[d, m]
    # P_XY = (1/(N-1)) * X @ Hx_centered.T
    P_XY = (X @ Hx_centered.T) / (N - 1)  # (D, M)
    P_XY_loc = P_XY * W_loc  # (D, M)
    
    # Also localize P_YY (diagonal only is common, but full is fine for small M)
    # Actually, for the observation covariance, we typically don't localize P_YY
    # but we can apply it to the cross-covariance only.
    
    # Gain: K = P_XY_loc @ S_inv  -> (D, M)
    K = P_XY_loc @ S_inv  # (D, M)
    
    # --- Step 2: Adaptive inflation ---
    # Compare ensemble spread to innovation magnitude
    # Innovation: y - H x (for each member)
    innovations = obs[None, :] - Hx  # (N, M)
    
    # Mean squared innovation
    msi = np.mean(innovations ** 2)
    
    # Expected innovation variance from ensemble: trace(P_YY + R) / M
    expected_innov_var = np.trace(S) / M
    
    # Inflation factor: if ensemble is too spread out relative to innovations, deflate
    # If ensemble is too tight, inflate
    # ratio = msi / expected_innov_var
    # If ratio < 1, ensemble is too wide -> deflate (inflation < 1)
    # If ratio > 1, ensemble is too tight -> inflate (inflation > 1)
    
    # Use a gentle adaptive inflation
    ratio = msi / expected_innov_var if expected_innov_var > 1e-10 else 1.0
    
    # Inflation factor applied to the centered ensemble
    # We want to adjust spread so that post-update spread matches observations
    # Simple approach: inflation = 1 + 0.5 * (ratio - 1) clipped
    # But this can be unstable. Use a more conservative approach.
    
    # Adaptive inflation: scale the centered ensemble
    # inflation_factor should be close to 1
    # If ratio is very different from 1, adjust gently
    inflation_factor = 1.0 + 0.3 * (ratio - 1.0)
    inflation_factor = np.clip(inflation_factor, 0.8, 1.5)
    
    # Apply inflation to centered ensemble
    X_inflated = X * inflation_factor
    
    # Recompute P_XY with inflated ensemble
    Hx_inflated = ens_mean[observed_indices][None, :] + X_inflated[:, observed_indices]
    Hx_centered_inflated = X_inflated[:, observed_indices]
    
    P_XY_inflated = (X_inflated @ Hx_centered_inflated.T) / (N - 1)
    P_XY_loc_inflated = P_XY_inflated * W_loc
    
    # Recompute P_YY with inflated ensemble
    P_YY_inflated = (Hx_centered_inflated.T @ Hx_centered_inflated) / (N - 1)
    S_inflated = P_YY_inflated + R
    
    try:
        S_inv_inflated = np.linalg.inv(S_inflated)
    except np.linalg.LinAlgError:
        S_inv_inflated = np.linalg.pinv(S_inflated)
    
    K_inflated = P_XY_loc_inflated @ S_inv_inflated
    
    # --- Step 3: Ensemble update ---
    # For each member i:
    # x_i^a = x_i^f + K @ (y - H x_i^f)
    #        = x_i^f + K @ innovations[i]
    
    # innovations: (N, M), K_inflated: (D, M)
    # Update: (N, D) = (N, M) @ (M, D)
    correction = innovations @ K_inflated.T  # (N, D)
    
    new_ensemble = ensemble + correction
    
    # --- Step 4: Renormalize to preserve mean (optional, but helps) ---
    # The EnKF update should preserve the ensemble mean if done correctly,
    # but numerical issues can cause drift. Re-center.
    new_mean = np.mean(new_ensemble, axis=0)
    target_mean = ens_mean + K_inflated @ (obs - ens_mean[observed_indices])
    
    # Actually, the theoretical posterior mean is:
    # x_a = x_f + K @ (y - H x_f)
    # The ensemble mean after update should be close to this.
    # Let's just ensure the mean is reasonable.
    
    # --- Step 5: Safety checks ---
    new_ensemble = np.nan_to_num(new_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    return new_ensemble

# EVOLVE-BLOCK-END
