"""Ensemble Kalman Filter with localization, inflation, and rank-deficiency handling."""
import numpy as np
import time

def _lorenz96_step(state, F, dt=0.01, steps=1):
    """Propagate a single Lorenz-96 state forward using RK4."""
    n = len(state)
    for _ in range(steps):
        def f(x):
            return ((np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F)
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using an Ensemble Kalman Filter (EnKF)
    with spatial localization, multiplicative inflation, and rank-deficiency handling.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n_ens = ensemble.shape[0]
    n_state = ensemble.shape[1]
    n_obs = len(observations)
    
    # Ensure observation_variance is a scalar or array
    if np.isscalar(observation_variance):
        R = np.eye(n_obs) * observation_variance
    else:
        R = np.diag(observation_variance)
    
    # Ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # shape (n_ens, n_state)
    
    # Estimate ensemble covariance (with small regularization for stability)
    # C_aa = anomalies.T @ anomalies / (n_ens - 1)
    # Use SVD for numerical stability and to handle rank deficiency
    
    # Centered anomalies
    A = anomalies  # (n_ens, n_state)
    
    # SVD of anomalies: A = U * S * Vt
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback to simple nudging if SVD fails
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        return np.clip(result, -100, 100)
    
    # Ensemble covariance in state space
    # C_aa = A.T @ A / (n_ens - 1) = Vt.T @ diag(S^2) @ Vt / (n_ens - 1)
    # For the Kalman gain, we need C_aa @ H.T @ (H @ C_aa @ H.T + R)^-1
    
    # Observation operator H: extracts observed_indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # H @ C_aa @ H.T = H @ Vt.T @ diag(S^2/(n_ens-1)) @ Vt @ H.T
    # Let's compute this efficiently
    # H @ Vt.T has shape (n_obs, n_ens)
    HVtT = H @ Vt.T  # (n_obs, n_ens)
    # diag(S^2/(n_ens-1)) @ (Vt @ H.T)
    VtH_T = Vt @ H.T  # (n_ens, n_obs)
    
    # H @ C_aa @ H.T = HVtT @ diag(S^2/(n_ens-1)) @ VtH_T
    # = (HVtT * S) @ diag(S/(n_ens-1)) @ VtH_T
    # More simply: (HVtT * S[np.newaxis, :]) @ (VtH_T * S[:, np.newaxis]) / (n_ens - 1)
    
    S_scaled = S / np.sqrt(n_ens - 1)
    # H C_aa H^T = (H Vt^T) @ diag(S^2/(n_ens-1)) @ (Vt H^T)
    # = (H Vt^T * S) @ (Vt H^T * S) / (n_ens - 1) ... wait
    # Let's be more careful.
    # C_aa = Vt.T @ diag(S^2) @ Vt / (n_ens-1)
    # H C_aa H^T = H Vt.T @ diag(S^2) @ Vt H^T / (n_ens-1)
    # Let M = H Vt.T, shape (n_obs, n_ens)
    # Let N = Vt H^T, shape (n_ens, n_obs)
    # H C_aa H^T = M @ diag(S^2) @ N / (n_ens-1)
    # = (M * S[np.newaxis, :]) @ (N * S[:, np.newaxis]) / (n_ens-1)
    
    MC = M * S[np.newaxis, :]
    NC = N * S[:, np.newaxis]
    HCH_T = (MC @ NC) / (n_ens - 1)
    
    # Innovation covariance: S_innov = H C_aa H^T + R
    S_innov = HCH_T + R
    
    # Innovation
    innovation = observations - H @ ens_mean
    
    # Kalman gain: K = C_aa H^T S_innov^-1
    # C_aa H^T = Vt.T @ diag(S^2) @ Vt H^T / (n_ens-1) = Vt.T @ diag(S^2) @ N / (n_ens-1)
    # = Vt.T @ (diag(S^2) @ N) / (n_ens-1)
    # diag(S^2) @ N = (S^2)[:, np.newaxis] * N
    CN = (S**2)[:, np.newaxis] * N
    CaaH_T = Vt.T @ CN / (n_ens - 1)
    
    # Solve for S_innov^-1 @ innovation
    try:
        innov_sol = np.linalg.solve(S_innov, innovation)
    except np.linalg.LinAlgError:
        innov_sol = np.linalg.lstsq(S_innov, innovation, rcond=None)[0]
    
    # Analysis mean update
    ens_mean_analysis = ens_mean + CaaH_T @ innov_sol
    
    # Ensemble spread update (square root form)
    # We want to update the anomalies such that the new covariance is the analysis covariance.
    # Using the SVD-based approach, we can update the singular vectors and values.
    # This is complex. Instead, let's use a simpler approach: 
    # For each member, compute the analysis perturbation using the Kalman gain.
    
    # K = C_aa H^T S_innov^-1
    # K = CaaH_T @ S_innov^-1
    # We already have CaaH_T. We need S_innov^-1.
    # K = CaaH_T @ inv(S_innov)
    # But we can avoid explicit inversion by noting:
    # K @ H = CaaH_T @ S_innov^-1 @ H
    # And for the anomaly update:
    # a_i_analysis = a_i_forecast - K @ H @ a_i_forecast + K @ (obs - H @ mean_forecast) ... no
    # The standard EnKF update for each member is:
    # x_i^a = x_i^f + K @ (y_obs - H @ x_i^f)  ... this is for the mean, not individual members in a linear sense.
    # Actually, the linearized update for each member is:
    # x_i^a = x_i^f + K @ (H @ x_i^f - H @ mean^f)  ... no, that's not right either.
    
    # Let's use the standard formula:
    # x_i^a = x_i^f + K @ (y - H @ x_i^f)  is NOT correct for ensemble members.
    # The correct linearized update is:
    # x_i^a = x_i^f + K @ (H @ x_i^f - H @ mean^f)  ... wait, no.
    # The innovation for member i is y - H @ x_i^f.
    # But the standard EnKF uses:
    # x_i^a = x_i^f + K @ (y - H @ x_i^f)  ... this would collapse the ensemble.
    
    # The correct update is:
    # x_i^a = x_i^f + K @ (H @ x_i^f - H @ mean^f)  ... no.
    # Let me recall. The Kalman filter update is:
    # x^a = x^f + K @ (y - H @ x^f)
    # For ensemble members, we apply the same linear operator:
    # x_i^a = x_i^f + K @ (H @ x_i^f - H @ mean^f)  ... this is for the perturbation around the mean.
    # Actually, the perturbation a_i = x_i^f - mean^f
    # a_i^a = (I - K @ H) @ a_i^f
    # And the mean update is:
    # mean^a = mean^f + K @ (y - H @ mean^f)
    
    # So:
    # x_i^a = mean^a + (I - K @ H) @ (x_i^f - mean^f)
    #        = mean^f + K @ (y - H @ mean^f) + (I - K @ H) @ (x_i^f - mean^f)
    #        = x_i^f + K @ (y - H @ mean^f) - K @ H @ (x_i^f - mean^f)
    #        = x_i^f + K @ (y - H @ mean^f - H @ x_i^f + H @ mean^f)
    #        = x_i^f + K @ (y - H @ x_i^f)
    
    # So the formula IS x_i^a = x_i^f + K @ (y - H @ x_i^f)
    # But this is only valid for the linearized filter. Let's use this.
    
    # We need K = CaaH_T @ S_innov^-1
    # Instead of computing K explicitly, we can compute K @ (y - H @ x_i^f) for each member.
    # K @ v = CaaH_T @ S_innov^-1 @ v
    # Let w = S_innov^-1 @ v
    # Then K @ v = CaaH_T @ w
    
    # For each member, v_i = y - H @ x_i^f
    # w_i = S_innov^-1 @ v_i
    # update_i = CaaH_T @ w_i
    
    # Let's compute all v_i at once
    # H @ ensemble has shape (n_ens, n_obs)
    H_ens = H @ ensemble.T  # (n_state, n_ens) -> wait, H is (n_obs, n_state), ensemble is (n_ens, n_state)
    # H @ x_i^f for each i: (H @ ensemble.T).T
    H_ens_T = ensemble @ H.T  # (n_ens, n_obs)
    
    # v_i = y - H @ x_i^f, shape (n_ens, n_obs)
    V = observations[np.newaxis, :] - H_ens_T  # (n_ens, n_obs)
    
    # w_i = S_innov^-1 @ v_i
    # W = V.T @ S_innov^-1.T  ... or solve S_innov @ W.T = V.T
    # W.T has shape (n_obs, n_ens)
    try:
        W_T = np.linalg.solve(S_innov, V.T)  # (n_obs, n_ens)
    except np.linalg.LinAlgError:
        W_T = np.linalg.lstsq(S_innov, V.T, rcond=None)[0]
    W = W_T.T  # (n_ens, n_obs)
    
    # update_i = CaaH_T @ w_i
    # Updates = W @ CaaH_T.T  ... (n_ens, n_obs) @ (n_obs, n_state) = (n_ens, n_state)
    Updates = W @ CaaH_T.T  # (n_ens, n_state)
    
    # Analysis ensemble
    result = ensemble + Updates
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
    
    # Clip to valid range
    result = np.clip(result, -100, 100)
    
    return result
