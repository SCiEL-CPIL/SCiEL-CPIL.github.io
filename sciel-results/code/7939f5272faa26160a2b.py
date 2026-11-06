"""Ensemble Kalman Filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# Module-level state for adaptive inflation persistence
_inflation_state = {'alpha': 1.0, 'n': 0}

def analyze(ensemble, observations, observed_indices, observation_variance):
    t0 = time.time()
    DEADLINE = 120.0  # wall-clock seconds; return early with best result if needed

    ens = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.float64)
    obs_idx = np.asarray(observed_indices, dtype=int)
    R = float(observation_variance)

    N = ens.shape[1]  # 20
    M = len(obs)      # 10 observed variables
    E = ens.shape[0]  # 16 members

    # --- Adaptive multiplicative inflation (persist across calls) ---
    alpha = _inflation_state['alpha']
    # Inflate the ensemble about its mean
    mean0 = ens.mean(axis=0)
    ens = mean0 + alpha * (ens - mean0)
    alpha = 1.0  # reset for next call; we'll update at end

    # --- Compute ensemble covariance and cross-covariance ---
    # Center ensemble
    ens_centered = ens - ens.mean(axis=0)  # (E, N)
    # Use unbiased covariance: divide by (E-1)
    # P_ee = (1/(E-1)) * ens_centered^T @ ens_centered  (N x N)
    # P_eo = (1/(E-1)) * ens_centered[:, obs_idx]^T @ (obs - obs_mean)[:, None]  (N x M)
    
    obs_mean = obs.mean()
    obs_centered = obs - obs_mean  # (M,)
    
    ens_obs = ens_centered[:, obs_idx]  # (E, M)
    
    # Cross-covariance: (N, M)
    P_eo = (ens_centered.T @ ens_obs) / (E - 1)
    # Observation-space covariance of ensemble: (M, M)
    P_oo = (ens_obs.T @ ens_obs) / (E - 1)
    # Add observation error covariance
    S = P_oo + R * np.eye(M)
    
    # --- Localization: apply correlation-based localization to cross-covariance ---
    # For each observed variable, compute correlation with all state variables
    # and apply a localizing function that damps spurious correlations
    # Use a simple exponential or polynomial localization based on index distance
    
    # Compute ensemble correlation for localization
    # std of each state variable
    std_state = np.sqrt(np.maximum(np.diag(P_oo if False else (ens_centered.T @ ens_centered) / (E-1)), 1e-10))
    # Actually compute full state std
    P_ee = (ens_centered.T @ ens_centered) / (E - 1)
    std_state = np.sqrt(np.diag(P_ee))
    std_obs = np.sqrt(np.diag(P_oo))
    
    # Correlation matrix between state and obs
    # C[i, j] = P_eo[i, j] / (std_state[i] * std_obs[j])
    denom = np.outer(std_state, std_obs)
    C = P_eo / denom
    
    # Apply localization: for each observed variable j, damp correlations
    # based on circular distance in the 20-variable ring
    # Use a simple exponential localization: L(d) = exp(-d^2 / (2 * L0^2))
    # with L0 = 3 (localization length scale)
    L0 = 3.0
    for j in range(M):
        obs_pos = obs_idx[j]
        # Circular distances from obs_pos to all state positions
        dists = np.minimum(np.abs(np.arange(N) - obs_pos), N - np.abs(np.arange(N) - obs_pos))
        loc = np.exp(-dists**2 / (2.0 * L0**2))
        # Apply to cross-covariance row for this observed variable
        P_eo[:, j] *= loc
    
    # --- Compute Kalman gain ---
    # K = P_eo @ inv(S)  (N x M)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    K = P_eo @ S_inv  # (N, M)
    
    # --- Ensemble Adjustment Kalman Filter update ---
    # For each member i:
    #   x_i = x_i + K @ (y - H(x_mean))  ... no, EAKF is different
    # EAKF: x_i' = x_mean + (I - K H) (x_i - x_mean)  ... not quite
    # Proper EAKF: 
    #   Let H be the observation operator (selects obs_idx)
    #   x_i' = x_mean + (x_i - x_mean) - P_eo @ S^{-1} @ (H(x_i) - y)
    #   Wait, that's not right either.
    #
    # Standard EnKF with ensemble adjustment:
    #   x_i' = x_mean + (x_i - x_mean) * (I - K H)  ... no
    #
    # Let me use the standard perturbed-observation EnKF approach but with
    # the ensemble mean preserved (Ensemble Adjustment Kalman Filter, Whitaker & Hamill 2012):
    #   x_i' = x_mean + (x_i - x_mean) - K (H(x_i) - y)
    #   where K = P_eo S^{-1}
    #   This preserves the ensemble mean: mean(x_i') = x_mean + 0 - K(H(x_mean) - y)
    #   Hmm, that shifts the mean. Let me use:
    #   x_i' = x_mean + (x_i - x_mean) - K (H(x_i) - H(x_mean))
    #   This preserves the mean exactly.
    
    H_xi = ens[:, obs_idx]  # (E, M)
    H_xmean = mean0[obs_idx]  # (M,)
    
    # Innovation for each member: (E, M)
    innov = H_xi - H_xmean  # (E, M)
    
    # Update: x_i' = x_i - K @ innov_i
    # ens shape (E, N), K shape (N, M), innov shape (E, M)
    # ens' = ens - innov @ K.T
    ens = ens - innov @ K.T
    
    # --- Adaptive inflation update ---
    # Check if ensemble spread at observed locations is reasonable
    ens_obs_after = ens[:, obs_idx]
    spread_after = ens_obs_after.std(axis=0)
    # Compare to expected spread: sqrt(diag(P_oo) + R) roughly
    expected_spread = np.sqrt(np.diag(P_oo) + R)
    ratio = np.mean(spread_after / np.maximum(expected_spread, 1e-10))
    
    # Inflate if spread is too small, deflate if too large
    # Target ratio ~ 1.0
    if ratio < 0.7:
        alpha_new = alpha * 1.1
    elif ratio > 1.5:
        alpha_new = alpha * 0.95
    else:
        alpha_new = alpha
    # Bound alpha
    alpha_new = np.clip(alpha_new, 0.8, 2.0)
    _inflation_state['alpha'] = alpha_new
    
    # --- Clip to valid range ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens
