"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis step with localization and adaptive inflation
    for Lorenz-96 partial observation assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Safety check: ensure inputs are finite
    if not np.all(np.isfinite(ensemble)) or not np.all(np.isfinite(observations)):
        return ensemble.copy()
    
    # Ensure observation_variance is positive
    if observation_variance <= 0 or not np.isfinite(observation_variance):
        observation_variance = 0.49
    
    N = ensemble.shape[1]  # 20
    M = len(observed_indices)  # 10
    Mm = ensemble.shape[0]  # 16
    
    # Compute ensemble mean and anomaly matrix
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # (Mm, N)
    
    # Sample covariance of anomalies
    # C = (1/(Mm-1)) * A @ A.T
    cov = (anomalies @ anomalies.T) / (Mm - 1.0)
    
    # Localization: apply a Gaussian or exponential localization to the covariance
    # For Lorenz-96 with periodic boundary conditions, use periodic distance
    # Localization width parameter
    loc_width = 3.0  # In index units
    
    # Build localization matrix
    # For each pair of variables i, j, compute periodic distance
    loc_matrix = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            # Periodic distance on a ring of size N
            dist = min(abs(i - j), N - abs(i - j))
            # Gaussian localization
            loc_matrix[i, j] = np.exp(-0.5 * (dist / loc_width) ** 2)
    
    # Apply localization to covariance (Hadamard product)
    # The localization should be applied to the state-space covariance
    # We need to localize the full N x N covariance, but we only have the M x M observed subspace
    # Actually, we need the full state covariance for the update. Let's compute it.
    
    # Full state covariance: C_full = (1/(Mm-1)) * A @ A.T where A is (Mm, N)
    # But we need to work with the observation operator H
    # H is M x N, with H[m, i] = 1 if i == observed_indices[m] else 0
    
    # Construct H
    H = np.zeros((M, N))
    for m, idx in enumerate(observed_indices):
        H[m, idx] = 1.0
    
    # Observation covariance R = observation_variance * I_M
    R = observation_variance * np.eye(M)
    
    # Innovation: y - H @ x_mean
    innovation = observations - H @ mean  # (M,)
    
    # Covariance of observations: P_y = H @ C_full @ H.T + R
    # C_full is N x N
    C_full = (anomalies @ anomalies.T) / (Mm - 1.0)  # (N, N)
    
    # Apply localization to C_full
    C_full_loc = C_full * loc_matrix
    
    P_y = H @ C_full_loc @ H.T + R  # (M, M)
    
    # Compute P_y inverse
    try:
        P_y_inv = np.linalg.inv(P_y)
    except np.linalg.LinAlgError:
        P_y_inv = np.linalg.pinv(P_y)
    
    # Gain: K = C_full_loc @ H.T @ P_y_inv
    K = C_full_loc @ H.T @ P_y_inv  # (N, M)
    
    # Analysis update: x_a = x + K @ (y - H @ x)
    updated_mean = mean + K @ innovation  # (N,)
    
    # Update ensemble members
    # x_a_m = x_m + K @ (y_m - H @ x_m) where y_m = y + noise? 
    # Actually, for each ensemble member:
    # x_a_m = x_m + K @ (y - H @ x_m)
    # This is the standard EnKF update
    # x_a_m = x_m + K @ (y - H @ x_m)
    
    # H @ x_m for each member: (Mm, M)
    Hx = ensemble @ H.T  # (Mm, M)
    
    # Innovation for each member: y - H @ x_m
    innovations_members = observations[None, :] - Hx  # (Mm, M)
    
    # Update: x_a_m = x_m + K @ innovation_m
    # K is (N, M), innovations_members is (Mm, M)
    # x_a_m = x_m + innovations_members @ K.T
    updated_ensemble = ensemble + innovations_members @ K.T  # (Mm, N)
    
    # Adaptive inflation to prevent filter collapse
    # Compute ratio of observation fit to expected
    # If the ensemble is too confident, inflate
    # Compute the spread of the analysis
    analysis_mean = np.mean(updated_ensemble, axis=0)
    analysis_anomalies = updated_ensemble - analysis_mean
    
    # Compute the spread of observations vs ensemble
    # Inflate if the ensemble is under-dispersive
    # Use a simple inflation factor based on the ratio of innovation norm to expected
    innovation_norm_sq = np.sum(innovation ** 2)
    expected_norm_sq = np.sum(np.diag(P_y))
    
    # Inflation factor
    if expected_norm_sq > 0:
        ratio = innovation_norm_sq / expected_norm_sq
        # If ratio is large, the ensemble may be too confident
        # Inflate to increase spread
        # Target ratio around 1.0-2.0
        if ratio > 1.5:
            inflation_factor = 1.0 + 0.05 * (ratio - 1.5)
            inflation_factor = min(inflation_factor, 1.2)  # Cap inflation
        else:
            inflation_factor = 1.0
    else:
        inflation_factor = 1.0
    
    # Apply inflation about the analysis mean
    if inflation_factor != 1.0:
        updated_ensemble = analysis_mean + inflation_factor * (updated_ensemble - analysis_mean)
    
    # Ensure output is within [-100, 100] as required
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(updated_ensemble)):
        # Fallback to original ensemble if something went wrong
        return ensemble.copy()
    
    return updated_ensemble
# EVOLVE-BLOCK-END
