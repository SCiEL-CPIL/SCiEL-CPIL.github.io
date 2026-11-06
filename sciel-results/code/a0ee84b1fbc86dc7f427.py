"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _ensemble_kalman_filter(ensemble, observations, observed_indices, observation_variance,
                            localization_length=4.0, inflation_factor=1.05,
                            max_inflation=1.5, n_localization=10):
    """
    Ensemble Kalman Filter update with Gaussian localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    observations : (N, M) array - observations for each ensemble member
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    inflation_factor : float - base multiplicative inflation factor
    max_inflation : float - maximum allowed inflation
    n_localization : int - number of nearest neighbors for localization
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean
    
    # Compute sample covariance (with N-1 normalization for unbiased estimate)
    # P = anomalies.T @ anomalies / (N - 1)
    # But we'll work with the anomaly matrix directly for efficiency
    
    # Observation operator: H selects observed components
    # H @ x = x[observed_indices]
    
    # Compute H @ P @ H^T (observation-space covariance)
    # P_obs = (anomalies[:, observed_indices].T @ anomalies[:, observed_indices]) / (N-1)
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    
    # Add observation noise
    R = observation_variance * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Compute Kalman gain
    # K = P @ H^T @ S^{-1}
    # P @ H^T = anomalies.T @ obs_anomalies / (N-1)  (D, M)
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Apply localization to the cross-covariance
    # For each observed index m and each state index d, compute distance
    # and apply Gaussian localization
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            # Minimum image distance on the ring
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = _gaussian_localization(dist, localization_length)
    
    # Apply localization to cross-covariance
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Compute S^{-1} @ innovations for each member
    # innovations_m = observations[m] - H @ x_m = observations[m] - x_m[observed_indices]
    # K @ innovations = PHt_loc @ S^{-1} @ innovations
    
    # Solve S @ y = innovations for each member
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Compute updated anomalies
    # delta_x_m = K @ (y_m - H @ x_m) = PHt_loc @ S_inv @ (y_m - x_m[obs])
    updated_ensemble = ensemble.copy()
    for m in range(N):
        innovation = observations[m] - ensemble[m, observed_indices]  # (M,)
        # S_inv @ innovation
        Si_innov = S_inv @ innovation  # (M,)
        # PHt_loc @ Si_innov
        delta_x = PHt_loc @ Si_innov  # (D,)
        updated_ensemble[m] += delta_x
    
    # Compute ensemble spread before and after inflation decision
    # Adaptive inflation: compare observation-space spread to innovation
    # Simple approach: inflate if ensemble is too tight
    
    # Compute pre-inflation ensemble mean in observation space
    obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
    obs_spread_pre = np.std(ensemble[:, observed_indices], axis=0, ddof=1)
    
    # Compute innovation magnitude
    innovations = observations - ensemble[:, observed_indices]  # (N, M)
    innov_mean = np.mean(innovations, axis=0)
    innov_spread = np.std(innovations, axis=0, ddof=1)
    
    # Inflation factor: if spread is too small relative to innovations, inflate
    # Use a simple heuristic
    ratio = np.mean(innov_spread) / (np.mean(obs_spread_pre) + 1e-10)
    
    # Target ratio around 1.0; if ratio > 1, we need inflation
    if ratio > 1.0:
        # Inflate by a factor that brings spread closer to innovation scale
        # inflation = 1 + alpha * (ratio - 1)
        alpha = 0.3
        inflation = 1.0 + alpha * (ratio - 1.0)
        inflation = min(inflation, max_inflation)
    else:
        inflation = 1.0
    
    # Apply inflation about the ensemble mean
    if inflation > 1.0:
        updated_ensemble = ens_mean + inflation * (updated_ensemble - ens_mean)
    
    # Ensure finite values and clip to valid range
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Basic EKF with localization and inflation
    result = _ensemble_kalman_filter(
        ensemble, observations, observed_indices, observation_variance,
        localization_length=4.0,
        inflation_factor=1.05,
        max_inflation=1.5
    )
    
    # If we have time, try a second pass with slightly different parameters
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Try with slightly tighter localization
        result2 = _ensemble_kalman_filter(
            ensemble, observations, observed_indices, observation_variance,
            localization_length=3.5,
            inflation_factor=1.03,
            max_inflation=1.4
        )
        
        # Simple selection: use the one with smaller spread in observation space
        spread1 = np.std(result[:, observed_indices], axis=0, ddof=1)
        spread2 = np.std(result2[:, observed_indices], axis=0, ddof=1)
        
        # If both are reasonable, blend them
        # Weight by how close the spread is to the expected observation noise
        expected_spread = np.sqrt(observation_variance)
        w1 = 1.0 / (np.mean(np.abs(spread1 - expected_spread)) + 1e-10)
        w2 = 1.0 / (np.mean(np.abs(spread2 - expected_spread)) + 1e-10)
        
        total_w = w1 + w2
        if total_w > 0:
            result = (w1 / total_w) * result + (w2 / total_w) * result2
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
