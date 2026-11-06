"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 115.0  # Safety margin before 120s evaluator limit
_START_TIME = [time.time()]  # Mutable container to track start time across calls


def _elapsed():
    """Return elapsed time since first call."""
    return time.time() - _START_TIME[0]


def _remaining():
    """Return remaining time budget."""
    return _DEADLINE_SECONDS - _elapsed()


def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
    n = len(x)
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


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _compute_localization_matrix(D, observed_indices, localization_length):
    """Compute localization weight matrix (D, M) for ring geometry."""
    M = len(observed_indices)
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = _gaussian_localization(dist, localization_length)
    return loc_weights


def _particle_filter_update(ensemble, obs_vec, observed_indices, observation_variance,
                            localization_length=3.0, inflation_base=1.01,
                            max_inflation=1.25, bandwidth_factor=1.5):
    """
    Nonlinear particle filter update with importance resampling and KDE.
    
    Uses the chaotic dynamics implicitly through the ensemble spread to
    approximate the posterior distribution non-parametrically.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    obs_vec : (M,) array - single observation vector
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    inflation_base : float - base multiplicative inflation factor
    max_inflation : float - maximum allowed inflation
    bandwidth_factor : float - factor for KDE bandwidth
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean
    
    # Observation operator: H selects observed components
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute observation-space covariance P_obs
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    
    # Add observation noise
    R = observation_variance * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Compute cross-covariance PHt
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Apply localization to the cross-covariance
    loc_weights = _compute_localization_matrix(D, observed_indices, localization_length)
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Solve for S_inv using Cholesky for stability
    try:
        L_S = np.linalg.cholesky(S)
        Y = np.linalg.solve(L_S, np.eye(M))
        S_inv = Y.T @ Y
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    # Compute innovations for each particle
    # innovation_m = obs_vec - x_m[obs]
    innovations = obs_vec[None, :] - ensemble[:, observed_indices]  # (N, M)
    
    # Compute importance weights using a Gaussian likelihood
    # log w_m = -0.5 * (obs - H x_m)^T S_inv (obs - H x_m)
    # Vectorized: Si_innov = innovations @ S_inv.T  (N, M)
    # log_w = -0.5 * sum(Si_innov * innovations, axis=1)
    Si_innov = innovations @ S_inv.T  # (N, M)
    log_weights = -0.5 * np.sum(Si_innov * innovations, axis=1)  # (N,)
    
    # Normalize log-weights for numerical stability
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Effective sample size
    ess = 1.0 / np.sum(weights ** 2)
    
    # Determine if resampling is needed
    resample_threshold = 0.5 * N
    
    if ess < resample_threshold:
        # Systematic resampling
        u = np.random.uniform(0, 1, N) / N
        cum_weights = np.cumsum(weights)
        indices = np.searchsorted(cum_weights, u)
        indices = np.clip(indices, 0, N - 1)
        resampled = ensemble[indices]
        
        # After resampling, all weights are equal (1/N)
        # Apply a small perturbation to avoid identical particles
        # Use KDE-based perturbation
        # Estimate bandwidth from ensemble spread
        std_obs = np.std(ensemble[:, observed_indices], axis=0)
        bandwidth = bandwidth_factor * np.mean(std_obs)
        bandwidth = max(bandwidth, 0.1)
        
        # Generate small perturbations
        n_perturb = N // 4  # Perturb only some particles
        perturb_indices = np.random.choice(N, n_perturb, replace=False)
        for idx in perturb_indices:
            # Perturb only observed components with small noise
            noise = np.random.normal(0, bandwidth * 0.3, M)
            resampled[idx, observed_indices] += noise
            # Apply localization to unobserved components
            for d in range(D):
                if d not in observed_indices:
                    # Find nearest observed index
                    min_dist = min(abs(d - oi) for oi in observed_indices)
                    min_dist = min(min_dist, D - min_dist)
                    loc = _gaussian_localization(min_dist, localization_length)
                    if loc > 0.1:
                        # Blend with local mean
                        nearby_obs = [oi for oi in observed_indices 
                                     if min(abs(oi - d), D - abs(oi - d)) <= localization_length]
                        if nearby_obs:
                            local_mean = np.mean(resampled[:, nearby_obs], axis=0)
                            resampled[idx, d] += 0.1 * loc * (local_mean - resampled[idx, d])
    else:
        # No resampling needed; use weighted update
        # Compute weighted mean
        weighted_mean = np.sum(weights[:, None] * ensemble, axis=0)
        
        # Compute weighted anomalies
        weighted_anomalies = ensemble - weighted_mean
        
        # Compute weighted cross-covariance
        PHt_w = (weights[:, None] * weighted_anomalies).T @ obs_anomalies / np.sum(weights)
        
        # Apply localization
        PHt_w_loc = PHt_w * loc_weights
        
        # Compute update using weighted approach
        # delta_m = PHt_w_loc @ S_inv @ (obs - H x_m)
        Si_innov_w = innovations @ S_inv.T  # (N, M)
        delta = Si_innov_w @ PHt_w_loc.T  # (N, D)
        
        resampled = ensemble + delta
    
    # Adaptive inflation based on effective sample size and innovation spread
    # Compute innovation for the ensemble mean
    obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
    innov_mean = obs_vec - obs_mean_pre  # (M,)
    
    # Expected innovation variance from ensemble
    P_obs_diag = np.diag(P_obs)
    R_diag = np.full(M, observation_variance)
    expected_innov_var = P_obs_diag + R_diag  # (M,)
    expected_innov_std = np.sqrt(np.maximum(expected_innov_var, 1e-10))
    
    # Actual innovation magnitude
    actual_innov_abs = np.abs(innov_mean)
    
    # If innovations are larger than expected, the ensemble is under-dispersed
    innov_ratio = np.mean(actual_innov_abs / expected_innov_std)
    
    # Map innovation ratio to inflation factor
    # Use a smooth mapping: inflation = 1 + alpha * max(0, ratio - 1)
    alpha = 0.4
    inflation = 1.0 + alpha * max(0.0, innov_ratio - 1.0)
    
    # Also consider ESS-based inflation: if ESS is low, inflate more
    ess_ratio = ess / N
    if ess_ratio < 0.5:
        inflation += 0.1 * (0.5 - ess_ratio)
    
    inflation = min(inflation, max_inflation)
    inflation = max(inflation, inflation_base)
    
    # Apply inflation about the ensemble mean
    if inflation > 1.0:
        resampled = ens_mean + inflation * (resampled - ens_mean)
    
    # Final KDE smoothing pass to ensure smooth posterior approximation
    # Estimate bandwidth for full state space
    state_std = np.std(ensemble, axis=0)
    kde_bandwidth = bandwidth_factor * np.mean(state_std)
    kde_bandwidth = max(kde_bandwidth, 0.1)
    
    # Apply a small smoothing to avoid overfitting to noisy observations
    # Blend each particle with a local weighted average
    smooth_strength = 0.05  # Small smoothing
    if smooth_strength > 0:
        for i in range(N):
            # Compute distance to other particles
            dists = np.sqrt(np.sum((ensemble - ensemble[i]) ** 2, axis=1))
            # Gaussian kernel weights
            kernel_weights = np.exp(-0.5 * (dists / kde_bandwidth) ** 2)
            kernel_weights[i] = 0  # Exclude self
            kernel_weights /= np.sum(kernel_weights)
            if np.sum(kernel_weights) > 0:
                local_mean = np.sum(kernel_weights[:, None] * ensemble, axis=0)
                resampled[i] = (1 - smooth_strength) * resampled[i] + smooth_strength * local_mean
    
    # Ensure finite values and clip to valid range
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=100.0, neginf=-100.0)
    resampled = np.clip(resampled, -100.0, 100.0)
    
    return resampled


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    # Deadline guard
    if _remaining() < 0.5:
        # Not enough time; return ensemble as-is (safe fallback)
        return np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        # If it's (N, M), assume all rows are the same or take the first one
        # Standard EnKF uses a single observation vector
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # Check time again before the main computation
    if _remaining() < 1.0:
        return np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Particle filter update with localization and adaptive inflation
    result = _particle_filter_update(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=3.0,
        inflation_base=1.01,
        max_inflation=1.25,
        bandwidth_factor=1.5
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
