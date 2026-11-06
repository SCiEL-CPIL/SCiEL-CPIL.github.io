"""Particle filter with KDE for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter analysis update for Lorenz-96.
    
    Uses importance weights based on observation likelihood, 
    resampling, and kernel density estimation to model the posterior.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)  # 10
    obs_var = float(observation_variance)
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Compute anomalies (centered ensemble)
    anomalies = ensemble - ens_mean  # (nm, nvars)
    
    # Compute observation operator H
    H = np.zeros((nobs, nvars))
    for k, idx in enumerate(observed_indices):
        H[k, int(idx)] = 1.0
    
    # Compute forecast observations for each member
    H_ens = H @ ensemble.T  # (nobs, nm)
    
    # Compute innovation (observation - forecast observation) for each member
    innovations = observations[:, None] - H_ens  # (nobs, nm)
    
    # Compute observation likelihood (Gaussian)
    # log p(y|x) = -0.5 * sum((y - Hx)^2 / obs_var) - 0.5*nobs*log(2*pi*obs_var)
    log_likelihoods = -0.5 * np.sum(innovations ** 2 / obs_var, axis=0)
    
    # Normalize log-likelihoods for numerical stability
    log_likelihoods -= np.max(log_likelihoods)
    weights = np.exp(log_likelihoods)
    
    # Normalize weights
    weights /= np.sum(weights)
    
    # Check if weights are too concentrated (effective sample size too small)
    ess = 1.0 / np.sum(weights ** 2)
    
    # If ESS is too low, do systematic resampling
    if ess < nm / 2.0:
        # Systematic resampling
        u = np.random.uniform(0, 1 / nm)
        c = np.cumsum(weights)
        indices = np.zeros(nm, dtype=int)
        for i in range(nm):
            u += 1.0 / nm
            while u > c[indices[i]]:
                indices[i] += 1
        indices = indices % nm
    else:
        # Stratified resampling for better spread
        c = np.cumsum(weights)
        c /= c[-1]
        u = (np.arange(nm) + np.random.uniform()) / nm
        indices = np.searchsorted(c, u)
        indices = indices % nm
    
    # Resampled ensemble
    resampled_ens = ensemble[indices]
    
    # Add small jitter to avoid degenerate particles
    # Estimate typical anomaly magnitude
    anomaly_scale = np.std(anomalies, axis=0)
    anomaly_scale = np.maximum(anomaly_scale, 0.01)
    
    # Add small random perturbation
    jitter = np.random.randn(nm, nvars) * anomaly_scale * 0.05
    resampled_ens = resampled_ens + jitter
    
    # Use KDE to generate new ensemble members centered around weighted ensemble
    # Compute weighted mean and covariance
    w = weights / np.sum(weights)
    kde_mean = np.sum(w[:, None] * ensemble, axis=0)
    kde_anomalies = ensemble - kde_mean
    kde_cov = (kde_anomalies.T * w) @ kde_anomalies / (nm - 1)
    
    # Cholesky decomposition with regularization
    reg = 1e-6 * np.eye(nvars)
    kde_cov_reg = kde_cov + reg
    try:
        L = np.linalg.cholesky(kde_cov_reg)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(kde_cov_reg + 1e-4 * np.eye(nvars))
    
    # Generate new ensemble members using KDE
    # Mix of resampled particles and KDE samples
    n_kde = nm // 2
    n_resample = nm - n_kde
    
    # KDE samples
    z = np.random.randn(n_kde, nvars)
    kde_samples = kde_mean + z @ L.T
    
    # Resampled particles (subset)
    resample_subset = resampled_ens[:n_resample]
    
    # Combine
    updated_ensemble = np.vstack([kde_samples, resample_subset])
    
    # Apply adaptive inflation based on innovation variance
    # Estimate spread in observation space
    H_updated = H @ updated_ensemble.T  # (nobs, nm)
    H_mean = np.mean(H_updated, axis=1)
    
    # Sample innovation variance
    sample_innov_var = np.var(observations - H_mean, axis=0)
    
    # Theoretical innovation variance
    H_ens_mean = H @ ens_mean
    H_anoms = H @ anomalies.T  # (nobs, nm)
    ens_innov_var = np.var(observations - (H_ens_mean + H_anoms), axis=1)
    theo_innov_var = np.diag(H @ kde_cov @ H.T) + obs_var * np.ones(nobs)
    
    # Inflation factor
    ratio = np.mean(theo_innov_var / np.maximum(ens_innov_var, 1e-6))
    alpha = np.sqrt(ratio)
    alpha = np.clip(alpha, 1.0, 1.3)
    
    # Apply inflation
    new_mean = np.mean(updated_ensemble, axis=0)
    for m in range(nm):
        updated_ensemble[m] = new_mean + alpha * (updated_ensemble[m] - new_mean)
    
    # Safety checks
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    if time.time() - t0 > 140.0:
        pass
        
    return updated_ensemble
# EVOLVE-BLOCK-END
