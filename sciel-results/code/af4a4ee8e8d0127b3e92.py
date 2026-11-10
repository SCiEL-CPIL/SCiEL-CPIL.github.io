"""ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# Precompute Gaspari-Cohn localization for N=20 ring topology
def _gc(d, c):
    """Gaspari-Cohn correlation function (vectorized)."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    # d <= c/2
    mask1 = d <= 0.5 * c
    r1 = d[mask1] / c
    result[mask1] = (-5.0/4.0) * r1**5 + 5.0 * r1**4 - 10.0 * r1**3 + 5.0 * r1
    # c/2 < d <= 3c/2
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    r2 = d[mask2] / c
    result[mask2] = (1.0/12.0) * r2**5 - (5.0/4.0) * r2**4 + (5.0/2.0) * r2**3 - 5.0 * r2**2 + (25.0/12.0) * r2 + 0.5
    # 3c/2 < d <= 5c/2
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    r3 = d[mask3] / c
    result[mask3] = (-1.0/16.0) * r3**5 + (1.0/4.0) * r3**4 - (5.0/12.0) * r3**3 + (1.0/4.0) * r3 + 105.0/128.0
    return result

# Precompute localization matrix for N=20
_N = 20
_CORR_LEN = 4.0  # correlation length in grid points
_idx = np.arange(_N)
_dist = np.minimum(np.abs(_idx[:, None] - _idx[None, :]), _N - np.abs(_idx[:, None] - _idx[None, :]))
_LOC = _gc(_dist, _CORR_LEN)
_LOC = 0.5 * (_LOC + _LOC.T)  # ensure symmetry
np.fill_diagonal(_LOC, 1.0)


def _localize_cov(cov, loc):
    """Apply localization to covariance matrix."""
    return cov * loc


def _adaptive_inflation(ensemble, observations, observed_indices, observation_variance):
    """Compute adaptive inflation factor to maintain ensemble spread."""
    n_ens, n_var = ensemble.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Ensemble mean and anomalies at observed points
    ens_mean = ensemble[:, obs_idx].mean(axis=0)
    ens_anom = ensemble[:, obs_idx] - ens_mean
    
    # Innovation: obs - ensemble mean
    innovation = observations - ens_mean
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Localized ensemble covariance at observed points
    ens_cov = (ens_anom.T @ ens_anom) / (n_ens - 1)
    # Localize
    obs_loc = _LOC[np.ix_(obs_idx, obs_idx)]
    ens_cov_loc = _localize_cov(ens_cov, obs_loc)
    
    # Total forecast covariance
    S = ens_cov_loc + R
    
    # Compute inflation factor
    # We want the inflated ensemble to match the observation spread
    # inflation_factor such that: (inflation^2 - 1) * trace(ens_cov_loc) + trace(R) matches innovation variance
    # Simpler: compute ratio of innovation variance to expected variance
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Mahalanobis distance squared
    innovation_vec = innovation.reshape(-1)
    mahalanobis_sq = innovation_vec @ S_inv @ innovation_vec
    
    # Expected Mahalanobis distance is n_obs
    # Inflation factor to match
    n_ens_minus_1 = n_ens - 1
    if mahalanobis_sq > 0:
        # inflation^2 = 1 + (mahalanobis_sq - n_obs) / (n_obs * (n_ens - 1))
        inflation_sq = 1.0 + max(0, (mahalanobis_sq - n_obs) / (n_obs * n_ens_minus_1))
        inflation = np.sqrt(inflation_sq)
    else:
        inflation = 1.0
    
    # Clip inflation to reasonable range
    inflation = np.clip(inflation, 1.0, 2.0)
    return inflation


def _esrf_update(ensemble, observations, observed_indices, observation_variance, inflation=1.0):
    """
    Local Ensemble Square Root Filter (ESRF) update.
    
    Parameters:
        ensemble: (n_ens, n_var) array
        observations: (n_obs,) array
        observed_indices: indices of observed variables
        observation_variance: scalar
        inflation: adaptive inflation factor
    
    Returns:
        updated ensemble: (n_ens, n_var)
    """
    n_ens, n_var = ensemble.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Apply inflation to anomalies
    ens_mean = ensemble.mean(axis=0)
    ens_anom = ensemble - ens_mean
    ensemble = ens_mean + inflation * ens_anom
    
    # Recompute mean after inflation
    ens_mean = ensemble.mean(axis=0)
    ens_anom = ensemble - ens_mean
    
    # Extract observed and unobserved parts
    obs_anom = ens_anom[:, obs_idx]  # (n_ens, n_obs)
    unobs_anom = ens_anom[:, ~np.isin(np.arange(n_var), obs_idx)] if n_var > n_obs else None
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Localized ensemble covariance at observed points
    ens_cov_obs = (obs_anom.T @ obs_anom) / (n_ens - 1)
    obs_loc = _LOC[np.ix_(obs_idx, obs_idx)]
    ens_cov_obs_loc = _localize_cov(ens_cov_obs, obs_loc)
    
    # Total forecast covariance at observed points
    S = ens_cov_obs_loc + R
    
    # Innovation
    innovation = observations - ens_mean[obs_idx]
    
    # ESRF transformation:
    # We want to find orthogonal transformation Q such that:
    # (I - K H) * inflated_anom = Q * inflated_anom
    # where K is the Kalman gain
    #
    # For ESRF, we solve:
    # (B_obs + R) * Q^T * Q = B_obs
    # where B_obs is the (localized) ensemble covariance
    #
    # Equivalently, we can use the square root form:
    # Let B_obs = W^T W / (n_ens - 1) where W is the anomaly matrix
    # Then we need Q such that Q^T Q = (B_obs + R)^{-1} B_obs
    #
    # A practical ESRF approach:
    # 1. Compute Kalman gain K = B H^T (H B H^T + R)^{-1}
    # 2. The update is x_a = x_f + K (y - H x_f)
    # 3. For the ensemble anomalies, apply a transformation that preserves spread
    
    # Compute Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Cross-covariance between all variables and observed variables
    ens_cov_all_obs = (ens_anom.T @ obs_anom) / (n_ens - 1)
    # Localize cross-covariance
    all_loc = _LOC[np.ix_(np.arange(n_var), obs_idx)]
    ens_cov_all_obs_loc = _localize_cov(ens_cov_all_obs, all_loc)
    
    # Kalman gain
    K = ens_cov_all_obs_loc @ S_inv  # (n_var, n_obs)
    
    # Analysis mean
    analysis_mean = ens_mean + K @ innovation
    
    # For ESRF, we need to transform the anomalies
    # The key idea: find orthogonal Q such that the transformed anomalies
    # have the correct analysis covariance
    
    # Analysis covariance (theoretical)
    B_f = (ens_anom.T @ ens_anom) / (n_ens - 1)
    # Localize
    B_f_loc = _localize_cov(B_f, _LOC)
    B_a = B_f_loc - K @ S @ K.T  # (n_var, n_var)
    
    # For ESRF, we use the fact that:
    # B_a = (I - K H) B_f (I - K H)^T + K R K^T
    # But for orthogonal transformation, we want:
    # B_a = Q B_f Q^T
    # This requires B_a and B_f to have the same eigenvalues, which is not generally true.
    #
    # Instead, ESRF uses a different approach:
    # We transform the ensemble such that the mean is updated and the covariance
    # matches B_a. This can be done via Cholesky or SVD.
    
    # Practical ESRF: use the "direct" method
    # 1. Form the augmented system
    # 2. Apply orthogonal transformation
    
    # A simpler approach that works well: use the Kalman update for the mean
    # and then adjust the anomalies to have the correct covariance.
    
    # Compute the analysis anomalies using the Kalman update
    analysis_anom = ens_anom - K @ (obs_anom - obs_anom.mean(axis=0))  # This isn't quite right
    
    # Let's use a cleaner ESRF formulation:
    # The ESRF update can be written as:
    # x_a = x_f + K (y - H x_f)
    # For each ensemble member:
    # x_a^i = x_f^i + K (y - H x_f^i)  -- this is the linear update
    # But this changes the spread. ESRF corrects this with an orthogonal transformation.
    
    # Linear Kalman update for each member
    linear_anom = ens_anom - K @ obs_anom
    
    # Now apply orthogonal transformation to preserve spread
    # We want: Q @ linear_anom = analysis_anom_target
    # where analysis_anom_target has covariance B_a
    
    # Compute target covariance
    # B_a = B_f - K S K^T (for the unlocalized case)
    # With localization, it's more complex
    
    # Practical approach: use the fact that for the observed variables,
    # the analysis covariance should be (B_obs^{-1} + R^{-1})^{-1}
    # and for unobserved, it should be the forecast covariance adjusted
    
    # Let's use a simpler but effective approach:
    # 1. Apply the linear Kalman update
    # 2. Then apply a spread correction (inflation/deflation)
    
    # Actually, let's use the proper ESRF algorithm:
    # Reference: Anderson (2003)
    
    # Step 1: Compute the "analysis" anomalies using the linear update
    # Step 2: Find orthogonal Q such that Q^T Q = B_a B_f^{-1} (in some sense)
    
    # A practical implementation:
    # 1. Compute B_f and B_a
    # 2. Compute B_f^{-1/2} and B_a^{1/2}
    # 3. Q = B_a^{1/2} B_f^{-1/2}
    # 4. But Q must be orthogonal, so use SVD: Q = U V^T where B_a^{1/2} B_f^{-1/2} = U S V^T
    
    try:
        # Cholesky of B_f (add small regularization)
        B_f_reg = B_f_loc + 1e-8 * np.eye(n_var)
        L_f = np.linalg.cholesky(B_f_reg)
        
        # B_a should be positive definite
        B_a_reg = B_a + 1e-8 * np.eye(n_var)
        # Ensure symmetry
        B_a_reg = 0.5 * (B_a_reg + B_a_reg.T)
        
        # Check if B_a is positive definite
        try:
            L_a = np.linalg.cholesky(B_a_reg)
        except np.linalg.LinAlgError:
            # If not PD, use eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(B_a_reg)
            eigvals = np.clip(eigvals, 1e-8, None)
            B_a_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L_a = np.linalg.cholesky(B_a_reg)
        
        # Q = L_a L_f^{-T}
        # L_f^{-T} = (L_f^{-1})^T
        L_f_inv_T = np.linalg.inv(L_f).T
        Q = L_a @ L_f_inv_T
        
        # Q should be orthogonal. Use SVD to project to nearest orthogonal matrix
        U, _, Vt = np.linalg.svd(Q)
        Q_orth = U @ Vt
        
        # Apply orthogonal transformation to the linearly updated anomalies
        # linear_anom is (n_ens, n_var)
        # We want: analysis_anom = linear_anom @ Q_orth^T
        analysis_anom = linear_anom @ Q_orth.T
        
    except Exception:
        # Fallback to linear Kalman update
        analysis_anom = linear_anom
    
    # Construct updated ensemble
    updated_ensemble = analysis_mean[None, :] + analysis_anom
    
    # Ensure finite values and clip to [-100, 100]
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    ESRF analysis update for Lorenz-96 partial-observation assimilation.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: array of observed values (even indices)
        observed_indices: indices of observed variables
        observation_variance: scalar observation noise variance
    
    Returns:
        Updated ensemble with same shape (16, 20)
    """
    start_time = time.time()
    
    # Compute adaptive inflation
    inflation = _adaptive_inflation(ensemble, observations, observed_indices, observation_variance)
    
    # Apply ESRF update
    updated_ensemble = _esrf_update(ensemble, observations, observed_indices, observation_variance, inflation)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin before 120s limit
        # Return the best result we have
        return updated_ensemble
    
    return updated_ensemble
