"""Ensemble assimilation for partial-observation Lorenz-96 using localized ensemble square-root filter."""
import numpy as np
import time

def _l96_rhs(x, F):
    """Compute Lorenz-96 RHS for a single state vector x of length 20."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """One RK4 step for Lorenz-96."""
    k1 = _l96_rhs(x, F)
    k2 = _l96_rhs(x + 0.5 * dt * k1, F)
    k3 = _l96_rhs(x + 0.5 * dt * k2, F)
    k4 = _l96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _gaspari_cohn(d, L):
    """Gaspari-Cohn localization function."""
    if d <= L:
        return 1.0
    elif d <= 2 * L:
        t = (d - L) / L
        return 0.5 * (t**5 - 5 * t**4 + 10 * t**3 - 10 * t**2 + 5 * t + 1)
    else:
        return 0.0

def _localization_matrix(n, L):
    """Build an n x n localization matrix with periodic distance."""
    loc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            d = min(d, n - d)
            loc[i, j] = _gaspari_cohn(d, L)
    return loc

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized ensemble square-root filter (ESRF) analysis update.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of states
    observations : (10,) array - observed values at observed_indices
    observed_indices : (10,) array - indices of observed coordinates (even indices)
    observation_variance : float - variance of observation noise
    
    Returns:
    --------
    updated ensemble with same (16, 20) shape
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    ens = np.array(ensemble, dtype=np.float64)
    n_ens = ens.shape[0]
    n_state = ens.shape[1]
    n_obs = len(observations)
    obs_idx = np.array(observed_indices, dtype=int)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ens, axis=0)
    anomalies = ens - mean[None, :]
    
    # Build localization matrix for the full state space
    L_loc = 5.0  # localization scale in index units
    loc_mat = _localization_matrix(n_state, L_loc)
    
    # Localize the ensemble covariance: apply localization to anomalies
    # P = (1/(N-1)) * A A^T, localized: P_loc[i,j] = loc[i,j] * P[i,j]
    # We work with localized anomalies for the update
    
    # Observation operator: H extracts observed_indices
    # H is n_obs x n_state, H[k, i] = 1 if i == obs_idx[k]
    
    # Compute localized background covariance
    # P_bg = (1/(N-1)) * sum_k anomalies[k,:] outer anomalies[k,:]
    # Localized: P_loc[i,j] = loc_mat[i,j] * P_bg[i,j]
    
    # For efficiency with small ensemble, use the square-root form
    # The ESRF update: x_a = x_b + P_bg H^T (H P_bg H^T + R)^{-1} (y - H x_b)
    
    # With localization, we approximate by updating each component with localized covariance
    
    # Simpler approach: use a localized version of the update
    # For each state variable i, compute its covariance with observed variables
    # using the localization-weighted anomalies
    
    # Compute the observation-space covariance matrix C = H P_loc H^T + R
    # C[k1, k2] = sum_i loc_mat[obs_idx[k1], i] * P_bg[i, obs_idx[k2]] + delta[k1,k2]*R
    
    # P_bg[i,j] = (1/(N-1)) * sum_n anomalies[n,i] * anomalies[n,j]
    
    # Let's compute C directly
    R = observation_variance
    C = np.zeros((n_obs, n_obs))
    
    for k1 in range(n_obs):
        for k2 in range(n_obs):
            i1 = obs_idx[k1]
            i2 = obs_idx[k2]
            # C[k1,k2] = (1/(N-1)) * sum_n loc_mat[i1, i2] * anomalies[n, i1] * anomalies[n, i2]
            # Wait, that's not quite right. Let me think more carefully.
            # P_loc[i,j] = loc_mat[i,j] * (1/(N-1)) * sum_n a[n,i]*a[n,j]
            # C = H P_loc H^T + R
            # C[k1,k2] = P_loc[obs_idx[k1], obs_idx[k2]] + (R if k1==k2 else 0)
            #          = loc_mat[obs_idx[k1], obs_idx[k2]] * (1/(N-1)) * sum_n a[n,obs_idx[k1]] * a[n,obs_idx[k2]] + ...
            
            if k1 == k2:
                C[k1, k2] = loc_mat[i1, i2] * np.sum(anomalies[:, i1] * anomalies[:, i2]) / (n_ens - 1) + R
            else:
                C[k1, k2] = loc_mat[i1, i2] * np.sum(anomalies[:, i1] * anomalies[:, i2]) / (n_ens - 1)
    
    # Ensure C is symmetric and positive definite
    C = 0.5 * (C + C.T)
    
    # Add small regularization for numerical stability
    C += 1e-10 * np.eye(n_obs)
    
    # Compute innovation
    innovation = observations - mean[obs_idx]
    
    # Compute Kalman gain in observation space
    # We need K = P_loc H^T C^{-1}
    # For each state variable i: K[i, k] = P_loc[i, obs_idx[k]] * (C^{-1})[k, :]
    
    # Compute C^{-1}
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C)
    
    # Compute the update for each state variable
    # x_a[i] = x_b[i] + sum_k P_loc[i, obs_idx[k]] * (C_inv @ innovation)[k]
    # P_loc[i, obs_idx[k]] = loc_mat[i, obs_idx[k]] * (1/(N-1)) * sum_n a[n,i]*a[n,obs_idx[k]]
    
    Ci = C_inv @ innovation  # (n_obs,)
    
    # For each ensemble member, compute the update
    # The deterministic part (mean update):
    # mean_a[i] = mean_b[i] + sum_k P_loc[i, obs_idx[k]] * Ci[k]
    
    # The stochastic part: for each member n:
    # a_a[n, i] = a_b[n, i] - (1/(N-1)) * sum_k loc_mat[i, obs_idx[k]] * a_b[n, obs_idx[k]] * Ci[k] * (N-1) / ... 
    # Actually, let's use the standard ESRF formula.
    
    # Standard ESRF:
    # x_a = x_b + K (y - H x_b)  for the mean
    # For anomalies: a_a = (I - K H) a_b
    # where K is the Kalman gain matrix (n_state x n_obs)
    
    # K[i, k] = P_loc[i, obs_idx[k]] * C_inv[k, :]
    # But this is for the mean. For the full ensemble update in ESRF:
    
    # Let's compute K as (n_state x n_obs)
    K = np.zeros((n_state, n_obs))
    for i in range(n_state):
        for k in range(n_obs):
            j = obs_idx[k]
            # P_loc[i, j] = loc_mat[i, j] * (1/(N-1)) * sum_n a[n,i]*a[n,j]
            P_loc_ij = loc_mat[i, j] * np.sum(anomalies[:, i] * anomalies[:, j]) / (n_ens - 1)
            K[i, k] = P_loc_ij
    
    # Now K = K @ C_inv
    K = K @ C_inv  # (n_state x n_obs)
    
    # Update mean
    mean_a = mean + K @ innovation
    
    # Update anomalies: a_a = (I - K H) a_b
    # (K H)[i, j] = sum_k K[i, k] * H[k, j] = K[i, obs_idx[j]] if j is observed, else 0
    KH = np.zeros((n_state, n_state))
    for k in range(n_obs):
        j = obs_idx[k]
        KH[:, j] = K[:, k]
    
    # a_a = (I - KH) a_b
    I_KH = np.eye(n_state) - KH
    anomalies_a = I_KH @ anomalies.T  # (n_state x n_ens)
    anomalies_a = anomalies_a.T  # (n_ens x n_state)
    
    # Reconstruct ensemble
    ens_a = mean_a[None, :] + anomalies_a
    
    # Check for non-finite values
    if not np.all(np.isfinite(ens_a)):
        # Fallback to simple nudging
        ens_a = ens.copy()
        ens_a[:, observed_indices] += 0.5 * (observations - ens_a[:, observed_indices])
    
    # Clip to [-100, 100]
    ens_a = np.clip(ens_a, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return ens_a
    
    return ens_a
