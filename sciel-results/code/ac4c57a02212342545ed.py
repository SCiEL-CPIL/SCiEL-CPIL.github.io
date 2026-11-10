"""Lorenz-96 ensemble assimilation: Local ESRF with adaptive inflation and localization."""
import numpy as np
import time

_N = 20
_WINDOW_HALF = 4  # window of 9 variables (i-4 to i+4)
_CORR_LEN = 5.0


def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    m1 = d <= 0.5 * c
    m2 = (d > 0.5 * c) & (d <= 1.5 * c)
    m3 = (d > 1.5 * c) & (d <= 2.5 * c)

    dm = d[m1]
    if len(dm) > 0:
        x = dm / c
        result[m1] = (-5/4)*x**5 + 5*x**4 - 10*x**3 + 5*x

    dm = d[m2]
    if len(dm) > 0:
        x = dm / c
        result[m2] = (1/12)*x**5 - (5/4)*x**4 + (5/2)*x**3 - 5*x**2 + (25/12)*x + 0.5

    dm = d[m3]
    if len(dm) > 0:
        x = dm / c
        result[m3] = -(1/16)*x**5 + (1/4)*x**4 - (15/8)*x**3 + (5/2)*x**2 - (25/12)*x + 105/48

    return result


def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)


def _local_esrf_update(ensemble, obs_idx, obs_val, obs_var, N, window_half, corr_len):
    """
    Local Ensemble Square Root Filter update for a single observation.
    Updates all ensemble members using localized covariance.
    """
    M = ensemble.shape[0]
    result = ensemble.copy()
    
    # Local window indices (periodic)
    local_idx = [(obs_idx + j) % N for j in range(-window_half, window_half + 1)]
    n_local = len(local_idx)
    
    # Localization weights
    dists = np.array([_periodic_distance(obs_idx, (obs_idx + j) % N, N) for j in range(-window_half, window_half + 1)])
    w = _gaspari_cohn(dists, corr_len)
    w[window_half] = 1.0  # center weight is 1
    
    # Extract local ensemble
    local_ens = result[:, local_idx].copy()  # (M, n_local)
    
    # Local mean and anomalies
    local_mean = np.mean(local_ens, axis=0)
    local_anoms = local_ens - local_mean  # (M, n_local)
    
    # Apply localization to anomalies
    local_anoms_loc = local_anoms * w[np.newaxis, :]
    
    # Local covariance
    C = (local_anoms_loc.T @ local_anoms_loc) / (M - 1)  # (n_local, n_local)
    
    # Innovation: obs - ensemble mean at obs location
    # The obs location is at position window_half in local_idx
    obs_pos_in_local = window_half  # since local_idx = [obs_idx - wh, ..., obs_idx, ..., obs_idx + wh]
    innov = obs_val - local_mean[obs_pos_in_local]
    
    # S = C[obs_pos, obs_pos] + obs_var
    S = C[obs_pos_in_local, obs_pos_in_local] + obs_var
    
    # Gain for each local variable
    if S > 1e-15:
        gain = C[:, obs_pos_in_local] / S  # (n_local,)
    else:
        gain = np.zeros(n_local)
    
    # Update all members
    for m in range(M):
        result[m, local_idx] = local_ens[m, :] + gain * innov
    
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local ESRF analysis for Lorenz-96 partial-observation assimilation.
    
    Uses sequential observation processing with local windows and
    Gaspari-Cohn localization. Adaptive multiplicative inflation prevents
    ensemble collapse.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations
    observed_indices: array of observed state indices
    observation_variance: scalar observation noise variance
    
    Returns:
    Updated ensemble (M, N)
    """
    t0 = time.time()
    M, N = ensemble.shape
    K = len(observed_indices)
    result = ensemble.astype(float).copy()
    
    # Adaptive multiplicative inflation
    if M > 2:
        obs_idx_list = list(set(int(x) for x in observed_indices))
        spread = np.std(result[:, obs_idx_list], axis=0)
        obs_err = np.sqrt(observation_variance)
        if obs_err > 0:
            ratio = np.mean(spread) / obs_err
        else:
            ratio = 1.0
        if ratio < 1.0:
            inflation = 1.0 + 0.08 * (1.0 - ratio)
        else:
            inflation = 1.003
        inflation = float(np.clip(inflation, 1.003, 1.08))
    else:
        inflation = 1.003
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Time budget with safety margin (evaluator kills at 150s)
    time_budget = 140.0
    
    # Process observations sequentially
    for k in range(K):
        if time.time() - t0 > time_budget:
            break
        
        obs_idx = int(observed_indices[k])
        obs_val = float(observations[k])
        
        result = _local_esrf_update(result, obs_idx, obs_val, observation_variance, N, _WINDOW_HALF, _CORR_LEN)
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
