"""Combined ensemble assimilation for Lorenz-96 partial observation using Stochastic EnKI."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Inversion (EnKI) assimilation for Lorenz-96.
    Uses iterative updates to handle non-linearities and sparse observations.
    Includes Gaussian localization and adaptive inflation.
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)
    L = len(obs_idx)

    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        # Simple fallback: blend with observation
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result

    X = np.array(ensemble, dtype=np.float64)
    y_obs = np.asarray(observations, dtype=np.float64)
    if y_obs.ndim == 1:
        y_obs = np.tile(y_obs, (M, 1))

    # Precompute localization matrix
    state_idx = np.arange(N)
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, N - diffs)
    L_loc = 3.0  # Correlation length scale tuned for L96
    loc_matrix = np.exp(-0.5 * (distances / L_loc) ** 2)  # (N, L)

    # Number of EnKI iterations
    n_iter = 3

    for k in range(n_iter):
        # 1. Compute background statistics from current ensemble
        mean_b = np.mean(X, axis=0)
        W_b = X - mean_b  # (M, N)
        
        # 2. Compute background covariance (N x N)
        # Using sample covariance
        P_b = (W_b.T @ W_b) / (M - 1)
        
        # 3. Localize cross-covariance
        P_xo = P_b[:, obs_idx]  # (N, L)
        P_xo_loc = P_xo * loc_matrix  # (N, L)
        
        # 4. Observation covariance (localized for consistency, though R is diagonal)
        P_oo = P_b[np.ix_(obs_idx, obs_idx)]
        S = P_oo + obs_var * np.eye(L)
        
        # 5. Compute Kalman Gain
        # K = P_xo_loc @ S^{-1}
        try:
            # Solve S.T @ Z = P_xo_loc.T for efficiency and stability
            S_inv_PbH = np.linalg.solve(S.T, P_xo_loc.T)
            K = S_inv_PbH.T  # (N, L)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = P_xo_loc @ S_inv

        # 6. Iterative EnKI Update
        # Update each ensemble member: x_new = x_old + K @ (y - H @ x_old)
        # H @ x_old is just x_old[:, obs_idx]
        # Innovation for member i: y_obs[i] - X[i, obs_idx]
        innovations = y_obs - X[:, obs_idx]  # (M, L)
        
        # Apply gain to innovations
        # Delta X = innovations @ K.T  (M, L) @ (L, N) -> (M, N)
        delta_X = innovations @ K.T
        
        # Update ensemble
        X = X + delta_X
        
        # 7. Adaptive Inflation (to prevent ensemble collapse due to iterative updates)
        # Calculate innovation variance vs expected
        # Expected innovation variance is trace(S) / L
        expected_innov_var = np.mean(np.diag(S))
        actual_innov_var = np.var(innovations, axis=0).mean() + 1e-10
        ratio = actual_innov_var / expected_innov_var
        
        # Inflation factor
        # If actual variance is lower than expected, we are under-dispersed -> inflate
        # If higher, we are over-dispersed -> deflate (but usually we clamp at 1.0 to avoid deflation in filter mode, 
        # but in EnKI/inversion mode, deflation can be useful to converge. However, for stability, we usually only inflate or use symmetric inflation.)
        # Let's use a symmetric inflation factor clipped to [0.95, 1.1]
        inflation_factor = np.sqrt(ratio)
        inflation_factor = np.clip(inflation_factor, 0.95, 1.1)
        
        # Apply inflation to perturbations around the new mean
        mean_a = np.mean(X, axis=0)
        W_a = X - mean_a
        W_a *= inflation_factor
        X = mean_a + W_a

        # 8. Safety: ensure finite and clip
        X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
        np.clip(X, -100.0, 100.0, out=X)

    return X

# EVOLVE-BLOCK-END
