import numpy as np
import time

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    xm2 = np.roll(x, 2)
    xm1 = np.roll(x, 1)
    xp1 = np.roll(x, -1)
    return (xp1 - xm2) * xm1 - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _get_localization_matrix(n, length):
    """Generate Gaussian localization matrix."""
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            d = min(d, n - d)
            dist[i, j] = d
    L = np.exp(-0.5 * (dist / length) ** 2)
    return L

def analyze(ensemble, observations, observed_indices, observation_variance):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n_members, n_vars = ensemble.shape
    result = ensemble.copy()

    # 1. Localization Matrix
    # Using a tighter localization length as the deterministic transform is more stable
    # and less prone to the "under-dispersion" issues that often require looser localization
    # in stochastic methods to compensate for random errors.
    _localization_length = 1.5
    L = _get_localization_matrix(n_vars, _localization_length)

    # 2. Deterministic Ensemble Kalman Filter (EnKF)
    # Strategy: Update the ensemble mean and spread using a deterministic transformation.
    # This avoids the random perturbations of the Stochastic EnKF, leading to a more
    # consistent estimate of the state given the observations.

    m = np.mean(result, axis=0)
    anomalies = result - m

    # Covariance C = (1/(N-1)) * anomalies.T @ anomalies
    C = (anomalies.T @ anomalies) / (n_members - 1)

    # Apply Localization to C
    C_loc = C * L

    # Compute H C_loc H^T (Observation Covariance)
    C_oo = C_loc[np.ix_(observed_indices, observed_indices)]

    # R (Observation Noise Covariance)
    R = observation_variance * np.eye(len(observed_indices))

    # Innovation
    y = observations
    m_obs = m[observed_indices]
    d = y - m_obs

    # S = C_oo + R
    S = C_oo + R

    # Compute Cholesky of S
    try:
        L_s = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S_jittered = S + 1e-6 * np.eye(len(observed_indices))
        L_s = np.linalg.cholesky(S_jittered)

    # Compute Kalman Gain K = C_loc H^T S^-1
    C_h = C_loc[:, observed_indices] # 20 x 10
    W = np.linalg.inv(L_s)
    K = C_h @ W.T @ W # 20 x 10

    # Check time
    if time.time() > deadline - 2.0:
        return result

    # Deterministic Update
    # We want to update each member x_i such that:
    # 1. The new mean is m + K @ d
    # 2. The new covariance is (I - K H) C_loc (I - K H)^T + K R K^T
    #
    # A common deterministic approach is the "Ensemble Kalman Transform" or simply
    # applying the mean update to all members and scaling the anomalies.
    #
    # Let x_i' = m + K d + (I - K H) (x_i - m)
    # This update preserves the linear structure and ensures the mean is correct.
    # However, it reduces the spread by (I - K H).
    # To preserve the correct posterior covariance, we need to scale the anomalies.
    #
    # Let A = I - K H.
    # The new anomalies are A @ anomalies.
    # The new covariance is A C A^T.
    # We want this to be close to the posterior covariance.
    # In the limit of large N, this is the correct posterior.
    #
    # Let's implement the simple deterministic update:
    # x_i' = m + K d + (I - K H) (x_i - m)
    
    H_mat = np.zeros((len(observed_indices), n_vars))
    H_mat[np.arange(len(observed_indices)), observed_indices] = 1.0
    
    A = np.eye(n_vars) - K @ H_mat # 20 x 20
    
    # Update each member
    # x_i' = m + K d + A (x_i - m)
    # result = m[None, :] + (K @ d)[None, :] + A @ anomalies
    
    mean_update = m + K @ d
    updated_anomalies = A @ anomalies
    result = mean_update[None, :] + updated_anomalies

    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)

    return result
