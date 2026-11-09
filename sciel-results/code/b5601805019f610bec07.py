"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Compute the Lorenz-96 RHS for a single state vector."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5*dt*k1, F)
    k3 = _lorenz96_rhs(x + 0.5*dt*k2, F)
    k4 = _lorenz96_rhs(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-KF) update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array
        Current ensemble of 16 members, 20 state variables.
    observations : (16, 10) array or (10,) array
        Noisy observations of the 10 observed (even-indexed) coordinates.
    observed_indices : array of int
        Indices of observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : (16, 20) array
        Updated ensemble after the SE-KF analysis step.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    N = ensemble.shape[0]  # 16 ensemble members
    dim = ensemble.shape[1]  # 20 state variables
    obs_dim = len(observed_indices)
    
    # Ensure observations is 2D (N, obs_dim)
    if observations.ndim == 1:
        observations = np.tile(observations, (N, 1))
    
    # Work in float64 for accuracy
    ens = ensemble.astype(np.float64)
    obs = observations.astype(np.float64)
    
    # --- SE-KF Analysis ---
    
    # 1. Compute ensemble mean and anomalies
    mean = ens.mean(axis=0)
    anomalies = ens - mean[None, :]
    
    # 2. Compute ensemble covariance (with bias correction)
    # C_ee = (1/(N-1)) * anomalies^T @ anomalies
    C_ee = (anomalies.T @ anomalies) / (N - 1.0)
    
    # 3. Compute ensemble-observation covariance
    # C_eo = (1/(N-1)) * anomalies[:, observed_indices].T @ (obs - mean[observed_indices])
    obs_mean = obs.mean(axis=0)
    obs_anomalies = obs - obs_mean[None, :]
    C_eo = (anomalies[:, observed_indices].T @ obs_anomalies) / (N - 1.0)
    
    # 4. Observation covariance
    R = np.eye(obs_dim) * observation_variance
    
    # 5. Compute Kalman gain
    # K = C_eo @ inv(C_oo + R)
    # C_oo = (1/(N-1)) * obs_anomalies^T @ obs_anomalies
    C_oo = (obs_anomalies.T @ obs_anomalies) / (N - 1.0)
    S = C_oo + R  # Innovation covariance
    
    # Solve for gain: K = C_eo @ S^{-1}
    # Use least squares / solve for numerical stability
    try:
        K = C_eo @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        K = C_eo @ np.linalg.pinv(S)
    
    # 6. Compute analysis for observed variables
    # For SE-KF, we use the stochastic formulation:
    # x_a = x_f + K(y - H x_f) + sqrt((I - K H) B) * eps
    # where B is the ensemble covariance, H is the observation operator,
    # and eps is a standard normal random vector.
    
    # For the observed variables, the analysis mean is:
    # mean_a[obs] = mean_f[obs] + K[:, obs_dim] @ (obs - mean_f[obs])
    # But we need to update the full state.
    
    # Full-state analysis mean:
    # mean_a = mean_f + K @ (obs - mean_f[observed_indices])
    innovation = obs - ens[:, observed_indices]  # (N, obs_dim)
    
    # Stochastic SE-KF update:
    # Each member gets: x_a = x_f + K @ (y - H x_f) + sqrt((I - K H) C_ee) * eps
    # But since we only observe half the variables, we need to be careful.
    
    # Alternative approach: use the standard SE-KF formulation where
    # we perturb both the ensemble and observations with a shared rotation.
    
    # Generate a random orthogonal matrix for the perturbation
    # Use a simple approach: generate random normal and orthogonalize
    rng = np.random.default_rng()
    
    # For the SE-KF, we can use the following formulation:
    # x_a = x_f + K @ (y - H x_f) + (I - K H) @ sqrt(C_ee) @ eps
    # where eps ~ N(0, I)
    
    # Compute (I - K H) @ sqrt(C_ee)
    # H is a selection matrix: H @ x = x[observed_indices]
    # K is (dim, obs_dim), H is (obs_dim, dim)
    
    # (I - K H) is (dim, dim)
    H = np.zeros((obs_dim, dim))
    for j, idx in enumerate(observed_indices):
        H[j, idx] = 1.0
    
    KH = K @ H  # (dim, dim)
    I_KH = np.eye(dim) - KH
    
    # sqrt(C_ee) via Cholesky or SVD
    try:
        L = np.linalg.cholesky(C_ee + 1e-10 * np.eye(dim))
    except np.linalg.LinAlgError:
        # Use SVD for non-PSD matrices
        U, s, Vt = np.linalg.svd(C_ee)
        L = U @ np.diag(np.sqrt(np.maximum(s, 0)))
    
    # Generate random perturbations
    eps = rng.standard_normal((dim, N))
    
    # Stochastic update for each member
    update_mean = K @ (obs - ens[:, observed_indices]).T  # (dim, N)
    stochastic_term = (I_KH @ L) @ eps  # (dim, N)
    
    ens_updated = ens.T + update_mean + stochastic_term
    ens_updated = ens_updated.T  # Back to (N, dim)
    
    # Ensure the ensemble mean matches the deterministic analysis mean
    # The stochastic term should have zero mean in expectation, but let's
    # correct for any drift to ensure the mean is exact
    target_mean = mean + K @ (obs_mean - mean[observed_indices])
    ens_updated += (target_mean - ens_updated.mean(axis=0))[None, :]
    
    # Clip to valid range
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return ens_updated.astype(np.float64)
    
    return ens_updated.astype(np.float64)
# EVOLVE-BLOCK-END
