import numpy as np
import time

# Module-level state for adaptive parameters
# We don't need adaptive inflation in Stochastic EnKF as it preserves spread naturally
# But we might need a very gentle adaptive mechanism if the ensemble collapses
# For now, we rely on the stochastic nature to maintain spread.
# We will use a fixed, small "spike" or just rely on the randomization.
# Actually, the prompt suggests "without the need for explicit multiplicative inflation".
# So we will remove the inflation logic entirely and trust the Stochastic EnKF.

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
    """
    Generate Gaussian localization matrix.
    Distance is circular distance on the ring of n points.
    """
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
    # Use a slightly tighter localization than before to reduce spurious correlations
    # 2.0 is often a good sweet spot for L96 with 20 variables
    _localization_length = 2.0
    L = _get_localization_matrix(n_vars, _localization_length)
    
    # 2. Stochastic Ensemble Kalman Filter (SE-Kalman)
    # Reference: Burgers et al. (2002) or similar.
    # The key is to perturb the observations with noise drawn from the ensemble covariance
    # such that the mean of the perturbed observations equals the true observation.
    
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
    
    # We need to generate perturbations for the observations.
    # Let y_pert = y + epsilon
    # where epsilon is a random vector with mean 0 and covariance C_oo.
    # To ensure the *mean* of the updated ensemble matches the Kalman gain applied to the *mean* innovation,
    # we use the "stochastic ensemble" approach.
    
    # Compute Cholesky of S for solving the linear system efficiently
    try:
        L_s = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # If S is not positive definite (can happen with localization), add jitter
        S_jittered = S + 1e-6 * np.eye(len(observed_indices))
        L_s = np.linalg.cholesky(S_jittered)
    
    # We need the Kalman Gain K = C_loc H^T S^-1
    # K = C_loc[:, observed_indices] @ inv(S)
    # We can compute this as C_loc[:, observed_indices] @ inv(L_s.T @ L_s)
    # inv(S) = inv(L_s.T) @ inv(L_s)
    
    C_h = C_loc[:, observed_indices] # 20 x 10
    
    # To avoid explicit inversion, we can solve for K directly or use the square root form.
    # However, for Stochastic EnKF, we often work with the "perturbed" observations.
    # The update for each member i is:
    # x_i^a = x_i + K (y + eps_i - H x_i)
    # where eps_i is a perturbation.
    
    # A common efficient implementation (e.g., from Houtekamer & Zhang 2006) uses a random rotation.
    # Let's use the simpler "perturbed observation" method.
    
    # We need eps_i such that mean(eps_i) = 0 and Cov(eps_i) = C_oo.
    # Since C_oo = H C_loc H^T, and we have the ensemble anomalies,
    # we can generate eps_i by projecting random noise through the localized covariance.
    
    # Actually, a robust way is to use the Cholesky factor of C_oo.
    # But C_oo might be singular.
    
    # Alternative: Use the "EnKF with perturbations" formula:
    # x_i^a = x_i + K (y - H x_i) + K C_oo^{1/2} z_i
    # where z_i is standard normal, and mean(z_i) = 0 is enforced by centering.
    
    # Let's compute K first.
    # K = C_h @ S^-1
    # S^-1 = inv(L_s.T) @ inv(L_s)
    # Let W = inv(L_s)
    # K = C_h @ W.T @ W
    
    W = np.linalg.inv(L_s)
    K = C_h @ W.T @ W # 20 x 10
    
    # Now we need the perturbations.
    # Let A be a matrix such that A A^T = C_oo.
    # We can use the Cholesky of C_oo if possible, or SVD.
    # C_oo = C_loc[obs, obs]
    # We can get a square root of C_oo from the ensemble anomalies.
    # Let anomalies_obs = anomalies[:, observed_indices] (16 x 10)
    # C_oo = (1/15) anomalies_obs.T @ anomalies_obs
    # So C_oo^{1/2} can be approximated by (1/sqrt(15)) * U * sqrt(s) * V^T from SVD of anomalies_obs.
    # Or simpler: We can generate perturbations by taking the ensemble anomalies themselves.
    
    # The "Stochastic EnKF" often implies:
    # y_i^pert = y + (x_i - m)_{obs}  <-- This is one variant (perturbing obs with ensemble spread)
    # But this introduces a bias if not careful.
    
    # Let's use the standard "Stochastic Ensemble Kalman Filter" (SE-Kalman) algorithm:
    # 1. Generate random vectors r_i ~ N(0, I)
    # 2. Center them: r_i = r_i - mean(r)
    # 3. Scale them to have covariance C_oo: eps_i = C_oo^{1/2} r_i
    # 4. Update: x_i^a = x_i + K (y + eps_i - H x_i)
    
    # To get C_oo^{1/2}, we can use the fact that C_oo is derived from the ensemble.
    # Let's use SVD of the observation anomalies.
    obs_anoms = anomalies[:, observed_indices] # 16 x 10
    U, s, Vt = np.linalg.svd(obs_anoms, full_matrices=False)
    # C_oo = (1/(N-1)) * V * diag(s^2) * V^T
    # So C_oo^{1/2} = (1/sqrt(N-1)) * V * diag(s) * V^T? No.
    # C_oo = (1/(N-1)) * (V diag(s) U^T) (U diag(s) V^T) = (1/(N-1)) V diag(s^2) V^T
    # So C_oo^{1/2} = (1/sqrt(N-1)) V diag(s) V^T? No, that's not a square root.
    # The square root is (1/sqrt(N-1)) V diag(s) U^T? No.
    # If C = A A^T, then A is a square root.
    # A = (1/sqrt(N-1)) obs_anoms.T ? No, obs_anoms is N x M.
    # obs_anoms.T @ obs_anoms / (N-1) = C_oo.
    # So A = obs_anoms.T / sqrt(N-1) is a "left" square root?
    # A A^T = obs_anoms.T @ obs_anoms / (N-1) = C_oo. Yes!
    # So C_oo^{1/2} (in the sense of A such that A A^T = C_oo) is obs_anoms.T / sqrt(N-1).
    # But we need to multiply this by r_i (10x1) to get eps_i (10x1).
    # eps_i = (obs_anoms.T / sqrt(N-1)) @ r_i
    # This eps_i will have covariance C_oo.
    
    # Let's generate r_i.
    rng = np.random.default_rng(42 + int(start_time * 1000) % 10000) # Seed based on time to be different each call?
    # Better to use a fixed seed or let it be random. The evaluator runs multiple times.
    # Using a fixed seed might be safer for reproducibility in testing, but random is fine for the task.
    # Let's use a random seed for true stochasticity.
    r = rng.standard_normal((n_members, len(observed_indices)))
    
    # Center the random vectors to ensure mean(eps) = 0
    r = r - np.mean(r, axis=0, keepdims=True)
    
    # Generate perturbations eps_i
    # eps = r @ (C_oo^{1/2})^T = r @ (obs_anoms / sqrt(N-1))
    # Wait, A = obs_anoms.T / sqrt(N-1). A is M x N.
    # We need eps (M x N).
    # eps_i = A^T r_i? No.
    # If A A^T = C, and we want eps ~ N(0, C), we can do eps = A^T z where z ~ N(0, I).
    # Here A = obs_anoms.T / sqrt(N-1) (M x N).
    # A^T = obs_anoms / sqrt(N-1) (N x M).
    # z is (M x N) matrix of random normals.
    # eps = (obs_anoms / sqrt(N-1)) @ z.T ?
    # Let's check dimensions.
    # obs_anoms: (N, M)
    # z: (M, N) -> z.T is (N, M)
    # (N, M) @ (N, M)^T? No.
    
    # Let's restart the perturbation generation logic.
    # We want eps_i (vector of length M) for each member i.
    # Cov(eps) = C_oo (M x M).
    # We have obs_anoms (N x M).
    # C_oo = obs_anoms.T @ obs_anoms / (N-1).
    # Let Z be a (N x M) matrix of i.i.d N(0,1).
    # Let eps = Z @ obs_anoms / (N-1) ?
    # Cov(eps) = E[eps eps^T] = E[ Z obs_anoms (obs_anoms^T Z^T) ] / (N-1)^2
    # = (1/(N-1)^2) obs_anoms (E[Z Z^T]) obs_anoms^T
    # E[Z Z^T] = N I_M (since Z is N x M, Z Z^T is M x M, diagonal is sum of N squares = N).
    # So Cov(eps) = (1/(N-1)^2) obs_anoms (N I) obs_anoms^T = (N/(N-1)^2) C_oo (N-1) = (N/(N-1)) C_oo.
    # This is close to C_oo but scaled by N/(N-1).
    # To get exactly C_oo, we can scale by sqrt((N-1)/N).
    
    Z = rng.standard_normal((n_members, len(observed_indices)))
    # Center Z to ensure mean(eps) = 0
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    
    # eps = Z @ obs_anoms / (N-1) * sqrt(N-1) ?
    # Let's just use the Cholesky of C_oo if it works, it's cleaner.
    try:
        L_oo = np.linalg.cholesky(C_oo)
        # eps = Z @ L_oo.T
        eps = Z @ L_oo.T
    except np.linalg.LinAlgError:
        # Fallback to SVD-based square root
        U, s, Vt = np.linalg.svd(C_oo, full_matrices=False)
        C_oo_sqrt = U * np.sqrt(s) # (M x M)
        eps = Z @ C_oo_sqrt.T
        
    # Check time
    if time.time() > deadline - 2.0:
        return result
        
    # Update each member
    # x_i^a = x_i + K (y + eps_i - H x_i)
    # H x_i = x_i[observed_indices]
    
    x_obs = result[:, observed_indices] # (N, M)
    y_pert = y[None, :] + eps # (N, M)
    
    # Innovation for each member: d_i = y_pert_i - x_obs_i
    d_all = y_pert - x_obs # (N, M)
    
    # Correction: K (20 x 10) @ d_all.T (10 x N) -> (20 x N)
    corr_all = K @ d_all.T
    
    # Update
    result = result + corr_all
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
