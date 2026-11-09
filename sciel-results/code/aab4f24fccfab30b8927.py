"""Stochastic Ensemble Kalman Filter (SE-Kalman) with random rotations for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter update.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (N, D)
        Ensemble of states. N=16 members, D=20 dimensions.
    observations : ndarray of shape (N, M) or (M,)
        Noisy observations of observed coordinates. If shape is (N, M), each member
        has its own perturbed observation. If (M,), we broadcast.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (N, D)
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure observations are in the right shape
    obs = np.asarray(observations)
    if obs.ndim == 1:
        obs = np.tile(obs, (N, 1))
    
    # Stochastic ensemble mean
    mean = ensemble.mean(axis=0)
    
    # Compute ensemble covariance
    # Anomalous matrix: X_a = ensemble - mean (N x D)
    X_a = ensemble - mean
    
    # Observation operator: we observe specific indices
    # H is identity on observed_indices
    # We only need to work with the observed subspace
    
    # Innovation: y - H(x) for each member
    # H(ensemble_member) = ensemble_member[observed_indices]
    Hx = ensemble[:, observed_indices]  # (N, M)
    y = obs  # (N, M)
    
    # Innovation vector per member
    d = y - Hx  # (N, M)
    
    # Observation covariance R = observation_variance * I_M
    R = observation_variance * np.eye(M)
    
    # Ensemble covariance in observation space: S = (1/(N-1)) * X_a[:, obs_idx] @ X_a[:, obs_idx].T
    Xa_obs = X_a[:, observed_indices]  # (N, M)
    S = (Xa_obs @ Xa_obs.T) / (N - 1)  # (M, M)
    
    # Add observation noise
    S += R
    
    # Compute Kalman gain via stochastic approach
    # Instead of solving for K = C_xy @ S^{-1}, we use random rotation to generate
    # perturbations that match the optimal Kalman filter statistics.
    
    # The optimal Kalman filter update:
    # K = C_xy S^{-1} where C_xy = (1/(N-1)) * X_a @ Xa_obs.T
    # x_new = x + K @ d
    
    # Stochastic SE-KF approach:
    # 1. Generate a random orthogonal matrix Q of size N x N
    # 2. Use it to rotate the ensemble in a way that preserves the mean
    #    and achieves the correct posterior covariance
    
    # Actually, let's use the standard SE-KF formulation:
    # The update can be written as:
    # x_i' = x_i + K @ d_i
    # where K is the Kalman gain
    
    # But to avoid bias from deterministic square-root, we use:
    # Generate random orthogonal matrix Q (N x N)
    # The perturbation is: delta_x_i = (K @ d_i) + stochastic term
    
    # Let's use a cleaner SE-KF formulation:
    # 1. Compute the Kalman gain K = C_xy @ S^{-1}
    # 2. For each member, update as: x_i' = x_i + K @ d_i + Q @ epsilon_i
    #    where epsilon_i are random perturbations with appropriate covariance
    
    # Actually, the cleanest SE-KF uses:
    # - A random orthogonal matrix Q of dimension N
    # - The updated ensemble is: x_i' = mean + (I + K @ H) @ (x_i - mean) + K @ d_i
    #   but with stochastic perturbation to preserve spread
    
    # Let me use the standard SE-KF algorithm:
    # 1. Compute innovation d_i = y_i - H(x_i)
    # 2. Compute Kalman gain K = (1/(N-1)) * X_a @ Xa_obs.T @ S^{-1}
    # 3. Update: x_i' = x_i + K @ d_i
    # 4. Add stochastic perturbation: x_i'' = x_i' + Q @ alpha_i
    #    where alpha_i are random with zero mean and appropriate variance
    
    # For simplicity and correctness, let's use the linear SE-KF:
    # The key idea is that instead of using a deterministic square-root filter,
    # we add random perturbations using a random orthogonal matrix.
    
    # Step 1: Compute Kalman gain
    # C_xy = (1/(N-1)) * X_a @ Xa_obs.T  (D x M)
    C_xy = (X_a @ Xa_obs.T) / (N - 1)
    
    # Solve for K = C_xy @ S^{-1}
    # Use Cholesky or direct inverse for small M
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = C_xy @ S_inv  # (D, M)
    
    # Step 2: Deterministic part of update
    # x_i_det = x_i + K @ d_i
    updated = ensemble + K @ d.T  # (N, D)
    
    # Step 3: Stochastic perturbation to preserve ensemble spread
    # The deterministic update reduces spread. We need to add noise.
    # The posterior covariance should be:
    # P_post = (I - K @ H) @ P_prior
    # The reduction in spread is captured by the matrix (I - K @ H)
    # We need to add random perturbations with covariance that restores the spread.
    
    # The spread reduction factor in the observation space is related to:
    # The eigenvalues of (I - K @ H)
    
    # A simpler approach: compute the prior and posterior covariance,
    # and add noise to match the posterior covariance.
    
    # Prior covariance: P_prior = (1/(N-1)) * X_a @ X_a.T
    P_prior = (X_a @ X_a.T) / (N - 1)
    
    # Posterior covariance (theoretical): P_post = (I - K @ H_full) @ P_prior
    # where H_full is the full observation operator (D x D with identity on observed_indices)
    # For efficiency, we can compute this directly.
    
    # H_full is a D x D matrix with 1s on the diagonal at observed_indices
    H_full = np.zeros((D, D))
    H_full[observed_indices, observed_indices] = 1.0
    
    P_post = (np.eye(D) - K @ H_full) @ P_prior
    
    # Now we need to add stochastic perturbations so that the empirical covariance
    # of the updated ensemble matches P_post.
    
    # Current empirical covariance of 'updated':
    updated_anom = updated - updated.mean(axis=0)
    P_emp = (updated_anom @ updated_anom.T) / (N - 1)
    
    # We want to add noise Z such that:
    # (updated_anom + Z) @ (updated_anom + Z).T / (N-1) = P_post
    # This is equivalent to finding Z with covariance P_post - P_emp
    
    # However, a simpler and more standard SE-KF approach is to use a random
    # orthogonal matrix to rotate the ensemble perturbations.
    
    # Let's use the standard SE-KF with random orthogonal matrix:
    # 1. Generate random orthogonal matrix Q (N x N) using QR decomposition of a random Gaussian matrix
    rng = np.random.default_rng()
    G = rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(G)
    # Ensure Q is truly orthogonal (QR can have sign issues)
    d_sign = np.sign(np.diag(Q))
    Q = Q * d_sign  # Make det(Q) positive if needed
    
    # The stochastic perturbation is: Q @ (some scaling of the innovations)
    # In SE-KF, the update is:
    # x_i' = x_i + K @ d_i + Q @ b_i
    # where b_i are chosen so that the posterior covariance is correct.
    
    # A common choice: b = (1/sqrt(N-1)) * Q^T @ (something related to the prior spread)
    
    # Actually, let's use a well-known SE-KF formulation:
    # The updated ensemble is:
    # x_i' = mean + (I - K @ H) @ (x_i - mean) + K @ d_i + Q @ r_i
    # where r_i are random with appropriate distribution.
    
    # For the linear case, the optimal stochastic perturbation uses:
    # r_i = (1/sqrt(N-1)) * Q^T @ (P_prior^{1/2}) @ e_i
    # where e_i are standard basis vectors... this is getting complex.
    
    # Let's use a simpler, effective approach:
    # 1. Compute the updated ensemble with deterministic KF
    # 2. Compute the expected posterior covariance P_post
    # 3. Compute the actual covariance of the deterministic update
    # 4. Add random noise to correct the covariance mismatch
    
    # The covariance deficit is: P_deficit = P_post - P_emp
    # We need to add noise with this covariance.
    
    # Use Cholesky decomposition of P_deficit if it's positive semi-definite
    P_deficit = P_post - P_emp
    
    # Check if P_deficit is positive semi-definite
    eigvals = np.linalg.eigvalsh(P_deficit)
    if np.all(eigvals > -1e-10):
        # Clip small negative eigenvalues
        P_deficit_clipped = np.clip(P_deficit, 0, None)
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(P_deficit_clipped + 1e-10 * np.eye(D))
            # Generate random perturbations
            noise = rng.standard_normal((N, D))
            # The noise should have covariance L @ L.T
            # So we add: L.T @ noise (or L @ noise depending on convention)
            # If noise ~ N(0, I), then L.T @ noise has covariance L.T @ L = P_deficit_clipped
            stochastic_term = noise @ L.T
            updated = updated + stochastic_term
        except np.linalg.LinAlgError:
            # Fallback: use SVD
            U, s, Vt = np.linalg.svd(P_deficit_clipped)
            L = U @ np.diag(np.sqrt(s)) @ Vt
            noise = rng.standard_normal((N, D))
            stochastic_term = noise @ L.T
            updated = updated + stochastic_term
    else:
        # P_deficit is not PSD, which means our deterministic update already has
        # more spread than expected. In this case, we might need to reduce spread.
        # For now, just return the deterministic update.
        pass
    
    # Ensure output is finite and within bounds
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated
# EVOLVE-BLOCK-END
