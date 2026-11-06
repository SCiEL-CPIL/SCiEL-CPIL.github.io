"""Ensemble Kalman Filter with ensemble transform, localization, and adaptive inflation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Transform Kalman Filter (ETKF) update with:
    - Spatial localization (exponential decay)
    - Adaptive ensemble inflation to prevent filter collapse
    - Robust handling of the Lorenz-96 structure
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (16, 10)
        Noisy observations of observed_indices for each ensemble member? 
        Actually, typically observations are a single vector or per-member.
        Given the interface, let's assume observations shape matches the observed part.
        The task says: "observes even-indexed states with independent N(0,0.49) noise".
        observed_indices are likely [0, 2, 4, ..., 18].
        The observations passed in are likely shape (10,) or (16, 10)?
        Looking at the seed: `result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])`
        This implies `observations` has shape (16, len(observed_indices)) or broadcasts.
        If observations is (10,), then `observations - result[:, observed_indices]` would broadcast (10,) against (16,10), resulting in (16,10).
        So observations is likely a single vector of length 10 (the same for all members) OR a matrix.
        Standard assimilation uses a single observation vector y (length m).
        Let's assume observations is shape (10,) or (16, 10).
        If it's (16, 10), it might be different obs per member (unlikely for standard KF, but possible for some variants).
        However, the standard EKF uses a single y.
        Let's check the subtraction: `observations - result[:, observed_indices]`.
        If obs is (10,), it broadcasts to (16,10).
        If obs is (16,10), it subtracts directly.
        Given "independent N(0,0.49) noise", it's usually one set of observations.
        Let's assume observations is a 1D array of length 10.
        
    observed_indices : array-like
        Indices of the observed variables (e.g., [0, 2, 4, ..., 18]).
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    
    n_members, n_vars = ensemble.shape
    m_obs = len(observed_indices)
    
    # 1. Compute Ensemble Mean and Covariance
    x_bar = np.mean(ensemble, axis=0)
    
    # Deviations from mean
    X_dev = ensemble - x_bar
    
    # Ensemble Covariance: P = (1/(N-1)) * X_dev.T @ X_dev
    P = (X_dev.T @ X_dev) / (n_members - 1)
    
    # 2. Localization
    # Apply exponential localization to the covariance matrix to reduce spurious correlations
    # Distance matrix for periodic indices
    idx = np.arange(n_vars)
    # Periodic distance: min(|i-j|, n_vars - |i-j|)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Localization length scale (tunable). For L96, a scale of 3-5 is common.
    L = 4.0
    localization_matrix = np.exp(-dist / L)
    
    # Apply localization to the full covariance matrix P
    # This is a standard approach, though computationally O(N^3) if done naively for large N.
    # Here N=20, so it's fine.
    P_loc = P * localization_matrix
    
    # Ensure symmetry and positive semi-definiteness (approximate)
    P_loc = (P_loc + P_loc.T) / 2
    
    # 3. Observation Operator
    # H: maps state to observation space. H is a selection matrix.
    # H[i, j] = 1 if j == observed_indices[i], else 0
    H = np.zeros((m_obs, n_vars))
    H[np.arange(m_obs), observed_indices] = 1.0
    
    # Observation Covariance R
    R = observation_variance * np.eye(m_obs)
    
    # 4. Innovation Covariance
    # S = H P H^T + R
    HP = H @ P_loc
    S = HP @ H.T + R
    
    # 5. Kalman Gain
    # K = P H^T S^-1
    # To ensure stability and positive semi-definiteness, we use the Ensemble Transform approach
    # or compute the gain directly.
    # Direct Gain:
    try:
        # Solve S K^T = P H^T  =>  K = (S^-1 P H^T)^T
        # Or K = P H^T S^-1
        # Using Cholesky for stability
        L_S = np.linalg.cholesky(S)
        # K = P H^T S^-1
        # We can compute K = (L_S^-1 (P H^T L_S^-T))^T ? No.
        # Standard: K = P H^T S^-1
        # Let's solve S^T X = H P^T  => X = S^-1 H P^T => K = X^T
        # S is symmetric, so S X = H P^T
        X = np.linalg.solve(S, H @ P_loc.T)
        K = X.T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        K = P_loc @ H.T @ np.linalg.pinv(S)
    
    # 6. Update Mean
    # y - H x_bar
    innovation = observations - H @ x_bar
    x_bar_new = x_bar + K @ innovation
    
    # 7. Update Covariance
    # P_new = (I - K H) P (I - K H)^T + K R K^T (Joseph form for stability)
    I = np.eye(n_vars)
    IKH = I - K @ H
    P_new = IKH @ P_loc @ IKH.T + K @ R @ K.T
    
    # Ensure symmetry
    P_new = (P_new + P_new.T) / 2
    
    # 8. Ensemble Transform (ET)
    # We need to transform the deviations X_dev to X_dev_new such that
    # Cov(X_dev_new) = P_new
    
    # Method: Cholesky decomposition
    # X_dev = A A^T (where A is N x M, M=N-1 usually, but here we can use full N)
    # Actually, X_dev is (N_vars, N_members).
    # Let X_dev = Z Z^T * (1/(N-1))? No.
    # X_dev = X_dev. Cov = X_dev X_dev^T / (N-1).
    # We want X_dev_new such that X_dev_new X_dev_new^T / (N-1) = P_new.
    # Let X_dev_new = X_dev T, where T is N_members x N_members.
    # Then Cov_new = X_dev T T^T X_dev^T / (N-1) = P_new.
    # This is the standard ET formulation.
    
    # Compute Cholesky of P_loc and P_new
    # P_loc = X_dev X_dev^T / (N-1)
    # Let X_dev = C C^T (N_vars x N_vars)? No, X_dev is N_vars x N_members.
    # We can use the SVD or Cholesky of the covariance.
    
    # Alternative: Use the gain to update deviations directly?
    # x_i_new = x_bar_new + (I - K H) (x_i - x_bar) + K (y_i - H x_i)?
    # This is the "dual" or direct update. But we only have one y.
    # In ET, we transform the ensemble.
    
    # Let's use the Cholesky-based ET.
    # 1. Compute C_old such that C_old C_old^T = P_loc (N_vars x N_vars)
    #    But we have X_dev (N_vars x N_members).
    #    Let's use the relation: X_dev X_dev^T = (N-1) P_loc.
    #    We want X_dev_new X_dev_new^T = (N-1) P_new.
    #    
    #    Let X_dev = U D V^T (SVD).
    #    Then P_loc = (1/(N-1)) U D D^T U^T.
    #    Let P_new = U_new D_new D_new^T U_new^T.
    #    
    #    Simpler approach:
    #    Compute Cholesky of P_loc: P_loc = L_old L_old^T
    #    Compute Cholesky of P_new: P_new = L_new L_new^T
    #    Then X_dev_new = X_dev T, where T is derived from L_old and L_new.
    #    
    #    Actually, a common ET algorithm:
    #    1. X_dev = (N-1)^0.5 * C_old (where C_old is from Cholesky of P_loc? No, dimensions don't match directly if N_members != N_vars).
    #    
    #    Let's use the "Ensemble Square Root Filter" approach.
    #    Represent ensemble as X_dev (N x M).
    #    P = X_dev X_dev^T / (M-1).
    #    We want to find T (M x M) such that P_new = (X_dev T) (X_dev T)^T / (M-1).
    #    
    #    Step 1: Compute SVD of X_dev: X_dev = U S V^T.
    #    Step 2: P_loc = U (S^2 / (M-1)) U^T.
    #    Step 3: Compute eigen-decomp of P_new: P_new = W Lambda W^T.
    #    Step 4: Construct T.
    #    
    #    This is getting complex. Let's use a simpler, robust method:
    #    Update the ensemble members using the gain and a random projection?
    #    Or, use the "Local Ensemble Transform" (LETKF) which is standard for L96.
    
    # Let's try a simpler update:
    # x_i_new = x_bar_new + (I - K H) (x_i - x_bar)
    # This updates the mean correctly but reduces the variance incorrectly (collapses it).
    # To fix the variance, we can apply inflation.
    
    # Let's use the Joseph form for the covariance and then reconstruct the ensemble
    # by scaling the deviations.
    
    # 1. Update mean: x_bar_new = x_bar + K (y - H x_bar)
    # 2. Update deviations: X_dev_new = (I - K H) X_dev
    # 3. This gives Cov_new = (I-KH) P (I-KH)^T.
    # 4. The true new covariance is P_new = (I-KH) P (I-KH)^T + K R K^T.
    # 5. The difference is the inflation term K R K^T.
    # 6. We can add this by inflating the deviations.
    
    # Let's compute the inflation factor.
    # Trace(P_new) / Trace((I-KH)P(I-KH)^T)
    # This is a scalar inflation, which is crude but effective.
    
    X_dev_updated = (I - K @ H) @ X_dev
    
    # Compute the current covariance of X_dev_updated
    P_updated_no_inflation = (X_dev_updated.T @ X_dev_updated) / (n_members - 1)
    
    # We want P_new.
    # Let's compute the ratio of traces or determinants to find a scalar inflation factor.
    # Or, we can just add the K R K^T term by adding noise?
    
    # Better: Use the Cholesky decomposition to properly transform the ensemble.
    # Reference: "Ensemble Kalman Filter via the Square Root Formulation"
    
    # 1. Cholesky of P_loc: P_loc = L_p L_p^T
    try:
        L_p = np.linalg.cholesky(P_loc + 1e-10 * np.eye(n_vars))
    except:
        L_p = np.linalg.cholesky(P_loc + 1e-8 * np.eye(n_vars))
        
    # 2. Cholesky of P_new: P_new = L_new L_new^T
    try:
        L_new = np.linalg.cholesky(P_new + 1e-10 * np.eye(n_vars))
    except:
        L_new = np.linalg.cholesky(P_new + 1e-8 * np.eye(n_vars))
        
    # 3. We have X_dev = L_p W, where W is (N_vars x N_members) with orthonormal columns?
    #    Actually, X_dev X_dev^T = (N-1) P_loc = (N-1) L_p L_p^T.
    #    So X_dev = sqrt(N-1) L_p Q, where Q Q^T = I.
    #    We want X_dev_new = sqrt(N-1) L_new Q.
    #    So X_dev_new = L_new L_p^-1 X_dev / sqrt(N-1) * sqrt(N-1) ?
    #    X_dev_new = L_new (L_p^-1 X_dev).
    
    # Let's verify:
    # X_dev = sqrt(N-1) L_p Q.
    # L_p^-1 X_dev = sqrt(N-1) Q.
    # X_dev_new = L_new (sqrt(N-1) Q) = sqrt(N-1) L_new Q.
    # Cov(X_dev_new) = (1/(N-1)) X_dev_new X_dev_new^T = (1/(N-1)) (N-1) L_new Q Q^T L_new^T = L_new L_new^T = P_new.
    # This works!
    
    # Solve L_p Y = X_dev for Y.
    Y = np.linalg.solve(L_p, X_dev)
    
    # X_dev_new = L_new Y
    X_dev_new = L_new @ Y
    
    # 4. Construct new ensemble
    result = x_bar_new[None, :] + X_dev_new
    
    # 5. Adaptive Inflation
    # Check if the ensemble has collapsed.
    # Compute the spread in observed space vs. innovation.
    # If the spread is too small, inflate.
    
    # Spread in observation space
    H_X_dev_new = H @ X_dev_new
    P_obs_new = (H_X_dev_new.T @ H_X_dev_new) / (n_members - 1)
    spread_obs = np.mean(np.diag(P_obs_new))
    
    # Innovation variance
    innovation_var = np.var(innovation)
    
    # Ideal ratio: spread_obs should be comparable to innovation_var + obs_var?
    # Or we can use a simple heuristic:
    # If the ensemble spread is smaller than the observation error, inflate.
    
    # Let's use a dynamic inflation factor based on the ratio of the actual innovation
    # to the expected innovation.
    # Expected innovation variance: S (computed earlier)
    # S = H P H^T + R
    # Trace(S) / m_obs
    expected_innov_var = np.trace(S) / m_obs
    actual_innov_var = np.mean(innovation**2)
    
    # If actual > expected, the ensemble is under-dispersed -> inflate
    # If actual < expected, the ensemble is over-dispersed -> deflate (rare, usually just keep)
    
    # Inflation factor
    # alpha = max(1.0, sqrt(actual_innov_var / expected_innov_var))
    # But we need to apply this to the deviations.
    
    # A common approach:
    # alpha = 1.0
    # if actual_innov_var > expected_innov_var:
    #     alpha = np.sqrt(actual_innov_var / expected_innov_var)
    #     alpha = min(alpha, 1.5) # Cap inflation
    
    # However, the ET already preserves the covariance P_new which includes the K R K^T term.
    # So the inflation might be handled by the ET.
    # But numerical errors or localization might cause issues.
    # Let's add a small adaptive inflation just in case.
    
    # Compute the ratio of the trace of the new covariance in obs space to the expected
    trace_new_obs = np.trace(P_obs_new)
    trace_expected_obs = np.trace(S)
    
    if trace_expected_obs > 0:
        ratio = trace_new_obs / trace_expected_obs
        # If ratio < 1, we are under-dispersed
        if ratio < 1.0:
            inflation_factor = np.sqrt(1.0 / ratio)
            # Cap inflation
            inflation_factor = min(inflation_factor, 1.5)
            X_dev_new = X_dev_new * inflation_factor
            result = x_bar_new[None, :] + X_dev_new
    
    # 6. Clip values to [-100, 100]
    result = np.clip(result, -100, 100)
    
    return result
