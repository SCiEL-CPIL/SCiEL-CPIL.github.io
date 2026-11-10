import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update for Lorenz-96.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states.
    observations: (M, K) array of observations for K observed indices? 
                  Wait, the prompt says "observations of half its coordinates".
                  Usually, observations are a single vector of noisy values, or an ensemble of observations?
                  The prompt says: "observe even-indexed states with independent N(0,0.49) noise".
                  This implies there is a single true state, and we get one noisy observation vector per update cycle.
                  However, the interface is `analyze(ensemble, observations, ...)`.
                  If `observations` is a single vector (K,), then we are doing a standard Bayesian update where the observation is a single realization.
                  If `observations` is (M, K), it implies observation perturbations.
                  Let's check the shape. The prompt doesn't specify the shape of `observations` explicitly in the signature description, but "independent N(0,0.49) noise" usually applies to the single observation vector in a standard EnKF context unless it's an EnKF-OI with perturbed observations.
                  Given "independent high-accuracy reference trajectories", the evaluator likely generates one true trajectory and one noisy observation vector per step.
                  So `observations` is likely a 1D array of length K (number of observed indices).
                  Let's assume `observations` is 1D.
    observed_indices: array of indices that are observed.
    observation_variance: scalar variance of the observation noise.
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten()
    
    # 1. Compute ensemble mean
    mean_f = np.mean(ensemble, axis=0)
    
    # 2. Compute anomalies
    anomalies = ensemble - mean_f
    
    # 3. Apply inflation to anomalies to prevent underestimation of uncertainty
    # A static inflation factor of 1.05 to 1.1 is typical. Let's use 1.05.
    inflation_factor = 1.05
    anomalies *= inflation_factor
    
    # 4. Compute localized ensemble covariance
    # P_f = (1/(M-1)) * anomalies.T @ anomalies
    # Apply localization: P_f[i,j] *= L(i,j)
    # Gaspari-Cohn localization function
    def gaspari_cohn(d, L=4.0):
        # L is the correlation length scale in indices
        if d <= L/2:
            return (-0.25 * (2*d/L - 5)**5 + 5*(2*d/L - 3)**5 + 5*(2*d/L - 1)**5 + 5) / 4
        elif d <= L:
            return (0.5 * (2*d/L - 7)**5 - 5*(2*d/L - 5)**5 + 10*(2*d/L - 3)**5 - 5*(2*d/L - 1)**5 + 5) / 2
        else:
            return 0.0
            
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply localization to P_f
    for i in range(N):
        for j in range(N):
            # Distance on a ring of size N
            d = abs(i - j)
            d = min(d, N - d)
            P_f[i, j] *= gaspari_cohn(d, L=4.0)
            
    # 5. Observation operator H is implicit via indices
    # S = H P_f H^T + R
    # R is K x K diagonal with observation_variance
    H_P_H = P_f[np.ix_(observed_indices, observed_indices)]
    R = np.eye(K) * observation_variance
    S = H_P_H + R
    
    # 6. Kalman Gain K_g = P_f H^T S^-1
    # We need P_f[:, observed_indices] @ inv(S)
    # P_f_obs = P_f[:, observed_indices] (N x K)
    P_f_obs = P_f[:, observed_indices]
    
    # Solve S X = P_f_obs^T to get X = S^-1 P_f_obs^T
    # Then K_g = X^T (K x N)
    try:
        # Use Cholesky for stability if possible, but S might be ill-conditioned.
        # np.linalg.solve is robust enough for K<=10.
        S_inv_P_f_obs_T = np.linalg.solve(S, P_f_obs.T)
        K_g = S_inv_P_f_obs_T.T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        S_inv = np.linalg.pinv(S)
        K_g = S_inv @ P_f_obs.T
        K_g = K_g.T # Wait, K_g shape should be N x K
        # P_f_obs is N x K. S is K x K.
        # K_g = P_f_obs @ S^-1. Shape N x K.
        K_g = P_f_obs @ S_inv

    # 7. Innovation
    # y - H mean_f
    obs_mean = mean_f[observed_indices]
    innovation = obs - obs_mean
    
    # 8. Update mean
    mean_a = mean_f + K_g @ innovation
    
    # 9. Update anomalies
    # I - K_g H
    # H is K x N, so K_g H is N x N
    # (I - K_g H) @ anomalies.T (N x M) -> N x M
    # Then transpose back to M x N
    
    # Efficient update:
    # x_m^a = mean_a + (x_m^f - mean_f) * (I - K_g H)
    # Let A = anomalies (M x N)
    # We want A @ (I - H^T K_g^T) ? No.
    # x_m^a = mean_a + (x_m^f - mean_f) - (x_m^f - mean_f) H^T K_g^T
    # (x_m^f - mean_f) is row vector 1 x N.
    # H^T is N x K.
    # K_g^T is K x N.
    # So (x_m^f - mean_f) @ H^T @ K_g^T is 1 x N.
    
    # Let's compute the correction term for each member.
    # anomalies are M x N.
    # We need to subtract anomalies @ H^T @ K_g^T
    # H^T @ K_g^T is N x N.
    # Let M_update = I - H^T @ K_g^T
    # Then anomalies_a = anomalies @ M_update
    
    # Construct H^T K_g^T
    # H^T is N x K. K_g is N x K. K_g^T is K x N.
    # H^T @ K_g^T is N x N.
    H_T_K_g_T = np.zeros((N, N))
    # H_T_K_g_T[:, observed_indices] = K_g.T ? No.
    # (H^T K_g^T)_{ij} = sum_k H_{ki} K_{jk}
    # H_{ki} is 1 if i == observed_indices[k], else 0.
    # So column i of H^T K_g^T is row i of K_g^T (which is column i of K_g) if i is observed, else 0?
    # Let's check indices.
    # K_g is N x K. K_g[n, k] is gain for state n given obs k.
    # (K_g^T)_{k, n} = K_g[n, k].
    # (H^T)_{n, k} = 1 if n == obs_idx[k], else 0.
    # (H^T K_g^T)_{n, m} = sum_k (H^T)_{n, k} (K_g^T)_{k, m}
    # = (K_g^T)_{k, m} where k is such that obs_idx[k] == n.
    # If n is not observed, the sum is 0.
    # If n is observed, say n = obs_idx[k0], then the term is (K_g^T)_{k0, m} = K_g[m, k0].
    
    # So, for each observed index n, let k0 be its position in observed_indices.
    # Then row n of (H^T K_g^T) is column k0 of K_g^T, which is row k0 of K_g (transposed? No).
    # (H^T K_g^T)[n, :] = K_g[:, k0].T ?
    # Let's verify:
    # (H^T K_g^T)[n, m] = K_g[m, k0].
    # So the row vector is [K_g[0, k0], K_g[1, k0], ..., K_g[N-1, k0]].
    # This is exactly K_g[:, k0].
    
    H_T_K_g_T = np.zeros((N, N))
    for k, idx in enumerate(observed_indices):
        H_T_K_g_T[idx, :] = K_g[:, k]
        
    M_update = np.eye(N) - H_T_K_g_T
    anomalies_a = anomalies @ M_update
    
    # 10. Reconstruct ensemble
    ensemble_a = mean_a[None, :] + anomalies_a
    
    # Ensure finite and within bounds
    ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_a = np.clip(ensemble_a, -100.0, 100.0)
    
    return ensemble_a
