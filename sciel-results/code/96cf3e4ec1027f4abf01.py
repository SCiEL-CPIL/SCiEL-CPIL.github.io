"""Local Ensemble Transform Kalman Filter (LETKF) for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LETKF) analysis update for Lorenz-96.
    
    Uses block-localization with radius 4 to maintain ensemble spread and
    positive semi-definiteness of the local covariance matrix.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)  # 10
    obs_var = float(observation_variance)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    # Block localization radius
    radius = 4
    
    # Process each observed index with local LETKF update
    updated_ensemble = ensemble.copy()
    
    for k, obs_idx in enumerate(observed_indices):
        obs_idx = int(obs_idx)
        obs_val = observations[k]
        
        # Determine local block: indices within radius of obs_idx (circular)
        local_indices = []
        for d in range(-radius, radius + 1):
            idx = (obs_idx + d) % nvars
            local_indices.append(idx)
        
        nlocal = len(local_indices)
        
        # Extract local ensemble (nm x nlocal)
        local_ens = updated_ensemble[:, local_indices]
        
        # Compute local mean
        local_mean = np.mean(local_ens, axis=0)
        
        # Center the local ensemble
        local_centered = local_ens - local_mean
        
        # Compute local covariance: C = (1/(nm-1)) * X_c^T * X_c
        # Shape: (nlocal, nlocal)
        local_cov = (local_centered.T @ local_centered) / (nm - 1)
        
        # Find the position of obs_idx within local_indices
        obs_pos = local_indices.index(obs_idx)
        
        # Observation operator for local block: only observes obs_pos
        # H is (1, nlocal)
        H = np.zeros((1, nlocal))
        H[0, obs_pos] = 1.0
        
        # Innovation: y - H * x_mean
        innovation = obs_val - np.dot(H, local_mean)
        
        # Innovation covariance: S = H * C * H^T + R
        # S is scalar
        S = local_cov[obs_pos, obs_pos] + obs_var
        S = max(S, 1e-10)
        
        # Kalman gain: K = C * H^T / S
        # K shape: (nlocal, 1)
        K = local_cov[:, obs_pos] / S
        
        # Compute the LETKF transform
        # We need to find a transform A such that:
        # (x_a - x_a_mean) = A * (x_f - x_f_mean)
        # where x_a = x_f + K * (y - H * x_f)
        # 
        # The LETKF approach:
        # Let Y = local_centered (nm x nlocal)
        # Let K be (nlocal, 1)
        # We want A such that Y_a = Y + K * (y - H*Y)^T
        # = Y + K * innovation - K * H * Y
        # = (I - K*H) * Y + K * innovation
        #
        # So A = I - K*H (nlocal x nlocal)
        # And the update is: Y_a = A * Y + K * innovation
        #
        # But we need A to be such that the ensemble spread is preserved.
        # The LETKF constructs A as a product of orthogonal and diagonal matrices.
        
        # Compute B = I - K*H
        B = np.eye(nlocal)
        B -= np.outer(K.flatten(), H.flatten())
        
        # Now we need to find A such that A @ A^T = B @ B^T (approximately)
        # Actually, the standard LETKF formulation:
        # Y_a = Y + K * (y - H*Y)^T
        # = Y + K * innovation - K * H * Y
        # = (I - K*H) * Y + K * innovation
        #
        # Let's just use the additive form but with proper normalization
        # to preserve the ensemble spread.
        
        # Compute the analysis ensemble directly
        # Y_a = Y + K * (y - H*Y)^T
        # The innovation for each member: y - H * Y_i
        # H * Y_i = Y_i[obs_pos]
        
        member_innovations = obs_val - local_centered[:, obs_pos] - local_mean[obs_pos]
        # Wait, let me redo this.
        
        # For each member i:
        # x_f_i = local_ens[i, :]
        # x_a_i = x_f_i + K * (y - H * x_f_i)
        # H * x_f_i = x_f_i[obs_pos]
        # y - H * x_f_i = obs_val - x_f_i[obs_pos]
        
        innovations_vec = obs_val - local_ens[:, obs_pos]  # (nm,)
        
        # Analysis ensemble
        local_analysis = local_ens + np.outer(innovations_vec, K.flatten())
        
        # Now, to preserve ensemble spread, we apply the LETKF transform.
        # The key idea: compute the transform A such that
        # (local_analysis - mean(local_analysis)) = A @ (local_ens - local_mean)
        
        analysis_mean = np.mean(local_analysis, axis=0)
        analysis_centered = local_analysis - analysis_mean
        
        # We want A such that analysis_centered = A @ local_centered
        # Using SVD: local_centered = U * S * V^T
        # analysis_centered = W * T * Z^T
        # A = W * Z^T (if both are nm x nlocal and we want the "closest" transform)
        
        # Actually, the standard LETKF approach is different. Let me use the 
        # proper formulation from Hunt et al. (2007).
        
        # The LETKF computes:
        # 1. Y = local_centered (nm x nlocal)
        # 2. Compute B = I_nlocal - K @ H  (nlocal x nlocal)
        # 3. Find A such that A @ A^T = B @ B^T
        #    Using SVD: B = U_B * S_B * V_B^T
        #    A = U_B * V_B^T (orthogonal) * sqrt(diag(S_B)) ... 
        #    Actually, the standard approach:
        #    A = Q * D where Q is orthogonal and D is diagonal
        #    such that A @ A^T = B @ B^T
        
        # Let me use a simpler but valid approach:
        # Compute the Cholesky-like decomposition to preserve spread.
        
        # Method: 
        # 1. Compute C_f = local_cov (prior covariance)
        # 2. Compute C_a = (I - K*H) @ C_f @ (I - K*H)^T + K * R * K^T
        #    But R = obs_var, and K*R*K^T is the correction term
        # 3. Actually, C_a = C_f - K @ H @ C_f (for the deterministic part)
        #    The full analysis covariance is:
        #    C_a = (I - K*H) C_f (I - K*H)^T + K R K^T
        
        # Let's compute C_a properly
        KH = np.outer(K.flatten(), H.flatten())  # (nlocal, nlocal)
        IKH = np.eye(nlocal) - KH
        C_a = IKH @ local_cov @ IKH.T + np.outer(K.flatten(), K.flatten()) * obs_var
        
        # Now we need to transform the ensemble to have covariance C_a
        # while maintaining the same mean.
        
        # Use the Cholesky decomposition approach:
        # If C_f = L_f @ L_f^T and C_a = L_a @ L_a^T
        # Then A = L_a @ L_f^{-T}
        
        # Add small regularization for numerical stability
        reg = 1e-6 * np.eye(nlocal)
        C_f_reg = local_cov + reg
        C_a_reg = C_a + reg
        
        try:
            L_f = np.linalg.cholesky(C_f_reg)
            L_a = np.linalg.cholesky(C_a_reg)
            # A = L_a @ inv(L_f^T)
            A = np.linalg.solve(L_f.T, L_a).T
        except np.linalg.LinAlgError:
            # Fallback: use the additive update directly
            updated_ensemble[:, local_indices] = local_analysis
            continue
        
        # Apply the transform
        # local_centered has shape (nm, nlocal)
        # A has shape (nlocal, nlocal)
        # We want: analysis_centered = local_centered @ A^T
        transformed = local_centered @ A.T
        
        # Add back the analysis mean
        updated_ensemble[:, local_indices] = transformed + analysis_mean
    
    # Final safety checks
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline check
    if time.time() - t0 > 115.0:
        return updated_ensemble
    
    return updated_ensemble
# EVOLVE-BLOCK-END
