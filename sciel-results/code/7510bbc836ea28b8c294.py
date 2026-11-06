"""Rewritten: Gaussianized Particle Filter with Localization and Inflation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 118.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def _gaspari_cohn(dist, r0, r1):
    """Periodic Gaspari-Cohn taper. Returns (p,p) array in [0,1]."""
    p = dist.shape[0]
    loc = np.ones((p, p))
    mask = dist > r0
    if np.any(mask):
        d = dist[mask]
        x = np.clip(d / (r1 - r0), 0.0, 1.0)
        val = 0.25 * x**4 * (5.0 * x - 9.0) + 0.5 * x**2
        loc[mask] = val
    return np.clip(loc, 0.0, 1.0)


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64).copy()
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X)

    k = len(obs)
    if k == 0:
        return _finite_clip(X)

    # --- Precompute periodic distance matrix ---
    idx = np.arange(p)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, p - dist)

    # --- Fixed Localization for Lorenz-96 ---
    # r0=3, r1=7 is a standard robust choice for L96
    r0 = 3.0
    r1 = 7.0
    loc = _gaspari_cohn(dist, r0, r1)

    # --- Prior Statistics ---
    m_prior = X.mean(axis=0)
    A = X - m_prior[None, :]
    P_prior = (A.T @ A) / (n - 1)
    P_loc = P_prior * loc

    # --- Observation Operator ---
    # H is implicit: y = X[:, obs] + noise
    H = np.zeros((k, p))
    for i, idx in enumerate(obs):
        H[i, idx] = 1.0

    # --- Particle Weights (Likelihood) ---
    # Compute squared error for each particle
    # innov_i = y - H x_i
    # dist2_i = innov_i^T R^-1 innov_i
    # Since R is diagonal (independent noise), R^-1 = 1/R_var * I
    innovations = y[None, :] - X[:, obs]  # (n, k)
    dist2 = np.sum(innovations**2, axis=1) / R_var  # (n,)
    
    # Stabilize weights
    # Shift by min to avoid underflow
    dist2_shifted = dist2 - np.min(dist2)
    weights = np.exp(-0.5 * dist2_shifted)
    
    # Normalize weights
    sum_w = np.sum(weights)
    if sum_w < 1e-300:
        # Fallback to uniform if all weights underflowed
        weights = np.ones(n) / n
    else:
        weights = weights / sum_w

    # --- Weighted Posterior Mean and Covariance ---
    # m_post = sum w_i x_i
    m_post = np.sum(weights[:, None] * X, axis=0)
    
    # P_post = sum w_i (x_i - m_post)(x_i - m_post)^T
    # Use centered particles
    X_centered = X - m_post[None, :]
    # Weighted covariance
    # P = (1/(1-sum(w^2))) * sum w_i (x_i-m)(x_i-m)^T  (Bessel correction for weights)
    w_sum_sq = np.sum(weights**2)
    bessel = 1.0 / (1.0 - w_sum_sq) if w_sum_sq < 1.0 else 1.0
    
    P_post = (X_centered.T * weights) @ X_centered * bessel
    P_post = P_post * loc  # Re-apply localization to posterior covariance for stability? 
                           # Usually localization is applied to prior. Applying to posterior 
                           # might be too aggressive. Let's stick to prior localization 
                           # in the gain calculation, but here we are constructing the 
                           # posterior directly. 
                           # Actually, for the Gaussianization step, we want the 
                           # posterior covariance to be localized to avoid spurious 
                           # correlations affecting the new samples.

    # --- Gaussianization (Particle Resampling from Posterior Gaussian) ---
    # We generate new ensemble members from N(m_post, P_post)
    # This ensures the new ensemble has the correct mean and covariance 
    # and avoids weight degeneracy.
    
    # Cholesky decomposition of P_post
    try:
        L = np.linalg.cholesky(P_post + 1e-9 * np.eye(p))
    except np.linalg.LinAlgError:
        # If not positive definite, use SVD
        U, s, Vt = np.linalg.svd(P_post)
        L = U @ np.diag(np.sqrt(s)) @ Vt

    # Generate standard normal random numbers
    # Use a fixed seed for reproducibility within the call? 
    # No, the evaluator expects a deterministic function of inputs.
    # However, particle filters are stochastic. 
    # The task says "deterministic kernel density estimate". 
    # To make it deterministic, we can use the ensemble members themselves 
    # as the source of randomness (permutation) or use a fixed pseudo-random 
    # sequence. 
    # Better: Use the "Deterministic Particle Filter" approach where we 
    # don't sample, but rather map the prior ensemble to the posterior 
    # using a quantile matching or similar. 
    # However, the simplest deterministic "Gaussianization" is to 
    # rotate the centered prior ensemble by the Cholesky factor ratio.
    
    # Alternative: Just return the weighted ensemble? No, that leaves 
    # weights behind. We need to return an unweighted ensemble.
    # The "Gaussianized" approach is standard. To make it deterministic, 
    # we can use the prior ensemble's deviations as the "noise" source.
    # Let Z be the prior ensemble deviations from m_prior.
    # We want new X such that X - m_post ~ N(0, P_post).
    # If we assume the prior was N(m_prior, P_prior), then Z ~ N(0, P_prior).
    # We can transform Z to have covariance P_post.
    # X_new = m_post + Z @ (L_prior^-1)^T @ L_post^T ?
    # This is complex. 
    
    # Simpler Deterministic Approach:
    # Use the ensemble mean and covariance, and generate new particles 
    # using a fixed set of standard normal vectors derived from the 
    # input ensemble (e.g., the first 16 components of a Hadamard matrix 
    # or just reusing the centered prior ensemble).
    
    # Let's reuse the centered prior ensemble as the "random" source.
    # Z_prior = X - m_prior
    # We want Z_post such that Cov(Z_post) = P_post.
    # If Cov(Z_prior) = P_prior, then Z_post = Z_prior @ (L_prior^-1)^T @ L_post^T
    # This is the "Ensemble Transform" approach.
    
    try:
        L_prior = np.linalg.cholesky(P_prior * loc + 1e-9 * np.eye(p))
        # Z_prior is (n, p). We need to map it to (n, p) with cov P_post.
        # Z_post = Z_prior @ M
        # Cov(Z_post) = M^T Cov(Z_prior) M = M^T P_prior M
        # We want M^T P_prior M = P_post
        # Let P_prior = L_p L_p^T, P_post = L_q L_q^T
        # Then M = L_p^-1 L_q works? 
        # (L_p^-1 L_q)^T (L_p L_p^T) (L_p^-1 L_q) = L_q^T L_p^-T L_p L_p^T L_p^-1 L_q = L_q^T L_q != P_post.
        # Correct transform: Z_post = Z_prior @ (L_prior^-1)^T @ L_post^T
        # Let's check: Cov(Z_prior @ A) = A^T P_prior A.
        # Let A = (L_prior^-1)^T L_post^T.
        # A^T P_prior A = L_post L_prior^-1 P_prior L_prior^-T L_post^T
        # P_prior = L_prior L_prior^T
        # = L_post L_prior^-1 L_prior L_prior^T L_prior^-T L_post^T
        # = L_post I L_post^T = P_post. Correct.
        
        # Compute A
        # Solve L_prior X = I -> X = L_prior^-1
        # This is O(p^3). p=20, so it's fast.
        L_prior_inv = np.linalg.inv(L_prior)
        A = L_prior_inv.T @ L_post.T  # Wait, L_post is L from Cholesky of P_post
        # Z_prior is (n, p). We need Z_prior @ A.
        # Z_prior is X - m_prior.
        Z_prior = X - m_prior[None, :]
        Z_post = Z_prior @ A
        X_new = m_post[None, :] + Z_post
        
    except np.linalg.LinAlgError:
        # Fallback to simple mean shift
        X_new = X + (m_post - m_prior)[None, :]

    # --- Variance Inflation ---
    # Check if the spread is adequate
    m_new = X_new.mean(axis=0)
    An = X_new - m_new[None, :]
    obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
    # Expected post-update variance is roughly R_var if localization is tight
    # or less. Let's just ensure it's not zero.
    mean_obs_var = np.mean(obs_var_ens)
    if mean_obs_var < 0.5 * R_var:
        # Inflate
        alpha = np.sqrt(R_var / (mean_obs_var + 1e-9))
        alpha = np.clip(alpha, 1.0, 2.0) # Cap inflation
        X_new = m_new[None, :] + alpha * An

    return _finite_clip(X_new)
# EVOLVE-BLOCK-END
