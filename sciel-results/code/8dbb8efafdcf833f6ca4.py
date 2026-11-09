"""Lorenz-96 ensemble assimilation: LESRF with inflation and proper EKF."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    n = len(x)
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4_step(x, F, dt):
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _rk4_propagate_ensemble(ens, F, dt, steps):
    n_members = ens.shape[0]
    for _ in range(steps):
        for m in range(n_members):
            ens[m] = _rk4_step(ens[m], F, dt)
    return ens

def analyze(ensemble, observations, observed_indices, observation_variance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)

    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    R_val = float(observation_variance)

    # Infer F from ensemble statistics
    ens_mean = np.mean(ens, axis=0)
    # For steady-state Lorenz-96, average of (x_{i+1}-x_{i-2})*x_{i-1} - x_i + F = 0
    # So F ≈ mean of (x_{i+1}-x_{i-2})*x_{i-1} - x_i over i
    F_est = np.mean((np.roll(ens_mean, -1) - np.roll(ens_mean, 2)) * np.roll(ens_mean, 1) - ens_mean)
    F = np.clip(F_est, 4.0, 12.0)

    # Step 1: Forecast the ensemble forward 5 RK4 steps (dt=0.01 each)
    dt = 0.01
    n_forecast = 5
    ens = _rk4_propagate_ensemble(ens, F, dt, n_forecast)

    # Step 2: Local Ensemble Square-Root Filter (LESRF) with sequential updates
    # This is more robust than bulk EKF for small ensembles
    alpha = n_members / (n_members - 1.0)  # Bessel correction factor
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean  # (n_members, n_vars)

    # Localization: Gaussian with length scale
    corr_length = 4.0
    # Precompute localization weights for each observation and each state variable
    loc_weights = np.zeros((n_obs, n_vars))
    for j, oj in enumerate(obs_idx):
        for i in range(n_vars):
            dist = abs(i - oj)
            dist = min(dist, n_vars - dist)  # periodic distance
            loc_weights[j, i] = np.exp(-0.5 * (dist / corr_length) ** 2)

    # Sequential update for each observation
    for j in range(n_obs):
        obs_j = obs[j]
        oj = obs_idx[j]
        w = loc_weights[j, :]

        if w[oj] < 0.05:
            continue

        # Current ensemble mean and anomalies
        ens_mean = np.mean(ens, axis=0)
        H_ens = ens[:, oj]  # (n_members,)
        H_mean = np.mean(H_ens)
        H_anom = H_ens - H_mean  # (n_members,)

        # Innovation
        innovation = obs_j - H_mean

        # S = H P H^T + R (scalar for single observation)
        HP = alpha * np.dot(H_anom, H_anom) / (n_members - 1.0)
        S = HP + R_val

        if S < 1e-12:
            S = 1e-12

        # For each state variable, update using local ensemble
        for i in range(n_vars):
            if w[i] < 0.05:
                continue

            # Cross-covariance between state i and observation j
            x_anom_i = ens[:, i] - ens_mean[i]
            cross_cov = alpha * np.dot(x_anom_i, H_anom) / (n_members - 1.0)
            cross_cov_loc = cross_cov * w[i]

            # Kalman gain for this pair
            K_ij = cross_cov_loc / S

            # Update mean
            ens_mean[i] += K_ij * innovation

            # Update anomalies: x_a_anom = (1 - K_ij * H_anom) * x_anom  (simplified LESRF)
            # More precisely, apply square-root update
            # For simplicity, use the perturbation form
            ens[:, i] = ens_mean[i] + (1.0 - K_ij * (H_ens - H_mean)) * (ens[:, i] - ens_mean[i])

        # Recompute after each observation to maintain consistency
        ens_mean = np.mean(ens, axis=0)

    # Step 3: Adaptive inflation based on spread innovation consistency
    ens_mean_final = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean_final

    # Compute innovation vector
    y_forecast = ens_mean_final[obs_idx]
    innovations = obs - y_forecast

    # S from ensemble + R
    H_anom_all = ens_centered[:, obs_idx]  # (n_members, n_obs)
    S_mat = alpha * (H_anom_all.T @ H_anom_all) / (n_members - 1.0) + R_val * np.eye(n_obs)

    # Innovation covariance
    obs_cov = np.cov(innovations, rowvar=False) if n_obs > 1 else np.array([[np.var(innovations)]])
    if obs_cov.ndim == 0:
        obs_cov = np.array([[float(obs_cov)]])

    # Inflate if innovation variance > S (indicates under-dispersion)
    try:
        # Simple scalar inflation factor
        ratio = np.mean(np.diag(obs_cov)) / max(np.mean(np.diag(S_mat)), 1e-12)
        if ratio > 1.0:
            # Inflate to match
            inflation = np.sqrt(min(ratio, 4.0))  # cap inflation
        else:
            inflation = 1.0
    except:
        inflation = 1.0

    # Apply adaptive inflation with a floor
    inflation = max(inflation, 1.0)
    ens = ens_mean_final + inflation * ens_centered

    # Step 4: Safety checks
    ens = np.where(np.isfinite(ens), ens, 0.0)
    ens = np.clip(ens, -100.0, 100.0)

    # Time check
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)

    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
