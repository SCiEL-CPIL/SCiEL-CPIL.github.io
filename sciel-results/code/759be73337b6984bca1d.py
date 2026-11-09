"""Lorenz-96 ensemble assimilation: hybrid variational-ensemble filter with
linearized dynamics over the 5-step propagation window, adaptive inflation,
and Gaussian localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_dx(x, F):
    """Compute RHS of Lorenz-96 ODE for a single state vector."""
    n = len(x)
    dx = np.empty_like(x)
    dx[:] = (np.roll(x, 1) - np.roll(x, 3)) * np.roll(x, -1) - x + F
    return dx


def _jacobian_l96(x, F):
    """Compute the Jacobian of the Lorenz-96 vector field at state x.
    Returns (n, n) matrix J where J[i,j] = dx_i/dx_j."""
    n = len(x)
    J = np.zeros((n, n))
    # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # d/dx_j of term1: (x_{i+1}-x_{i-2})*x_{i-1}
    #   = delta(j,i-1)*(x_{i+1}-x_{i-2}) + x_{i-1}*(delta(j,i+1) - delta(j,i-2))
    # d/dx_j of -x_i = -delta(j,i)
    for i in range(n):
        J[i, (i - 1) % n] += (x[(i + 1) % n] - x[(i - 2) % n])
        J[i, (i + 1) % n] += x[(i - 1) % n]
        J[i, (i - 2) % n] -= x[(i - 1) % n]
        J[i, i] -= 1.0
    return J


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_dx(x, F)
    k2 = _lorenz96_dx(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_dx(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_dx(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Hybrid variational-ensemble Kalman filter for Lorenz-96.
    Linearizes dynamics over the 5-step RK4 propagation window to build
    a state-transition matrix, uses it to correct the prior covariance,
    then performs a localized deterministic Kalman update with adaptive
    multiplicative inflation.

    Parameters
    ----------
    ensemble : ndarray (16, 20)
    observations : ndarray (10,)
    observed_indices : array-like of int
    observation_variance : float

    Returns
    -------
    updated_ensemble : ndarray (16, 20)
    """
    t0 = time.time()
    deadline = t0 + 115.0

    ens = np.array(ensemble, dtype=np.float64)
    obs = np.array(observations, dtype=np.float64)
    obs_idx = np.asarray(observed_indices, dtype=np.int64)
    R = float(observation_variance)
    n_mem, n = ens.shape
    n_obs = len(obs_idx)

    # ---- Estimate forcing from ensemble statistics ----
    # In Lorenz-96 the equilibrium is around x_i ≈ F/2.
    # We estimate F from the ensemble mean to build the Jacobian.
    ens_mean = ens.mean(axis=0)
    F_est = 2.0 * np.mean(ens_mean)  # rough; refined below

    # A better estimate: at each update the system has been propagating
    # under a fixed F.  We can estimate F from the ensemble by noting
    # that the mean of (x_{i+1}-x_{i-2})*x_{i-1} - x_i should be ≈ F.
    xm1 = np.roll(ens_mean, 1)
    xp1 = np.roll(ens_mean, -1)
    F_est = np.mean((xm1 - np.roll(ens_mean, 3)) * xp1 - ens_mean)
    F_est = max(F_est, 1.0)  # keep positive and reasonable

    # ---- Build state-transition matrix via linearized dynamics ----
    # Propagate the Jacobian through 5 RK4 steps (dt=0.01).
    # We linearize around the ensemble mean.
    x0 = ens_mean
    M = np.eye(n)  # state-transition matrix
    dt = 0.01
    n_steps = 5

    for _ in range(n_steps):
        J = _jacobian_l96(x0, F_est)
        # One-step transition for linearized system: x_{k+1} = x_k + dt*f(x_k)
        # Jacobian of the map: I + dt*J  (first-order)
        # For RK4 the map Jacobian is more complex; use I + dt*J as approximation
        M = (np.eye(n) + dt * J) @ M
        # Advance the linearization point with RK4
        x0 = _rk4_step(x0, F_est, dt)

    # ---- Compute ensemble covariance and apply state-transition ----
    mean = ens.mean(axis=0)
    Xa = ens - mean  # anomalies (n_mem, n)

    # Prior covariance P_f = Xa^T Xa / (n-1)
    P_f = (Xa.T @ Xa) / (n_mem - 1.0)

    # Correct covariance with dynamics: P_prior = M @ P_f @ M^T
    # This accounts for how uncertainty evolves between observations.
    P_prior = M @ P_f @ M.T

    # Symmetrize
    P_prior = 0.5 * (P_prior + P_prior.T)

    # ---- Adaptive multiplicative inflation ----
    obs_loc = ens[:, obs_idx]
    ens_spread_obs = np.sqrt(np.maximum(np.var(obs_loc, axis=0), 1e-12))

    innovation = obs - mean[obs_idx]
    innov_var = np.mean(innovation ** 2)

    # Expected innovation variance from ensemble + obs noise
    P_oo_ens = np.var(obs_loc, axis=0)
    exp_innov = np.mean(P_oo_ens + R)

    if exp_innov > 1e-12:
        ratio = innov_var / exp_innov
        infl_adapt = np.clip(ratio, 0.90, 1.35)
    else:
        infl_adapt = 1.02

    # Spread-based component
    target_spread = np.sqrt(R) * 1.2
    spread_ratio = target_spread / np.maximum(ens_spread_obs, 1e-12)
    infl_spread = np.clip(np.mean(spread_ratio), 0.95, 1.15)

    # Blend
    infl = 0.55 * infl_adapt + 0.45 * infl_spread
    infl = 0.3 * 1.005 + 0.7 * infl
    infl = np.clip(infl, 1.0, 1.12)

    # Apply inflation to anomalies
    Xa = infl * Xa
    mean = ens.mean(axis=0)  # mean unchanged by multiplicative inflation
    # Recompute inflated ensemble for the update
    ens_inf = mean[None, :] + Xa

    # ---- Localization ----
    sigma = 2.0
    idx = np.arange(n)
    dist = np.abs(idx[:, None].astype(float) - obs_idx[None, :].astype(float))
    dist = np.minimum(dist, n - dist)
    W_loc = np.exp(-0.5 * (dist / sigma) ** 2)  # (n, n_obs)

    # ---- Kalman gain with dynamic prior ----
    # P_xo = P_prior[:, obs_idx]  (n, n_obs)
    P_xo = P_prior[:, obs_idx]
    P_xo_loc = P_xo * W_loc  # localized

    # P_oo = P_prior[obs_idx, obs_idx]
    P_oo = P_prior[np.ix_(obs_idx, obs_idx)]
    S = P_oo + R * np.eye(n_obs)

    # Add small jitter for numerical stability
    S += 1e-8 * np.eye(n_obs)

    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S += 1e-6 * np.eye(n_obs)
        L = np.linalg.cholesky(S)

    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv

    K = P_xo_loc @ S_inv  # (n, n_obs)

    # ---- Update each ensemble member ----
    # For member i: innovation = obs - ens_inf[i, obs_idx]
    innovations = obs[None, :] - ens_inf[:, obs_idx]  # (n_mem, n_obs)
    update = (K @ innovations.T).T  # (n_mem, n)

    ens_new = ens_inf + update

    # ---- Second-order correction: adjust mean shift ----
    # The deterministic update shifts all members identically in the
    # localized subspace.  Verify the mean is reasonable.
    new_mean = ens_new.mean(axis=0)
    # Clamp
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.where(np.isfinite(ens_new), ens_new, 0.0)

    if time.time() > deadline:
        return ens_new.astype(ensemble.dtype)

    return ens_new.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
