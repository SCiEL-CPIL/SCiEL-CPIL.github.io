import numpy as np

def _rk4_step(state, f, dt):
    s = np.asarray(state, dtype=float)
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _l96(x, F):
    x = np.asarray(x, dtype=float)
    return (np.roll(x, 1) - np.roll(x, 2)) * np.roll(x, -1) - x + F

def _localize(dist, sigma):
    # Gaspari-Cohn 5th-order compact support
    r = dist / sigma
    out = np.zeros_like(r)
    m1 = r < 0.5
    m2 = (r >= 0.5) & (r < 1.5)
    m3 = (r >= 1.5) & (r < 2.5)
    m4 = r >= 2.5
    out[m1] = (-4.0 / 3.0) * r[m1]**5 + r[m1]**4 + (5.0 / 4.0) * r[m1]**2 - 5.0 / 12.0
    out[m2] = (8.0 / 3.0) * r[m2]**5 - 5.0 * r[m2]**4 + (70.0 / 11.0) * r[m2]**3 - 35.0 * r[m2]**2 + 10.0 * r[m2] + 4.0 / 3.0
    out[m3] = -(2.0 / 3.0) * r[m3]**5 + 5.0 * r[m3]**4 - 16.0 * r[m3]**3 + (40.0 / 3.0) * r[m3]**2 - 2.0
    out[m4] = 0.0
    return np.clip(out, 0.0, 1.0)

def _local_cov(cov, obs_idx, n):
    nobs = len(obs_idx)
    L = np.zeros((n, nobs))
    dist = np.abs(np.subtract.outer(np.arange(n), obs_idx))
    dist = np.minimum(dist, n - dist)
    L = _localize(dist, 2.0)
    L[np.diag_indices(nobs)] = 0.0  # identity handled separately
    return L

def analyze(ensemble, observations, observed_indices, observation_variance):
    import time
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin vs 120s kill

    ens = np.asarray(ensemble, dtype=float)
    obs = np.asarray(observations, dtype=float)
    obs_idx = np.asarray(observed_indices, dtype=int)
    R = float(observation_variance)
    n_ens, n = ens.shape
    nobs = len(obs_idx)

    mean = ens.mean(axis=0)
    # anomalies
    A = ens - mean  # (n_ens, n)
    Aobs = A[:, obs_idx]

    # background covariance of observations
    Rm = n_ens - 1.0
    Pobs = (Aobs.T @ Aobs) / Rm  # (nobs, nobs)

    # localization on observation space
    dist = np.abs(np.subtract.outer(obs_idx, obs_idx))
    dist = np.minimum(dist, n - dist)
    loc = _localize(dist, 2.0)
    Pobs = Pobs * loc + np.diag(np.diag(Pobs))

    # regularize
    Pobs = 0.5 * (Pobs + Pobs.T) + 1e-6 * np.eye(nobs)

    # innovation
    innov = obs - mean[obs_idx]  # (nobs,)

    try:
        K = np.linalg.solve(Pobs + R * np.eye(nobs), Pobs)  # (nobs, nobs)
    except np.linalg.LinAlgError:
        K = Pobs / (np.trace(Pobs) / nobs + R)

    # Kalman gain to all states: need P_full,obs. Compute via anomalies.
    # P_full = (A^T @ A)/Rm ; P_full_obs = (A^T @ Aobs)/Rm
    Pfull_obs = (A.T @ Aobs) / Rm  # (n, nobs)
    Pfull_obs_loc = Pfull_obs * loc.T  # localize cross-cov
    Pfull_obs_loc += np.zeros((n, nobs))
    # add diagonal identity part of Pobs into cross-cov? cross-cov diag contribution:
    for j, oj in enumerate(obs_idx):
        Pfull_obs_loc[oj, j] += Pobs[j, j]

    # gain per state
    Gain = Pfull_obs_loc @ K  # (n, nobs)
    update = Gain @ innov    # (n,)

    # deterministic mean update
    new_mean = mean + update

    # spread preservation on observed coords: match innovation variance
    # observed background spread
    bg_spread = Pobs[np.diag_indices(nobs)]
    # desired observed spread after: combine Pobs - Pobs^2/(Pobs+R)
    post_obs = bg_spread - (Pobs[np.diag_indices(nobs)]**2) / (bg_spread + R)
    ratio = np.sqrt(np.maximum(post_obs, 1e-12) / np.maximum(bg_spread, 1e-12))

    # multiplicative inflation on full state
    ratio_full = np.ones(n)
    ratio_full[obs_idx] = ratio
    # gentle: blend toward 1
    ratio_full = 0.5 + 0.5 * ratio_full

    # inflate anomalies
    ens_new = np.zeros_like(ens)
    for m in range(n_ens):
        ens_new[m] = new_mean + A[m] * ratio_full

    # safety: keep finite and bounded
    if not np.all(np.isfinite(ens_new)):
        ens_new = np.nan_to_num(ens_new, nan=0.0, posinf=100.0, neginf=-100.0)
    ens_new = np.clip(ens_new, -100.0, 100.0)

    # deadline guard: if we are still finishing within budget we are fine; else return best
    if time.time() > deadline:
        return np.clip(np.nan_to_num(ens_new), -100.0, 100.0)

    return ens_new
