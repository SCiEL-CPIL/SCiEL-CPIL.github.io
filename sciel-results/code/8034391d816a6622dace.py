"""Lorenz-96 assimilation: localized EnSRF with multi-scattering (ES-MSC) and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 140.0
_t0 = None

def _check_deadline():
    """Return True if we are close to the wall-clock limit."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side with periodic indices (nondimensional units)."""
    n = x.shape[0]
    xm2 = np.roll(x, 2)      # x_(i-2)
    xm1 = np.roll(x, 1)      # x_(i-1)
    xp1 = np.roll(x, -1)     # x_(i+1)
    return (xp1 - xm2) * xm1 - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step of the Lorenz-96 system for one state vector."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _rk4_propagate(ens, n_steps, dt, F):
    """Propagate each ensemble member forward by n_steps RK4 steps of size dt."""
    out = ens.copy()
    for _ in range(n_steps):
        for m in range(out.shape[0]):
            out[m] = _rk4_step(out[m], dt, F)
    return out

def _localize_gain(K, observed_indices, n_state, decay_length):
    """Apply circular-Gaussian localization to the (n_state, n_obs) gain."""
    localized_K = np.zeros_like(K)
    for j in range(K.shape[1]):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    return localized_K

def _ensrf_update(ens, observations, observed_indices, observation_variance, decay_length):
    """One localized Ensemble Square Root Filter (EnSRF) analysis update."""
    n_members, n_state = ens.shape
    n_obs = len(observations)

    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    cov += 1e-6 * np.eye(n_state)

    R = np.eye(n_obs) * observation_variance

    ens_obs = ens[:, observed_indices]
    ens_obs_mean = np.mean(ens_obs, axis=0)
    innovation = observations - ens_obs_mean

    P_oo = cov[np.ix_(observed_indices, observed_indices)]
    S = P_oo + R
    S_reg = S + 1e-8 * np.eye(n_obs)

    PHt = cov[:, observed_indices]
    try:
        K = np.linalg.solve(S_reg, PHt.T).T
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(S_reg)

    localized_K = _localize_gain(K, observed_indices, n_state, decay_length)

    member_innovations = observations[None, :] - ens_obs
    update = member_innovations @ localized_K.T
    ens_new = ens + update

    # Adaptive spread-matching inflation
    ens_mean_new = np.mean(ens_new, axis=0)
    anomalies_new = ens_new - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    P_oo_new = cov_new[np.ix_(observed_indices, observed_indices)]
    S_expected = P_oo_new + R
    S_expected_diag = np.diag(S_expected)

    ratios = np.ones(n_obs)
    for j in range(n_obs):
        exp_var = S_expected_diag[j]
        act_var = innovation[j] ** 2 + observation_variance
        if exp_var > 1e-10 and act_var > 1e-10:
            ratios[j] = act_var / exp_var

    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.95, 1.25))

    ens_new = ens_mean_new + inflation * (ens_new - ens_mean_new)
    return ens_new

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized EnSRF analysis with a multi-scattering (ES-MSC) loop and adaptive inflation.

    The base update is a localized Ensemble Square Root Filter. To capture the
    non-Gaussian, non-linear error structure of the Lorenz-96 flow, we run a
    short scattering loop: after each linear update we re-propagate the ensemble
    through the known dynamics (RK4, dt=0.01, 5 steps) and re-apply the update,
    using a decaying gain so the sequence is damped and stable. The first
    scattering is the full linear update (matches the proven EnSRF baseline);
    subsequent scatterings use a reduced gain to correct residual non-linearity.
    """
    n_members, n_state = ensemble.shape
    ens = ensemble.astype(np.float64).copy()

    # F is not given to analyze; infer it from the ensemble statistics.
    # At a quasi-steady state of the Lorenz-96 system, E[x] ~ F (since
    # E[(x_{i+1}-x_{i-2}) x_{i-1}] - E[x_i] + F = 0 and the mean is ~F).
    ens_mean = np.mean(ens, axis=0)
    F = float(np.clip(np.mean(ens_mean), 0.0, 20.0))

    # Number of RK4 steps between updates (known from the task: 5 steps, dt=0.01).
    n_steps = 5
    dt = 0.01

    # Number of scattering iterations. Start with 2 (base update + one correction);
    # a third is attempted only if we have ample wall-clock budget.
    n_scatter = 2
    if not _check_deadline():
        # Budget check: propagating 16 members x 5 RK4 steps is cheap; a third
        # scattering is affordable if we are well within the deadline.
        n_scatter = 3

    decay_length = 2.0  # wider localization balances bias/variance (inspiration finding)

    # Gain schedule across scatterings: full update first, then damped corrections.
    gain_schedule = [1.0, 0.6, 0.35]

    for s in range(n_scatter):
        if _check_deadline():
            break

        g = gain_schedule[s] if s < len(gain_schedule) else 0.3

        if s == 0:
            ens = _ensrf_update(ens, observations, observed_indices, observation_variance, decay_length)
        else:
            # Damped correction: apply a fraction of the EnSRF update.
            n_mem, n_st = ens.shape
            ens_mean_s = np.mean(ens, axis=0)
            anomalies_s = ens - ens_mean_s
            cov_s = (anomalies_s.T @ anomalies_s) / (n_mem - 1)
            cov_s += 1e-6 * np.eye(n_st)

            R = np.eye(len(observations)) * observation_variance
            ens_obs_s = ens[:, observed_indices]
            ens_obs_mean_s = np.mean(ens_obs_s, axis=0)
            innovation_s = observations - ens_obs_mean_s

            P_oo_s = cov_s[np.ix_(observed_indices, observed_indices)]
            S_s = P_oo_s + R
            S_reg_s = S_s + 1e-8 * np.eye(len(observations))
            PHt_s = cov_s[:, observed_indices]
            try:
                K_s = np.linalg.solve(S_reg_s, PHt_s.T).T
            except np.linalg.LinAlgError:
                K_s = PHt_s @ np.linalg.pinv(S_reg_s)

            localized_K_s = _localize_gain(K_s, observed_indices, n_st, decay_length)
            member_innov_s = observations[None, :] - ens_obs_s
            update_s = member_innov_s @ localized_K_s.T
            ens = ens + g * update_s

        # Re-propagate through the dynamics for the next scattering.
        ens = _rk4_propagate(ens, n_steps, dt, F)

    # Final safety: clip and ensure finiteness.
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)

    return ens

# EVOLVE-BLOCK-END
