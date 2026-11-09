"""Bermudan exercise-policy: model-and-optimize with Gauss-Hermite quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps

    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = training_paths.shape[0]

    # High-resolution log-spaced grid for sharp exercise boundary
    s_min = 0.1
    s_max = 3.0 * K
    n_grid = 8192
    grid = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_grid))

    # Gauss-Hermite quadrature with more points for accurate risk-neutral expectation
    n_gh = 128
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n_gh)
    z_nodes = np.sqrt(2.0) * gh_nodes
    z_weights = gh_weights / np.sqrt(np.pi)
    z_weights /= z_weights.sum()

    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    exp_factors = np.exp(drift + vol * z_nodes)  # (n_gh,)
    discount = np.exp(-r * dt)

    # Single high-accuracy backward induction pass
    g = grid
    ef = exp_factors
    zw = z_weights

    V = np.maximum(K - g, 0.0)
    decisions = {}

    for t in range(n_steps, 0, -1):
        # Vectorized: (n_grid, n_gh) matrix of next spots
        ns = g[:, None] * ef[None, :]
        ns_c = np.clip(ns, g[0], g[-1])

        idx = np.searchsorted(g, ns_c)
        idx = np.clip(idx, 1, len(g) - 1)
        sl = g[idx - 1]
        su = g[idx]
        denom = su - sl
        denom = np.where(denom < 1e-12, 1e-12, denom)
        w = np.clip((ns_c - sl) / denom, 0.0, 1.0)

        Vl = V[idx - 1]
        Vu = V[idx]
        Vi = (1.0 - w) * Vl + w * Vu

        cont = discount * (Vi @ zw)
        ex = np.maximum(K - g, 0.0)
        V_new = np.maximum(ex, cont)
        if t < n_steps:
            decisions[t] = ex >= cont
        V = V_new

    g, dec, ns, Kp, rp, dtp = g, decisions, n_steps, K, r, dt

    class ExercisePolicy:
        def __init__(self, grid, decisions, n_steps, K):
            self.grid = grid
            self.decisions = decisions
            self.n_steps = n_steps
            self.K = K

        def exercise(self, current_state, time):
            if time < 1 or time > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            cs = np.asarray(current_state, dtype=np.float64)
            if time not in self.decisions:
                return cs < self.K
            idx = np.searchsorted(self.grid, cs)
            idx = np.clip(idx, 0, len(self.grid) - 1)
            return self.decisions[time][idx]

    return ExercisePolicy(g, dec, n_steps, K)
# EVOLVE-BLOCK-END
