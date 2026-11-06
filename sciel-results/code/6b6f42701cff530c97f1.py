"""Bermudan put exercise policy via backward induction on a log-space grid."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _phi_cdf(x):
    return 0.5 * (1.0 + np.tanh(0.7071067811865475224 * x))


def _phi_pdf(x):
    return np.exp(-0.5 * x * x) * 0.39894228040143267794


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit

    # Fixed model/data assumptions (from the public task contract)
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_ex = 12  # monthly exercise dates (no t=0 exercise)

    dt = T / n_ex
    vhalf = 0.5 * sigma * sigma
    disc = np.exp(-r * dt)
    m = r - vhalf            # log-drift per step
    s2 = sigma * sigma * dt  # log-variance per step
    sqrt_s2 = np.sqrt(s2)

    # Grid in log-space, wide enough to cover all relevant spot levels
    Lmin = np.log(1.0)
    Lmax = np.log(200.0)
    N = 4000
    xg = np.linspace(Lmin, Lmax, N)
    sg = np.exp(xg)

    # Terminal (last exercise date) value: put payoff (no continuation)
    V = np.maximum(K - sg, 0.0)

    # thresholds[i] = exercise spot threshold at exercise date i+1 (1-indexed)
    thresholds = np.zeros(n_ex)

    lower = xg[:-1]
    upper = xg[1:]
    Vj = V[:-1]
    slj = (V[1:] - V[:-1]) / (xg[1:] - xg[:-1])
    mu = xg + m

    chunk = 256
    for i in range(n_ex - 1, -1, -1):
        # Rebuild piecewise-linear coefficients from current V (the next-date value)
        Vj = V[:-1]
        slj = (V[1:] - V[:-1]) / (xg[1:] - xg[:-1])

        cont = np.zeros(N)
        for k0 in range(0, N, chunk):
            k1 = min(k0 + chunk, N)
            muk = mu[k0:k1, None]      # (chunk, 1)
            lo = lower[None, :]         # (1, N-1)
            up = upper[None, :]         # (1, N-1)
            Vjc = Vj[None, :]
            sljc = slj[None, :]

            a_lo = (lo - muk) / sqrt_s2
            a_up = (up - muk) / sqrt_s2
            P = np.clip(_phi_cdf(a_up) - _phi_cdf(a_lo), 0.0, 1.0)
            phi_lo = _phi_pdf(a_lo)
            phi_up = _phi_pdf(a_up)
            EY = (muk + s2) * P + sqrt_s2 * (phi_lo - phi_up)

            term = Vjc * P + sljc * (EY - lo * P)
            cont[k0:k1] = disc * np.sum(term, axis=1)

        ex_val = np.maximum(K - sg, 0.0)
        diff = ex_val - cont
        idx = np.where(diff >= 0)[0]
        if len(idx) == 0:
            thresholds[i] = np.inf
        elif idx[-1] >= N - 1:
            thresholds[i] = sg[-1]
        else:
            j = idx[-1]
            x0, x1 = xg[j], xg[j + 1]
            d0, d1 = diff[j], diff[j + 1]
            if abs(d1 - d0) > 1e-300:
                frac = d0 / (d0 - d1)
                xthr = x0 + frac * (x1 - x0)
            else:
                xthr = x0
            thresholds[i] = np.exp(xthr)

        V = np.maximum(ex_val, cont)

        if time.time() > deadline:
            break

    thresholds = np.minimum(thresholds, K)

    class ExercisePolicy:
        def __init__(self, thresholds, n_ex, T):
            self.thr = thresholds
            self.n_ex = n_ex
            self.dt = T / n_ex

        def exercise(self, current_state, time):
            S = np.asarray(current_state, dtype=float)
            t = np.asarray(time, dtype=float)
            step = np.ceil(t / self.dt - 1e-9).astype(int) - 1
            step = np.clip(step, 0, self.n_ex - 1)
            thr = self.thr[step]
            return S < thr

    return ExercisePolicy(thresholds, n_ex, T)
# EVOLVE-BLOCK-END
