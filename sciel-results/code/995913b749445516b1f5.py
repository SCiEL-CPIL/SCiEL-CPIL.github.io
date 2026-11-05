"""
Minimizes the third autocorrelation inequality constant C3 upper bound.

Structural change: represent h as a real window times a real part of a Zadoff-Chu
sequence modulated by a complex chirp (quadratic phase). Optimize window width,
chirp rate, and phase via scipy.optimize.minimize, then perform local coordinate
descent refinement on the full 400-dim h to reduce the objective.

value = abs(2*N*max(convolve(h,h))/sum(h)^2)
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0  # safety margin before 150s kill


def _evaluate(h):
    """Compute the objective value."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))


def _zadoff_chu(K, u, n):
    """
    Zadoff-Chu sequence of length K with root u, sampled at indices n.
    zc[n] = exp(-j * pi * u * n * (n+1) / K) for n in 0..K-1
    For n outside, wrap around.
    """
    n_mod = n % K
    return np.exp(-1j * np.pi * u * n_mod * (n_mod + 1) / K)


def _build_h_from_params(params):
    """
    Build h from parameters:
    params[0]: window width parameter w (0.1 to 1.0)
    params[1]: Zadoff-Chu root u (integer-ish, 1..N-1)
    params[2]: chirp rate alpha
    params[3]: global phase phi
    """
    w = params[0]
    u = int(round(params[1])) % N
    if u == 0:
        u = 1
    alpha = params[2]
    phi = params[3]

    n = np.arange(N, dtype=np.float64)
    # Symmetric window centered in [0, N)
    center = (N - 1) / 2.0
    window = np.exp(-((n - center) / (w * N * 0.25)) ** 2)

    # Zadoff-Chu
    zc = _zadoff_chu(N, u, n.astype(np.int64))

    # Chirp modulation
    chirp = np.exp(1j * alpha * n * n / N)

    # Combine
    z = window * zc * chirp * np.exp(1j * phi)
    h = np.real(z)

    # Ensure non-zero sum
    s = np.sum(h)
    if abs(s) < 1e-10:
        h = h + 0.01
    return h


def _objective(params):
    h = _build_h_from_params(params)
    return _evaluate(h)


def _local_refine(h, time_budget=30.0):
    """
    Local coordinate descent refinement on h.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    best_val = _evaluate(h)
    start = time.time()
    step = 0.1
    improved = True

    while improved and (time.time() - start) < time_budget:
        improved = False
        # Coordinate descent: try small perturbations on random subsets
        for _ in range(200):
            if (time.time() - start) > time_budget:
                break
            idx = np.random.randint(0, N, size=min(10, N))
            for i in idx:
                for delta in [-step, step]:
                    h_new = h.copy()
                    h_new[i] += delta
                    v = _evaluate(h_new)
                    if v < best_val - 1e-10:
                        h = h_new
                        best_val = v
                        improved = True
                        break
        # Decay step size
        step *= 0.85
        if step < 1e-6:
            break

    return h, best_val


def construct():
    """
    Returns the optimal step heights.
    """
    t0 = time.time()
    deadline = t0 + DEADLINE - 5.0  # 5s safety margin

    np.random.seed(42)

    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # 2.0

    # Phase 1: Optimize parametric family
    # Multiple restarts with different initial params
    u_candidates = [1, 7, 13, 31, 53, 79, 127, 199, 251, 311]
    best_param_val = best_val

    for u_init in u_candidates:
        if time.time() > deadline:
            break
        w_init = np.random.uniform(0.3, 0.9)
        alpha_init = np.random.uniform(-2.0, 2.0)
        phi_init = np.random.uniform(0, 2 * np.pi)

        x0 = np.array([w_init, float(u_init), alpha_init, phi_init])

        try:
            res = minimize(
                _objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                         'adaptive': True}
            )
            if res.fun < best_param_val:
                best_param_val = res.fun
                h_candidate = _build_h_from_params(res.x)
                best_h = h_candidate
        except Exception:
            pass

    # Phase 2: Local refinement on best parametric result
    if time.time() < deadline - 10.0:
        budget = min(40.0, deadline - time.time() - 5.0)
        if budget > 5.0:
            best_h, best_val = _local_refine(best_h, time_budget=budget)

    # Phase 3: Additional random local refinement from a few random starts
    remaining = deadline - time.time() - 3.0
    if remaining > 15.0:
        for trial in range(3):
            if time.time() > deadline - 5.0:
                break
            # Random chirp-based initial point
            u_rand = np.random.randint(1, N)
            alpha_rand = np.random.uniform(-3.0, 3.0)
            phi_rand = np.random.uniform(0, 2 * np.pi)
            w_rand = np.random.uniform(0.2, 1.0)
            h_rand = _build_h_from_params([w_rand, float(u_rand), alpha_rand, phi_rand])
            v_rand = _evaluate(h_rand)
            if v_rand < best_val + 0.1:
                budget = min(20.0, deadline - time.time() - 3.0)
                if budget > 5.0:
                    h_ref, v_ref = _local_refine(h_rand, time_budget=budget)
                    if v_ref < best_val:
                        best_val = v_ref
                        best_h = h_ref

    return best_h
