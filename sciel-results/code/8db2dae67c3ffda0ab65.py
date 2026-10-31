"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combined strategy:
  - Global search via Differential Evolution (structural change to escape local minima)
  - Local refinement via SLSQP with epigraph formulation
  - Fourier pre-optimization for a good starting point
  - Multi-start local refinement
  - Final polish
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
STEP_W = 2.0 / N  # width of each step


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2].
    This is the actual metric: max over all shifts s of integral h(x)*(1-h(x+s)) dx.
    For step functions, the overlap function C(s) is piecewise linear in s, so the
    maximum occurs at s = k * step_w for integer k. We evaluate at all integer shifts.
    """
    n = len(h)
    h_padded = np.zeros(2 * n)
    h_padded[n:2*n] = h
    max_ov = 0.0
    for k in range(-n, n + 1):
        # overlap at shift k*step_w: sum_i h[i] * (1 - h[i+k]) * step_w
        # h[i+k] is 0 if i+k outside [0,n)
        ov = 0.0
        for i in range(n):
            j = i + k
            h_j = h[j] if 0 <= j < n else 0.0
            ov += h[i] * (1.0 - h_j)
        ov *= STEP_W
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _score_fast(h):
    """Faster score using numpy for the inner loop (still O(N^2) but vectorized per shift)."""
    n = len(h)
    max_ov = 0.0
    for k in range(-n, n + 1):
        h2 = np.zeros(n)
        if k >= 0:
            h2[k:] = h[:n - k]
        else:
            h2[:n + k] = h[-k:]
        ov = np.sum(h * (1.0 - h2)) * STEP_W
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))


def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(100):
        s = np.sum(h)
        diff = N / 2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    diff = N / 2.0 - s
    if abs(diff) > 1e-14:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h


def _fourier_basis(N, K):
    """Create Fourier basis matrix: K complex modes mapped to real."""
    i = np.arange(N)
    basis = np.zeros((N, 2 * K - 1))
    basis[:, 0] = 1.0
    for k in range(1, K):
        basis[:, 2 * k - 1] = np.cos(2 * np.pi * k * i / N)
        basis[:, 2 * k] = np.sin(2 * np.pi * k * i / N)
    return basis


def _local_refine(h0, deadline, maxiter=3000, ftol=1e-16):
    """SLSQP local refinement with epigraph variable t."""
    def objective(params):
        h = params[:N]
        t = params[N]
        # Compute all shift overlaps
        n = N
        overlaps = np.zeros(2 * n + 1)
        for idx, k in enumerate(range(-n, n + 1)):
            h2 = np.zeros(n)
            if k >= 0:
                h2[k:] = h[:n - k]
            else:
                h2[:n + k] = h[-k:]
            overlaps[idx] = np.sum(h * (1.0 - h2)) * STEP_W
        smax = _smooth_max(overlaps, alpha=500.0)
        return t + 10000.0 * np.maximum(0.0, smax - t) ** 2

    def constraint_sum(params):
        return np.sum(params[:N]) - N / 2.0

    def constraint_epigraph(params):
        h = params[:N]
        t = params[N]
        n = N
        max_ov = 0.0
        for k in range(-n, n + 1):
            h2 = np.zeros(n)
            if k >= 0:
                h2[k:] = h[:n - k]
            else:
                h2[:n + k] = h[-k:]
            ov = np.sum(h * (1.0 - h2)) * STEP_W
            if ov > max_ov:
                max_ov = ov
        return t - max_ov

    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_epigraph}
    ]

    h0 = _enforce_sum(h0)
    t0 = _score_fast(h0) + 0.01
    x0 = np.concatenate([h0, [t0]])

    res = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
    )

    if res.fun < 100.0:
        h_out = _enforce_sum(res.x[:N])
        return h_out
    return h0


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Initial solution: constant
    h_const = np.full(N, 0.5)
    current_score = _score_fast(h_const)
    h_final = h_const.copy()

    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N / 2.0) < 1e-9:
                sc = _score_fast(h_c)
                if sc < current_score - 1e-12:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Fourier-based optimization to get a good starting point
    try:
        if time.time() < deadline - 80.0:
            K = 10
            basis = _fourier_basis(N, K)
            n_params = 2 * K - 1

            c0_init = np.zeros(n_params)
            c0_init[0] = 0.5

            def fourier_h(c):
                return basis @ c

            def fourier_objective(c):
                h = fourier_h(c)
                penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0) ** 2) +
                                   np.sum(np.maximum(0, -h) ** 2))
                sum_penalty = 500.0 * (np.sum(h) - N / 2.0) ** 2
                # Compute overlaps at integer shifts
                n = N
                overlaps = np.zeros(2 * n + 1)
                for idx, k in enumerate(range(-n, n + 1)):
                    h2 = np.zeros(n)
                    if k >= 0:
                        h2[k:] = h[:n - k]
                    else:
                        h2[:n + k] = h[-k:]
                    overlaps[idx] = np.sum(h * (1.0 - h2)) * STEP_W
                smax = _smooth_max(overlaps, alpha=200.0)
                return smax + penalty + sum_penalty

            result_f = minimize(
                fourier_objective, c0_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
            )

            if result_f.fun < 10.0:
                h_f = fourier_h(result_f.x)
                _try_update(h_f)

            # Try more modes
            for K2 in [14, 20]:
                if time.time() > deadline - 70.0:
                    break
                basis2 = _fourier_basis(N, K2)
                n_params2 = 2 * K2 - 1
                c0_init2 = np.zeros(n_params2)
                c0_init2[0] = 0.5

                def fourier_objective2(c, b=basis2):
                    h = b @ c
                    penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0) ** 2) +
                                       np.sum(np.maximum(0, -h) ** 2))
                    sum_penalty = 500.0 * (np.sum(h) - N / 2.0) ** 2
                    n = N
                    overlaps = np.zeros(2 * n + 1)
                    for idx, k in enumerate(range(-n, n + 1)):
                        h2 = np.zeros(n)
                        if k >= 0:
                            h2[k:] = h[:n - k]
                        else:
                            h2[:n + k] = h[-k:]
                        overlaps[idx] = np.sum(h * (1.0 - h2)) * STEP_W
                    smax = _smooth_max(overlaps, alpha=200.0)
                    return smax + penalty + sum_penalty

                result_f2 = minimize(
                    fourier_objective2, c0_init2,
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-15, 'disp': False}
                )

                if result_f2.fun < 10.0:
                    h_f2 = basis2 @ result_f2.x
                    _try_update(h_f2)

    except Exception:
        pass

    # Phase 2: Differential Evolution global search (structural change)
    try:
        if time.time() < deadline - 50.0:
            bounds_de = [(0.0, 1.0)] * N

            def de_objective(h):
                # Fast evaluation for DE
                h = np.clip(h, 0.0, 1.0)
                n = N
                max_ov = 0.0
                for k in range(-n, n + 1):
                    h2 = np.zeros(n)
                    if k >= 0:
                        h2[k:] = h[:n - k]
                    else:
                        h2[:n + k] = h[-k:]
                    ov = np.sum(h * (1.0 - h2)) * STEP_W
                    if ov > max_ov:
                        max_ov = ov
                # Sum constraint penalty
                s = np.sum(h)
                sum_pen = 100.0 * (s - N / 2.0) ** 2
                return max_ov + sum_pen

            de_result = differential_evolution(
                de_objective, bounds_de,
                seed=42, maxiter=300, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, init='latinhypercube'
            )

            if de_result.fun < 10.0:
                h_de = _enforce_sum(de_result.x)
                _try_update(h_de)

    except Exception:
        pass

    # Phase 3: Local refinement from current best
    try:
        if time.time() < deadline - 25.0:
            h_refined = _local_refine(h_final, deadline, maxiter=3000, ftol=1e-16)
            _try_update(h_refined)
    except Exception:
        pass

    # Phase 4: Multi-start local refinement
    try:
        if time.time() < deadline - 15.0:
            np.random.seed(123)
            n_starts = 6
            for trial in range(n_starts):
                if time.time() > deadline - 10.0:
                    break

                i = np.arange(N)
                if trial == 0:
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * i / N)
                elif trial == 1:
                    h_rand = 0.5 + 0.3 * np.sin(4 * np.pi * i / N)
                elif trial == 2:
                    h_rand = 0.5 + 0.2 * np.sin(6 * np.pi * i / N) + 0.1 * np.sin(2 * np.pi * i / N)
                elif trial == 3:
                    # Golden ratio blocks
                    A = (3.0 - np.sqrt(5.0)) / 2.0
                    B = 1.0 - A
                    h_rand = np.zeros(N)
                    half = N // 2
                    h_rand[:half] = A
                    h_rand[half:] = B
                elif trial == 4:
                    h_rand = 0.5 + 0.3 * np.cos(2 * np.pi * i / N) + 0.1 * np.cos(4 * np.pi * i / N)
                else:
                    h_rand = np.random.rand(N)
                    kernel = np.ones(5) / 5
                    h_rand = np.convolve(h_rand, kernel, mode='same')
                    h_rand = 0.2 + 0.6 * (h_rand - h_rand.min()) / (h_rand.max() - h_rand.min() + 1e-10)

                h_rand = np.clip(h_rand, 0.0, 1.0)
                h_rand = _enforce_sum(h_rand)

                h_ms = _local_refine(h_rand, deadline, maxiter=1000, ftol=1e-14)
                _try_update(h_ms)

    except Exception:
        pass

    # Phase 5: Final polish
    try:
        if time.time() < deadline - 3.0:
            h_polished = _local_refine(h_final, deadline, maxiter=5000, ftol=1e-18)
            _try_update(h_polished)
    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
