"""
Strategy: Chirp-based representation for the third autocorrelation inequality.

We represent h[n] = offset + amplitude * cos(phase + chirp_rate * n * n / N)
This is a linear chirp (frequency increases linearly with n). The time-varying
phase helps decorrelate the autocorrelation sidelobes. We optimize the four
scalar parameters (offset, amplitude, phase, chirp_rate) using differential
evolution followed by Nelder-Mead local refinement.

The value to minimize is:
    value = abs(2 * N * max(convolve(h, h)) / sum(h)^2)

Since the convolution is of length 2N-1, and the maximum is taken over signed
entries (no abs inside), we want the maximum signed entry to be as small as
possible (ideally close to zero or negative, but the abs makes it positive).
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_DEADLINE = 140.0  # wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _chirp_h(params):
    """
    Generate step heights from chirp parameters.
    params = [offset, amplitude, phase, chirp_rate]
    h[n] = offset + amplitude * cos(phase + chirp_rate * n * n / N)
    """
    offset, amplitude, phase, chirp_rate = params
    n = np.arange(N, dtype=float)
    # Use n*n/N to normalize the chirp so chirp_rate is dimensionless
    arg = phase + chirp_rate * n * n / N
    h = offset + amplitude * np.cos(arg)
    return h


def _objective(params):
    """Objective function for optimization."""
    h = _chirp_h(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    return val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Best solution so far
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0

    # Define bounds for differential evolution
    # offset: keep sum away from zero, allow negative offsets
    # amplitude: can be positive or negative
    # phase: [0, 2*pi)
    # chirp_rate: allow a range of chirp rates
    bounds = [
        (-2.0, 2.0),    # offset
        (-3.0, 3.0),    # amplitude
        (0.0, 2 * np.pi),  # phase
        (-50.0, 50.0),  # chirp_rate
    ]

    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False,  # We'll do Nelder-Mead separately
                seed=42,
                init='sobol',
            )
            if de_result.success or _evaluate(_chirp_h(de_result.x)) < best_val:
                h_de = _chirp_h(de_result.x)
                val_de = _evaluate(h_de)
                if np.isfinite(val_de) and val_de < best_val:
                    best_val = val_de
                    best_h = h_de
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 20:
        # Get the best params from DE or use a reasonable default
        if time.time() < deadline - 20:
            try:
                # Re-run DE with polish=True to get polished result, or
                # use the de_result.x if available
                # Let's just do Nelder-Mead from the best point found
                if hasattr(de_result, 'x') and np.all(np.isfinite(de_result.x)):
                    x0 = de_result.x
                else:
                    x0 = np.array([1.0, 0.5, 0.0, 5.0])
            except NameError:
                x0 = np.array([1.0, 0.5, 0.0, 5.0])

            nm_result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
            )
            if np.all(np.isfinite(nm_result.x)):
                h_nm = _chirp_h(nm_result.x)
                val_nm = _evaluate(h_nm)
                if np.isfinite(val_nm) and val_nm < best_val:
                    best_val = val_nm
                    best_h = h_nm

    # Phase 3: Multi-start with different initial chirp parameters
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            offset = rng.uniform(-1.0, 1.5)
            amplitude = rng.uniform(-2.0, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            chirp_rate = rng.uniform(-30.0, 30.0)
            x0 = np.array([offset, amplitude, phase, chirp_rate])
            try:
                nm = minimize(
                    _objective, x0, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8},
                )
                if np.all(np.isfinite(nm.x)):
                    h_c = _chirp_h(nm.x)
                    v_c = _evaluate(h_c)
                    if np.isfinite(v_c) and v_c < best_val:
                        best_val = v_c
                        best_h = h_c
            except Exception:
                pass

    # Phase 4: Final greedy local search on the best h
    if time.time() < deadline - 3:
        best_val = _evaluate(best_h)
        rng = np.random.default_rng(999)
        for _ in range(50000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = best_h.copy()
            h_new[idx] += rng.normal(0, 0.02)
            v_new = _evaluate(h_new)
            if np.isfinite(v_new) and v_new < best_val - 1e-12:
                best_val = v_new
                best_h = h_new

    # Final validation
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)

    return best_h
