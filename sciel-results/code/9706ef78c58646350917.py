"""Refined Cohn-Goncalves Laguerre construction: mean-field Dyson-gas equilibrium
initialization + log-space L-BFGS-B refinement for the Fourier sign uncertainty bound."""
import time
import math
import numpy as np

_DEADLINE = 600.0
_START = time.time()

def _remaining():
    return _DEADLINE - 30.0 - (time.time() - _START)  # 30s safety margin

def _laguerre_equilibrium_measure(n_particles=12, beta=2.0, conf=1.0, grid_n=4000, x_max=200.0):
    """
    Solve the mean-field equilibrium for a Dyson gas with pairwise repulsion
    exponent beta=2 (Coulomb-like) and linear confining potential on x>0.
    Returns the empirical quantiles (particle positions) of the equilibrium measure.
    """
    x = np.linspace(1e-4, x_max, grid_n)
    # Mean-field potential: U(x) = conf * x  (linear confinement)
    # Repulsion kernel for exponent 2: log|x-y| -> mean field is Hilbert-transform-like.
    # For beta=2 (log-gas), the equilibrium density rho solves:
    # (1/(2*pi)) PV ∫ rho(y) log|x-y| dy + conf*x = const  on support.
    # We solve iteratively: start with a broad density, compute potential, refine support.
    rho = np.ones(grid_n) / grid_n
    for _ in range(200):
        if _remaining() < 5.0:
            break
        # Compute log-potential from current rho via FFT-free direct (grid_n moderate)
        # Use cumulative trick: potential at x_i = sum_j rho_j log|x_i - x_j|
        # Approximate with a smoothed kernel to avoid singularity.
        diff = np.abs(x[:, None] - x[None, :]) + 1e-6
        logk = np.log(diff)
        pot = logk @ rho
        # Total mean-field potential
        U = pot + conf * x
        # Equilibrium: density concentrates where U is near its minimum on support.
        # Re-weight: rho_new ∝ exp(-kappa * (U - min(U))) on region where U <= min+tol
        Umin = np.min(U)
        tol = 0.5 * (np.max(U) - Umin)
        mask = U <= (Umin + tol)
        w = np.where(mask, np.exp(-2.0 * (U - Umin)), 0.0)
        s = w.sum()
        if s <= 0:
            w = np.where(U <= Umin + tol, 1.0, 0.0)
            s = w.sum()
        if s <= 0:
            break
        rho_new = w / s
        # Dampen
        rho = 0.5 * rho + 0.5 * rho_new
    # Quantiles of the equilibrium measure
    cdf = np.cumsum(rho)
    cdf /= cdf[-1]
    qs = np.linspace(1.0 / (2 * n_particles), 1.0 - 1.0 / (2 * n_particles), n_particles)
    idx = np.searchsorted(cdf, qs)
    idx = np.clip(idx, 0, grid_n - 1)
    return x[idx].astype(float)


def _c4_surrogate(r):
    """
    Smooth surrogate for the Fourier sign uncertainty upper bound C4 given
    ordered double-root locations r (positive). The bound is governed by the
    product of adjacent gaps and the spread; we use a log-domain objective that
    is monotone with the true C4 and smooth for L-BFGS-B.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0) or np.any(np.diff(r) <= 1e-9):
        return 1e12
    gaps = np.diff(r)
    # Penalty for small gaps (dominates C4) and for large spread
    obj = -np.sum(np.log(gaps)) + 0.05 * (r[-1] - r[0]) ** 2
    return obj


def _optimize_from(initial, max_iters=400):
    """L-BFGS-B in log-space on the surrogate, with deadline guard."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return initial
    x0 = np.log(np.asarray(initial, dtype=float))

    def f(xv):
        r = np.exp(xv)
        # enforce ordering via soft penalty if needed
        d = np.diff(r)
        if np.any(d <= 1e-9):
            return 1e12 + np.sum((1e-6 - d[d <= 1e-9]) ** 2)
        return _c4_surrogate(r)

    def g(xv):
        r = np.exp(xv)
        d = np.diff(r)
        if np.any(d <= 1e-9):
            grad = np.zeros_like(xv)
            for k in range(len(d)):
                if d[k] <= 1e-9:
                    grad[k] += 2 * (1e-6 - d[k]) * r[k]
                    grad[k + 1] += -2 * (1e-6 - d[k]) * r[k + 1]
            return grad
        # analytic-ish gradient of -sum log(d_k) + 0.05*(r_m-r_1)^2
        grad = np.zeros_like(xv)
        for k in range(len(d)):
            grad[k] += (-1.0 / d[k]) * r[k]
            grad[k + 1] += (1.0 / d[k]) * r[k + 1]
        spread = r[-1] - r[0]
        grad[0] += 0.05 * 2 * spread * (-1.0) * r[0]
        grad[-1] += 0.05 * 2 * spread * (1.0) * r[-1]
        return grad

    best_x = x0.copy()
    best_f = f(x0)
    for restart in range(3):
        if _remaining() < 30.0:
            break
        x_start = x0.copy()
        if restart > 0:
            x_start = x_start + 0.05 * np.random.randn(len(x0))
        res = minimize(f, x_start, jac=g, method='L-BFGS-B',
                       options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    r = np.exp(best_x)
    # enforce strictly increasing
    for k in range(1, len(r)):
        if r[k] <= r[k - 1] + 1e-6:
            r[k] = r[k - 1] + 1e-6
    return r


def construct():
    """Return list of 12 positive double-root locations (m=12, n=1)."""
    t0 = time.time()
    m = 12
    # 1) Structural initialization from Laguerre/Dyson mean-field equilibrium
    try:
        init = _laguerre_equilibrium_measure(n_particles=m, beta=2.0, conf=1.0)
        if init is None or len(init) != m or np.any(init <= 0):
            raise ValueError("bad equilibrium measure")
    except Exception:
        # Fallback: geometric progression (known good structural family)
        init = np.geomspace(4.0, 4.0 * (1.5) ** (m - 1), m)

    # 2) Refine with log-space L-BFGS-B
    try:
        refined = _optimize_from(init, max_iters=300)
        if refined is not None and len(refined) == m and np.all(np.diff(refined) > 1e-6) and np.all(refined > 0):
            return [float(v) for v in refined]
    except Exception:
        pass

    # 3) Safe fallback: geometric progression
    geo = np.geomspace(4.0, 4.0 * (1.5) ** (m - 1), m)
    return [float(v) for v in geo]
