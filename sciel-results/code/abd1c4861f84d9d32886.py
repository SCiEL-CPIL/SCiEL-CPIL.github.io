"""Optimize the Hermite-Gaussian uncertainty-product upper bound (C4).

The test function is g(x) = sum_k a_k * H_{4k}(x) * exp(-x^2/2).
Because the (physicists') Hermite-Gaussian basis is invariant under the
unitary Fourier transform up to a phase (i^{4k} = 1 for k a multiple of 2),
the Fourier transform of g is again a Hermite-Gaussian combination, which lets
us compute the uncertainty product A(g)*A(g_hat) exactly in coefficient space.
The P(0)=0 constraint fixes the H12 weight from (a0, a4, a8).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import factorial, eval_hermite
from scipy.optimize import minimize, differential_evolution


# ---------------------------------------------------------------------------
# Exact Hermite-Gaussian machinery
# ---------------------------------------------------------------------------
def _hermite_coeff(k, n):
    """Coefficient of x^n in the physicists' Hermite polynomial H_k(x)."""
    if k < n or (k - n) % 2 != 0:
        return 0.0
    m = (k - n) // 2
    return float((2.0 ** k) * factorial(k, exact=True)
                 / (factorial(n, exact=True) * factorial(m, exact=True)))


# Monomial moments  ∫ x^n e^{-x^2} dx  (odd -> 0)
_MOM = {0: np.sqrt(np.pi), 2: 0.5 * np.sqrt(np.pi), 4: 0.75 * np.sqrt(np.pi),
        6: 3.0 * np.sqrt(np.pi), 8: 10.5 * np.sqrt(np.pi),
        10: 42.0 * np.sqrt(np.pi), 12: 165.0 * np.sqrt(np.pi)}


def _poly_moments(coeffs):
    """Integral of (Σ c_j x^j) e^{-x^2} dx."""
    s = 0.0
    for j, c in enumerate(coeffs):
        if c != 0.0 and j % 2 == 0:
            s += c * _MOM[j]
    return s


def _h4_coeffs(k):
    """Polynomial coefficients (ascending) of H_{4k}(x)."""
    return [_hermite_coeff(4 * k, n) for n in range(4 * k + 1)]


# Precompute per-degree data: monomial coeffs, norm^2, and second-moment
# integrals (these are the x^2-weighted monomial moments).
_DEG = [0, 4, 8, 12]
_HCOEF = [_h4_coeffs(k) for k in range(4)]
_NORM2 = [_poly_moments(c) for c in _HCOEF]
_S2 = [_poly_moments([0.0 if j < 2 else c[j - 2] for c in _HCOEF])
       for c in _HCOEF]


def _g_moments(a0, a4, a8, a12):
    """(I0, I2, I4) = ∫ g^2, ∫ x^2 g^2, ∫ x^4 g^2  dx."""
    I0 = I2 = I4 = 0.0
    w = [a0, a4, a8, a12]
    for i in range(4):
        wi = w[i]
        if wi == 0.0:
            continue
        I0 += wi * wi * _NORM2[i]
        I2 += wi * wi * _S2[i]
        for j in range(4):
            wj = w[j]
            if wj == 0.0:
                continue
            prod = [0.0] * (len(_HCOEF[i]) + len(_HCOEF[j]) - 1)
            for p, ci in enumerate(_HCOEF[i]):
                if ci == 0.0:
                    continue
                for q, cj in enumerate(_HCOEF[j]):
                    if cj == 0.0:
                        continue
                    prod[p + q] += ci * cj
            I4 += wi * wj * _poly_moments(prod)
    return I0, I2, I4


# ---------------------------------------------------------------------------
# Objective: uncertainty product  A(g) * A(g_hat)
# ---------------------------------------------------------------------------
def _product(a0, a4, a8):
    a12 = -(a0 + a4 + a8) / _HCOEF[3][0]
    I0, I2, I4 = _g_moments(a0, a4, a8, a12)
    if I0 <= 1e-300:
        return np.inf
    # A(g) = (I4 - I2^2/I0) / I0 ;  A(g_hat) = (I2 - I4/I0) / I0
    return (I4 - I2 * I2 / I0) * (I2 - I4 / I0) / (I0 * I0)


def _grad(a0, a4, a8):
    """Analytic gradient of _product w.r.t. (a0, a4, a8) via the quotient rule."""
    a12 = -(a0 + a4 + a8) / _HCOEF[3][0]
    I0, I2, I4 = _g_moments(a0, a4, a8, a12)
    if I0 <= 1e-300:
        return np.zeros(3)
    w = [a0, a4, a8, a12]
    dI0 = [2.0 * w[i] * _NORM2[i] for i in range(4)]
    dI2 = [2.0 * w[i] * _S2[i] for i in range(4)]
    dI4 = [0.0] * 4
    for i in range(4):
        wi = w[i]
        if wi == 0.0:
            continue
        for j in range(4):
            wj = w[j]
            if wj == 0.0:
                continue
            prod = [0.0] * (len(_HCOEF[i]) + len(_HCOEF[j]) - 1)
            for p, ci in enumerate(_HCOEF[i]):
                if ci == 0.0:
                    continue
                for q, cj in enumerate(_HCOEF[j]):
                    if cj == 0.0:
                        continue
                    prod[p + q] += ci * cj
            dI4[i] += 2.0 * wj * _poly_moments(prod)
    dI4[3] *= _HCOEF[3][0]  # d a12 / d a_m = -1/H12(0)

    P = I4 - I2 * I2 / I0
    Q = I2 - I4 / I0
    dP = dI4 - (2.0 * I2 / I0) * dI2 + (I2 * I2 / I0 ** 2) * dI0
    dQ = dI2 - (1.0 / I0) * dI4 + (I4 / I0 ** 2) * dI0
    dlog = (dP / P + dQ / Q - 2.0 * dI0 / I0)
    return np.array([dlog[m] for m in range(3)])


# ---------------------------------------------------------------------------
# Multi-start optimization with a hard wall-clock deadline
# ---------------------------------------------------------------------------
def construct():
    deadline = time.time() + 165.0   # safety margin under the 210 s kill
    starts = [
        (-1.13, 0.04, 0.00031),
        (-1.0, 0.05, 0.0003),
        (-1.2, 0.03, 0.0004),
        (-0.9, 0.06, 0.0002),
        (-1.1, 0.0, 0.0003),
        (-1.3, 0.05, 0.0005),
        (-0.8, 0.04, 0.0002),
        (-1.13, 0.04, 0.00031),  # duplicate guard
    ]
    best = None
    best_val = np.inf

    # 1) Global polish with differential evolution (bounded box).
    try:
        if time.time() < deadline - 40.0:
            res = differential_evolution(
                _product,
                bounds=[(-2.0, 0.0), (-0.5, 0.5), (-0.01, 0.02)],
                seed=12345, maxiter=250, tol=1e-12,
                popsize=15, polish=True, init='sobol',
                workers=-1,
                options={'x0': [-1.13, 0.04, 0.00031]},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best = np.array(res.x, dtype=float)
    except Exception:
        pass

    # 2) Local SLSQP refinement from each start point (with analytic gradient).
    for s in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                _product, np.array(s, dtype=float),
                jac=_grad, method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best = np.array(res.x, dtype=float)
        except Exception:
            pass

    if best is None:
        # Fallback: seed coefficients (guaranteed finite, valid).
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    return best
# EVOLVE-BLOCK-END
