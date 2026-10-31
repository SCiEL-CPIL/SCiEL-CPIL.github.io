"""Optimize Hermite combination to reduce the A(f)*A(F(f)) upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

_START = time.time()
_DEADLINE = _START + 200.0  # safety margin before 210s kill

# Physicist's Hermite polynomials (even)
def H0(x): return np.ones_like(x)
def H2(x): return 4*x**2 - 2
def H4(x): return 16*x**4 - 48*x**2 + 12
def H6(x): return 64*x**6 - 480*x**4 + 720*x**2 - 120
def H8(x): return 256*x**8 - 3072*x**6 + 11520*x**4 - 11520*x**2 + 1680
def H12(x): return 4096*x**12 - 98304*x**10 + 1036800*x**8 - 5898240*x**6 + 15482880*x**4 - 18662400*x**2 + 6227040

_H12_0 = -10395.0  # H12(0)

def _g_and_dg(x, a):
    a0, a4, a8 = a
    a12 = (a0 + a4 + a8) / (-_H12_0)
    g = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    dg = np.gradient(g, x)
    return g, dg

def _objective(a, x):
    a = np.asarray(a, dtype=float)
    g, dg = _g_and_dg(x, a)
    g4 = g**4
    num = np.trapz(g*g, x)
    den = np.trapz(g4, x)
    if den <= 0 or not np.isfinite(den):
        return 1e18
    A_f = num / den
    # A(F(g)): F(g) = g (Fourier invariant), so same form with dg
    num2 = np.trapz(dg*dg, x)
    den2 = np.trapz(dg**4, x)
    if den2 <= 0 or not np.isfinite(den2):
        return 1e18
    A_F = num2 / den2
    return A_f * A_F

def _objective_grad(a, x):
    a = np.asarray(a, dtype=float)
    g, dg = _g_and_dg(x, a)
    g4 = g**4
    num = np.trapz(g*g, x)
    den = np.trapz(g4, x)
    if den <= 0 or not np.isfinite(den):
        return 1e18, np.array([0.0, 0.0, 0.0])
    A_f = num / den
    num2 = np.trapz(dg*dg, x)
    den2 = np.trapz(dg**4, x)
    if den2 <= 0 or not np.isfinite(den2):
        return 1e18, np.array([0.0, 0.0, 0.0])
    A_F = num2 / den2
    val = A_f * A_F

    # Analytic gradient
    dg0 = np.gradient(H0(x), x); dg4 = np.gradient(H4(x), x); dg8 = np.gradient(H8(x), x)
    dg12 = np.gradient(H12(x), x)
    a12 = (a[0] + a[1] + a[2]) / (-_H12_0)

    grad = np.zeros(3)
    for i, (di, dgi) in enumerate([(H0(x), dg0), (H4(x), dg4), (H8(x), dg8)]):
        # dA_f/da_i
        dnum = 2*np.trapz(g*di, x)
        dden = 4*np.trapz(g**3*di, x)
        dA_f = (dnum*den - num*dden) / den**2
        # dA_F/da_i
        dnum2 = 2*np.trapz(dg*dgi, x)
        dden2 = 4*np.trapz(dg**3*dgi, x)
        dA_F = (dnum2*den2 - num2*dden2) / den2**2
        # a12 depends on a_i: da12/da_i = 1/(-H12_0)
        # contribution from a12 change
        dA_f_a12 = (2*np.trapz(g*H12(x), x)*den - num*4*np.trapz(g**3*H12(x), x)) / den**2
        dA_F_a12 = (2*np.trapz(dg*dg12, x)*den2 - num2*4*np.trapz(dg**3*dg12, x)) / den2**2
        da12_da_i = 1.0 / (-_H12_0)
        grad[i] = A_F*(dA_f + dA_f_a12*da12_da_i) + A_f*(dA_F + dA_F_a12*da12_da_i)
    return val, grad

def _solve(x):
    """Multi-start L-BFGS-B with analytic gradient + polish."""
    best_val = 1e18
    best_a = np.array([-113.0/100, 1.0/25, 1.0/3240])
    seed_val = _objective(best_a, x)
    best_val = seed_val

    starts = [
        np.array([-113.0/100, 1.0/25, 1.0/3240]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.0, 0.05, 0.0002]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-0.9, 0.06, 0.0001]),
        np.array([-1.1, 0.045, 0.00025]),
    ]
    # Add random starts
    rng = np.random.RandomState(42)
    for _ in range(6):
        starts.append(np.array([-1.0 + rng.randn()*0.2, 0.04 + rng.randn()*0.01, 0.0003 + rng.randn()*0.0001]))

    for s in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective, s, args=(x,), jac=_objective_grad,
                          method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_a = res.x.copy()
        except Exception:
            pass

    # High-precision polish
    if time.time() < _DEADLINE - 15:
        try:
            res2 = minimize(_objective, best_a, args=(x,), jac=_objective_grad,
                          method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-15})
            if res2.fun < best_val:
                best_val = res2.fun
                best_a = res2.x.copy()
        except Exception:
            pass

    return best_a

def construct():
    # Fine grid for accurate integration
    x = np.linspace(-8, 8, 4001)
    a = _solve(x)
    # Final check
    val = _objective(a, x)
    if not np.isfinite(val) or val > 1.0:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
    return np.array([float(a[0]), float(a[1]), float(a[2])])
# EVOLVE-BLOCK-END
