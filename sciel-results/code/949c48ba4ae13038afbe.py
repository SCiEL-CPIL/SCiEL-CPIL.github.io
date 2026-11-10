# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

# Precomputation: build the quadrature and the spherical-harmonic matrix Y.
# Quadrature: 70-point Gauss-Legendre in u = cos(theta), 130 equally spaced phi.
_NU = 70
_NPHI = 130
_K = 10

def _build_quadrature():
    # Gauss-Legendre nodes and weights on [-1, 1]
    u, wu = np.polynomial.legendre.leggauss(_NU)
    # phi grid, equally spaced
    phi = 2.0 * np.pi * np.arange(_NPHI) / _NPHI
    # Surface measure dOmega = dphi * du ; total mass 4*pi.
    # Weight for each (u, phi) point: wu_i * (2*pi/_NPHI)
    # Since phi is uniform, each phi carries weight 2*pi/_NPHI.
    wphi = 2.0 * np.pi / _NPHI
    # Build full weight grid (NU, NPHI)
    W = np.outer(wu, wphi)  # shape (NU, NPHI)
    # Verify total mass
    assert np.isclose(W.sum(), 4.0 * np.pi, rtol=1e-10), W.sum()
    return u, phi, W

def _spherical_harmonics(u, phi):
    """
    Return real orthonormal degree-K spherical harmonics Y_m for m=-K..K
    evaluated on the grid (NU, NPHI).

    Y_0  = N_0 P^0
    Y_m  = sqrt(2) N_m P^m cos(m phi)   for m > 0
    Y_m  = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
    with P^m = P_K^m(cos theta) including Condon-Shortley phase,
    N_m = sqrt(21/(4 pi) * (K-m)!/(K+m)!).
    """
    k = _K
    n = 2 * k + 1
    Y = np.zeros((u.shape[0], phi.shape[0], n), dtype=np.float64)
    # Precompute factorials
    import math
    log_fact = np.array([math.lgamma(i + 1) for i in range(2 * k + 1)])
    for m in range(-k, k + 1):
        am = abs(m)
        # N_m
        logN = 0.5 * (math.log(21.0) - math.log(4.0 * np.pi) + log_fact[k - am] - log_fact[k + am])
        N = math.exp(logN)
        # P_K^am(cos theta) with Condon-Shortley phase
        P = lpmv(am, k, u)  # lpmv includes Condon-Shortley phase (-1)^m
        idx = m + k
        if m == 0:
            Y[:, :, idx] = N * P
        elif m > 0:
            Y[:, :, idx] = math.sqrt(2.0) * N * P * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = math.sqrt(2.0) * N * P * np.sin(am * phi)[None, :]
    return Y

# Build global objects once at import time (fast, deterministic).
_U, _PHI, _W = _build_quadrature()
_Y = _spherical_harmonics(_U, _PHI)  # (NU, NPHI, 21)
# Flatten to (N, 21) where N = NU*NPHI
_Yflat = _Y.reshape(-1, 2 * _K + 1)
_Wflat = _W.reshape(-1)
_N = _Yflat.shape[0]

def _f_at(c):
    """Evaluate f at all quadrature points: shape (N,)."""
    return _Yflat @ c

def _obj_and_grad(c):
    """
    Objective: ||f||_4^4 with ||f||_2^2 = c.c = 1 (enforced by constraint).
    Return (value, gradient).
    value = sum_p w_p f_p^4
    grad  = 4 * Y^T (w * f^3)
    """
    f = _Yflat @ c
    f3 = f * f * f
    val = np.dot(_Wflat, f3 * f)
    grad = 4.0 * (_Yflat.T @ (_Wflat * f3))
    return val, grad

def _l4norm(c):
    f = _Yflat @ c
    return np.sqrt(np.dot(_Wflat, f * f * f * f)) ** 0.5

def _ratio(c):
    """Compute R = ||f||_4 / ||f||_2 (the actual evaluator quantity)."""
    f = _Yflat @ c
    n2 = np.dot(c, c)  # since Y orthonormal, ||f||_2^2 = c.c
    n4 = np.dot(_Wflat, f * f * f * f)
    if n2 <= 0:
        return np.inf
    return (n4 ** 0.25) / (n2 ** 0.5)

def _validate(c):
    """Check the public-task validity conditions."""
    if c is None or c.shape != (21,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    mx = np.max(np.abs(c))
    if mx < 1e-3 or mx > 1e5:
        return False
    return True

def rudin_problem10() -> np.ndarray:
    """
    Minimize R = ||f||_4 / ||f||_2 over real degree-10 spherical harmonics.
    Because R is scale-invariant, fix ||f||_2 = 1 (i.e. sum c_m^2 = 1) and
    minimize ||f||_4^4 = sum_p w_p f_p^4, a smooth quartic form in c with an
    analytic gradient. Use SLSQP with the sphere constraint and multiple
    random restarts, keeping the best valid result found.
    """
    deadline = time.time() + 600.0
    safety = 30.0  # safety margin in seconds
    rng = np.random.default_rng(12345)

    best_c = None
    best_ratio = np.inf

    def _consider(c):
        nonlocal best_c, best_ratio
        if not _validate(c):
            return
        r = _ratio(c)
        if np.isfinite(r) and r < best_ratio:
            best_ratio = r
            best_c = c.copy()

    # Seed: all-ones normalized to unit L2 (matches baseline direction).
    seed = np.ones(2 * _K + 1, dtype=np.float64)
    seed /= np.linalg.norm(seed)
    _consider(seed)

    # Constraints: sphere (equality).
    cons = [
        {
            'type': 'eq',
            'fun': lambda c: np.dot(c, c) - 1.0,
            'jac': lambda c: 2.0 * c,
        }
    ]

    def _objective(c):
        val, grad = _obj_and_grad(c)
        return val, grad

    # Number of restarts: build a set of diverse starting points.
    # We'll run a batch of random restarts plus a few structured ones.
    n_restarts = 40

    for ridx in range(n_restarts):
        if time.time() > deadline - safety:
            break
        if ridx == 0:
            c0 = seed.copy()
        else:
            c0 = rng.standard_normal(2 * _K + 1)
            c0 /= np.linalg.norm(c0)
        try:
            res = minimize(
                _objective,
                c0,
                jac=True,
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 400, 'ftol': 1e-12, 'disp': False},
            )
            c = res.x
            # Project back onto sphere to be safe
            nrm = np.linalg.norm(c)
            if nrm > 0:
                c = c / nrm
            _consider(c)
        except Exception:
            pass

    # If we have time, do a high-precision polish on the best with trust-constr
    if best_c is not None and time.time() < deadline - safety:
        try:
            from scipy.optimize import LinearConstraint, NonlinearConstraint
            cons2 = [NonlinearConstraint(lambda c: np.dot(c, c), 1.0, 1.0,
                                         jac=lambda c: 2.0 * c)]
            res2 = minimize(
                _objective,
                best_c,
                jac=True,
                method='trust-constr',
                constraints=cons2,
                options={'maxiter': 800, 'gtol': 1e-12, 'xtol': 1e-14, 'disp': False},
            )
            c2 = res2.x
            nrm = np.linalg.norm(c2)
            if nrm > 0:
                c2 = c2 / nrm
            _consider(c2)
        except Exception:
            pass

    if best_c is None:
        # Fallback to the seed (should not happen).
        best_c = seed

    # Final: ensure validity. The evaluator scales-invariant, but the contract
    # requires max|c| in [1e-3, 1e5]. Unit-norm vector has max|c| <= 1 and
    # >= 1/sqrt(21) ~ 0.218, so it's valid.
    if not _validate(best_c):
        best_c = seed

    return best_c

# EVOLVE-BLOCK-END
