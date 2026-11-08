# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_START = _time.time()
_DEADLINE = 312.0

_N = 80
_DT = 0.5
_T = _N * _DT
_ANH = -2.0 * np.pi * 0.25
_AMP = 0.4
_SLEW = 0.12
_T1 = 25000.0
_TPHI = 40000.0

# 3-level ladder operators
_A = np.diag([1.0, np.sqrt(2.0), 1.0]).astype(complex)
_ADAG = _A.conj().T
_NOP = _ADAG @ _A
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1A = _L1.conj().T
_L2A = _L2.conj().T

# reduced 2x2 operators (|0>,|1>)
_A2 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_ADAG2 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
_N2 = _ADAG2 @ _A2
_L12 = _A2 / np.sqrt(_T1)
_L22 = _N2 * np.sqrt(2.0 / _TPHI)
_L12A = _L12.conj().T
_L22A = _L22.conj().T

_AMP2 = np.array([[0.0, 1.0], [0.0, 0.0]])
_ADAG2m = np.array([[0.0, 0.0], [1.0, 0.0]])
_SLEW2 = np.array([[0.0, 0.0], [1.0, 0.0]])
_SLEW2m = np.array([[0.0, 0.0], [0.0, 1.0]])


def _seed_pulse():
    t = np.arange(_N) * _DT + _DT / 2.0
    g = np.exp(-(t - _T / 2.0) ** 2 / (2.0 * (_T / 6.0) ** 2))
    I = np.pi * g / np.sum(g)
    Q = np.zeros(_N)
    return I, Q


def _liou2(det, sc, I, Q):
    H = det * _N2 + 0.5 * _ANH * _N2 @ (_N2 - np.eye(2)) + 0.5 * sc * (
        I * (_A2 + _ADAG2) + 1j * Q * (_A2 - _ADAG2))
    L = -1j * (np.kron(H, np.eye(2)) - np.kron(np.eye(2), H))
    L += _L12A @ _L12 - 0.5 * (np.kron(_L12, _L12) + _L12A @ _L12A)
    L += _L22A @ _L22 - 0.5 * (np.kron(_L22, _L22) + _L22A @ _L22A)
    return L


def _step2(L):
    return expm(L * _DT)


def _fid_case(I, Q, det, sc):
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    for i in range(_N):
        rho = _step2(_liou2(det, sc, I[i], Q[i])) @ rho
    r00 = rho[0, 0].real
    r11 = rho[3, 3].real
    r01 = rho[1, 0]
    fid = 0.5 * (r00 + r11) + abs(r01)
    return 1.0 - fid


def _score9(I, Q):
    tot = 0.0
    mx = 0.0
    for det in (-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002):
        for sc in (0.97, 1.0, 1.03):
            f = _fid_case(I, Q, det, sc)
            tot += f
            if f > mx:
                mx = f
    return tot / 9.0 + mx


def _enforce_constraints(P):
    I, Q = P
    # amplitude
    r = np.hypot(I, Q)
    m = r > _AMP
    if np.any(m):
        s = _AMP / r[m]
        I[m] *= s
        Q[m] *= s
    # slew from zero into step 1
    r0 = np.hypot(I[0], Q[0])
    if r0 > _SLEW * _DT:
        s = _SLEW * _DT / r0
        I[0] *= s
        Q[0] *= s
    # slew between steps
    for i in range(_N - 1):
        dI = I[i + 1] - I[i]
        dQ = Q[i + 1] - Q[i]
        rr = np.hypot(dI, dQ)
        if rr > _SLEW * _DT:
            s = _SLEW * _DT / rr
            I[i + 1] = I[i] + dI * s
            Q[i + 1] = Q[i] + dQ * s
    # slew from last step to zero
    rl = np.hypot(I[-1], Q[-1])
    if rl > _SLEW * _DT:
        s = _SLEW * _DT / rl
        I[-1] *= s
        Q[-1] *= s
    return I, Q


def pulse(target_gate, hardware_model):
    I0, Q0 = _seed_pulse()
    base = np.concatenate([I0, Q0])
    best = base.copy()
    best_s = _score9(best[:_N], best[_N:])

    def obj(x):
        P = _enforce_constraints((base[:_N] + x[:_N], base[_N:] + x[_N:]))
        return _score9(P[0], P[1])

    bounds = [(-0.15, 0.15)] * (2 * _N)
    try:
        res = minimize(obj, np.zeros(2 * _N), method="L-BFGS-B",
                       bounds=bounds,
                       options=dict(maxiter=60, maxfun=400,
                                    disp=False, ftol=1e-9))
        if res.fun < best_s:
            best = res.x
            best_s = res.fun
    except Exception:
        pass
    return _enforce_constraints((best[:_N], best[_N:]))
# EVOLVE-BLOCK-END
