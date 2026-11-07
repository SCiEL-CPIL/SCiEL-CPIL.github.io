"""Robust X-pi pulse for a 3-level transmon.
Two-Gaussian + DRAG basis, optimized via precomputed Lindblad simulation.
"""
import time
import numpy as np
from scipy.linalg import expm

T_END = time.time() + 300.0
_DT = 0.5
_N = 80
_ANH = -2 * np.pi * 0.25
_T1 = 25000.0
_TPHI = 40000.0
_AMP = 0.4
_SLEW = 0.12

# precomputed operators
_A = np.diag(np.sqrt([1.0, 2.0]), 1)
_ADAG = _A.conj().T
_NN = _ADAG @ _A
_L1 = _A / np.sqrt(_T1)
_L2 = _NN * np.sqrt(2.0 / _TPHI)

def _liouv(H):
    Hc = H.conj()
    return -1j * (np.kron(H, np.eye(3)) - np.kron(Hc, np.eye(3)))

def _linv(H):
    L = _liouv(H)
    L += _L1 @ _L1.conj().T @ _L1.conj() - 0.5 * (_L1 @ _L1.conj().T)
    L += _L2 @ _L2.conj().T @ _L2.conj() - 0.5 * (_L2 @ _L2.conj().T)
    return L

def _H_of(det, scale, I, Q):
    return det * _NN + 0.5 * _ANH * (_NN @ (_NN - np.eye(3))) \
        + 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))

# precompute Lindbladians for the 9 detuning/scale cases
_CASES = [(d, s) for d in (-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002)
          for s in (0.97, 1.0, 1.03)]
_LINV_CACHE = {}
for (det, scale) in _CASES:
    _LINV_CACHE[(det, scale)] = []

def _precompute(I, Q):
    for (det, scale) in _CASES:
        arr = _LINV_CACHE[(det, scale)]
        arr.clear()
        for i in range(_N):
            arr.append(expm(_DT * _linv(_H_of(det, scale, I[i], Q[i]))))
        arr.append(expm(_DT * _linv(_H_of(det, scale, 0.0, 0.0))))

def _pulse_vec(p):
    I = np.zeros(_N)
    Q = np.zeros(_N)
    t = np.arange(_N) * _DT
    g1 = np.exp(-((t - p[1]) ** 2) / (2 * p[2] ** 2))
    g2 = np.exp(-((t - p[3]) ** 2) / (2 * p[4] ** 2))
    g = p[0] * g1 + p[5] * g2
    dg = p[0] * g1 * (-(t - p[1]) / p[2] ** 2) + p[5] * g2 * (-(t - p[3]) / p[4] ** 2)
    I = g
    Q = p[6] * dg
    return I, Q

def _valid(I, Q):
    if np.max(np.sqrt(I * I + Q * Q)) > _AMP:
        return False
    dI = np.diff(I); dQ = np.diff(Q)
    if np.max(np.hypot(dI, dQ)) / _DT > _SLEW:
        return False
    if np.hypot(I[0], Q[0]) / _DT > _SLEW:
        return False
    if np.hypot(I[-1], Q[-1]) / _DT > _SLEW:
        return False
    return True

def _fidelity(U, psi):
    psi = psi / np.linalg.norm(psi)
    out = U @ psi
    fid = np.abs(np.vdot(psi, out)) ** 2
    return fid

def _score(p):
    I, Q = _pulse_vec(p)
    if not _valid(I, Q):
        return 1e9
    _precompute(I, Q)
    cases = []
    for (det, scale) in _CASES:
        U = np.eye(3)
        for i in range(_N + 1):
            U = _LINV_CACHE[(det, scale)][i] @ U
        psi0 = np.array([1, 0, 0], complex)
        psi1 = np.array([0, 1, 0], complex)
        psip = np.array([1, 1, 0], complex) / np.sqrt(2)
        psim = np.array([1, -1, 0], complex) / np.sqrt(2)
        psip2 = np.array([1, 1j, 0], complex) / np.sqrt(2)
        psim2 = np.array([1, -1j, 0], complex) / np.sqrt(2)
        f = (_fidelity(U, psi0) + _fidelity(U, psi1) + _fidelity(U, psip)
             + _fidelity(U, psim) + _fidelity(U, psip2) + _fidelity(U, psim2)) / 6.0
        cases.append(1.0 - f)
    return np.mean(cases) + np.max(cases)

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        # fallback to a valid DRAG gaussian
        t = np.arange(_N) * _DT
        g = np.exp(-((t - 20.0) ** 2) / (2 * (40.0 / 6.0) ** 2))
        I = g * (np.pi / np.sum(g) / _DT)
        Q = np.zeros(_N)
        return np.column_stack([I, Q])
    # seed: single gaussian DRAG
    t = np.arange(_N) * _DT
    g = np.exp(-((t - 20.0) ** 2) / (2 * (40.0 / 6.0) ** 2))
    I0 = g * (np.pi / np.sum(g) / _DT)
    Q0 = -0.4 * np.gradient(I0, _DT)
    best_p = np.array([1.0, 20.0, 6.67, 0.0, 6.67, 0.0, -0.4])
    best_p[5] = 0.0
    best = _score(best_p)
    if best > 1e8:
        best = 1e9
    # Nelder-Mead
    from scipy.optimize import minimize
    try:
        res = minimize(_score, best_p, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-6,
                                'adaptive': True})
        if res.fun < best:
            best_p = res.x
            best = res.fun
    except Exception:
        pass
    # local coordinate-descent refinement
    if time.time() < T_END - 60:
        step = 1.0
        for _ in range(120):
            if time.time() > T_END - 45:
                break
            improved = False
            for k in range(7):
                for sign in (1, -1):
                    cand = best_p.copy()
                    cand[k] += sign * step
                    v = _score(cand)
                    if v < best:
                        best_p = cand
                        best = v
                        improved = True
                        break
            if not improved:
                step *= 0.7
            if step < 1e-4:
                break
    I, Q = _pulse_vec(best_p)
    if not _valid(I, Q):
        I = I0; Q = Q0
    return np.column_stack([I, Q])
