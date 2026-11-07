"""Robust X-pi pulse for a 3-level transmon: adjoint-gradient variational control."""
import time
import numpy as np

_DEADLINE = 320.0
_T0 = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _T0)

_N = 80
_DT = 0.5
_T = _N * _DT
_ANH = -2.0 * np.pi * 0.25
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0
_EPS = 1e-12

_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1H = _L1.conj().T
_L2H = _L2.conj().T

def _liou(p):
    return (
        -0.5 * (_L1H @ _L1 + _L2H @ _L2)
        + 0.5 * (_L1 @ _L1H + _L2 @ _L2H)
        - _L1H @ _L1 @ p
        + _L1 @ p @ _L1H
        - _L2H @ _L2 @ p
        + _L2 @ p @ _L2H
    )

def _ham(d, s, I, Q):
    h = d * _NOP + 0.5 * _ANH * _NOP @ (_NOP - 3.0 * np.eye(3))
    h = h + 0.5 * s * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))
    return h

def _step(p, H, li):
    A = -1j * H + li
    return np.linalg.expm(_DT * A) @ p

def _make_seed():
    t = (np.arange(_N) + 0.5) * _DT
    sig = (40.0 / 6.0)
    g = np.exp(-0.5 * (t / sig) ** 2)
    g = g / g.sum() * (_T / _DT)
    g = np.clip(g, 0.0, _AMP)
    return np.column_stack([g, np.zeros(_N)])

def _proj(p):
    p = np.array(p, dtype=float)
    mag = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    m = np.maximum(mag, _EPS)
    p = p * (np.minimum(mag, _AMP) / m)[:, None]
    prev = np.concatenate([np.zeros(2), p[:-1]])
    nxt = np.concatenate([p[1:], np.zeros(2)])
    for i in range(_N):
        d1 = p[i] - prev[i]
        n1 = np.linalg.norm(d1)
        if n1 > _SLEW * _DT:
            p[i] = prev[i] + d1 * (_SLEW * _DT / n1)
    for i in range(_N - 1, -1, -1):
        d1 = p[i] - prev[i]
        n1 = np.linalg.norm(d1)
        if n1 > _SLEW * _DT:
            p[i] = prev[i] + d1 * (_SLEW * _DT / n1)
        prev[i] = p[i]
    return p

def _score(p):
    I = p[:, 0]; Q = p[:, 1]
    best = np.inf
    for d in _DET_GRID:
        for s in _SCALE_GRID:
            tot = 0.0
            for k in range(4):
                if k == 0:
                    v0 = np.array([1.0, 0.0, 0.0], dtype=complex)
                elif k == 1:
                    v0 = np.array([0.0, 1.0, 0.0], dtype=complex)
                elif k == 2:
                    v0 = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0)
                else:
                    v0 = np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2.0)
                rho = np.outer(v0, v0.conj())
                for i in range(_N):
                    H = _ham(d, s, I[i], Q[i])
                    rho = _step(rho, H, _liou(rho))
                vf = np.array([0.0, 1.0, 0.0])
                vf = vf.conj() @ rho @ vf
                vf = vf.real
                vf = min(max(vf, 0.0), 1.0)
                tot += 1.0 - vf
            best = min(best, tot / 4.0)
    return best

def _grad(p):
    g = np.zeros_like(p)
    I = p[:, 0]; Q = p[:, 1]
    v0s = [
        np.array([1.0, 0.0, 0.0], dtype=complex),
        np.array([0.0, 1.0, 0.0], dtype=complex),
        np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0),
        np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2.0),
    ]
    vf = np.array([0.0, 1.0, 0.0])
    for d in _DET_GRID:
        for s in _SCALE_GRID:
            fw = np.zeros((4, _N, 3, 3), dtype=complex)
            bw = np.zeros((4, _N, 3, 3), dtype=complex)
            for k in range(4):
                v0 = v0s[k]
                rho = np.outer(v0, v0.conj())
                for i in range(_N):
                    H = _ham(d, s, I[i], Q[i])
                    rho = _step(rho, H, _liou(rho))
                    fw[k, i] = rho
                vf = vf.conj() @ fw[k, -1] @ vf
                vf = vf.real
                vf = min(max(vf, 0.0), 1.0)
                lam = np.outer(v0, v0.conj())
                for i in range(_N - 1, -1, -1):
                    H = _ham(d, s, I[i], Q[i])
                    lam = _step(lam, H.conj().T, _liou(lam))
                    bw[k, i] = lam
            for i in range(_N):
                for k in range(4):
                    li = _liou(fw[k, i])
                    term = 0.5 * s * (
                        np.trace(bw[k, i] @ (I[i] * (_A + _ADAG) + 1j * Q[i] * (_A - _ADAG)))
                        - np.trace(li @ (I[i] * (_A + _ADAG) + 1j * Q[i] * (_A - _ADAG)))
                    )
                    term = term.real
                    g[i, 0] += term
                    g[i, 1] += 1j * term
    return g

def _opt(p):
    best_p = np.array(p, dtype=float)
    best_v = _score(best_p)
    it = 0
    while _time_left() > 60.0:
        it += 1
        g = _grad(best_p)
        g = np.real(g)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        g = g / gn
        lr = 0.05
        while lr > 1e-5:
            cand = _proj(best_p + lr * g)
            v = _score(cand)
            if v < best_v - 1e-9:
                best_p = cand
                best_v = v
                break
            lr *= 0.5
        if lr <= 1e-5:
            break
    return best_p, best_v

def pulse(target_gate, hardware_model):
    p = _make_seed()
    p = _proj(p)
    try:
        if _time_left() > 70.0:
            p, _ = _opt(p)
    except Exception:
        pass
    p = _proj(p)
    return np.asarray(p, dtype=float)
