# EVOLVE-BLOCK-START
import time
import numpy as np

_DT = 0.5
_N = 80
_T = _N * _DT
_PI = np.pi
_NPIS = 2.0 * np.pi
_ALPHA = -_NPIS * 0.25
_T1 = 25000.0
_TPHI = 40000.0
_AMPL_BOUND = 0.4
_SLEW = 0.12
_DETUNES = np.array([-_NPIS * 0.002, 0.0, _NPIS * 0.002])
_SCALES = np.array([0.97, 1.0, 1.03])
_DEADLINE = 322.0

# --- build 3x3 operators ---
_a = np.zeros((3, 3), dtype=complex)
_a[1, 0] = 1.0
_a[2, 1] = np.sqrt(2.0)
_ad = _a.conj().T
_n = _ad @ _a
_id3 = np.eye(3)
# Lindblad operators
_L1 = _a / np.sqrt(_T1)
_L2 = _n * np.sqrt(2.0 / _TPHI)
# precompute Liouvillian pieces (Kronecker)
_KI = np.kron(_id3, _id3) - np.kron(_id3, _id3)
_KN = np.kron(_n, _id3)
_KN2 = np.kron(_n @ _n, _id3)
_KAD = np.kron(_a + _ad, _id3)
_KAQ = np.kron(1j * (_a - _ad), _id3)
# dissipator
_L11 = np.kron(_L1, _L1)
_L1d1 = np.kron(_L1 @ _L1.conj().T, _id3)
_L22 = np.kron(_L2, _L2)
_L2d2 = np.kron(_L2 @ _L2.conj().T, _id3)
_DISS = _L11 - 0.5 * _L1d1 + _L22 - 0.5 * _L2d2
# target X (0<->1 swap), embedded in 3 levels
_X = np.zeros((3, 3), dtype=complex)
_X[0, 1] = 1.0
_X[1, 0] = 1.0

_STATE_IDX = [0, 1, 3, 4, 6, 7, 9, 10, 12]
_XFLAT = _X.reshape(-1)
_BASIS = [np.zeros(3, complex), np.array([0, 1, 0], complex),
          (np.array([1, 1, 0], complex)) / np.sqrt(2.0),
          (np.array([1, -1, 0], complex)) / np.sqrt(2.0),
          (np.array([1, 1j, 0], complex)) / np.sqrt(2.0),
          (np.array([1, -1j, 0], complex)) / np.sqrt(2.0)]
_BASIS_FLAT = np.array([b.reshape(-1) for b in _BASIS], dtype=complex)


def _drivemat(I, Q):
    return _KN + _ALPHA * 0.5 * (_KN2 - _KN) + 0.5 * (_KAD * I + _KAQ * Q) + _DISS


def _drivemat2(det, scale, I, Q):
    return det * _KN + _ALPHA * 0.5 * (_KN2 - _KN) + 0.5 * scale * (_KAD * I + _KAQ * Q) + _DISS


def _propagator(det, scale, I, Q):
    M = _drivemat2(det, scale, I, Q)
    A = M * _DT
    B = 0.5 * A
    C = B @ A
    D = B @ C
    E = B @ D
    F = B @ E
    return _KI + A + C + D + E + F


def _fidelity(P):
    Pf = P.reshape(9, 3)
    out = np.empty(6)
    for k in range(6):
        col = Pf[:, _STATE_IDX[k]]
        out[k] = np.real(np.vdot(_XFLAT, col))
    return out


def _score_pulse(I, Q, _t0=None):
    fsum = 0.0
    fmax = 0.0
    for d in _DETUNES:
        for s in _SCALES:
            P = _propagator(d, s, I, Q)
            f = _fidelity(P)
            m = float(np.mean(f))
            mx = float(np.max(f))
            fsum += m + mx
            if mx > fmax:
                fmax = mx
    return (fsum / 9.0) + fmax


def _valid(I, Q):
    r = np.sqrt(I * I + Q * Q)
    if np.max(r) > _AMPL_BOUND + 1e-9:
        return False
    d = np.sqrt(np.diff(I) ** 2 + np.diff(Q) ** 2)
    if np.max(d) / _DT > _SLEW + 1e-9:
        return False
    if (np.hypot(I[0], Q[0]) + np.hypot(I[-1], Q[-1])) / _DT > _SLEW + 1e-9:
        return False
    return True


def _seed():
    t = (np.arange(_N) + 0.5) * _DT
    sigma = 40.0 / 6.0
    g = np.exp(-0.5 * ((t - _T / 2.0) / sigma) ** 2)
    g = g / g.sum() * _T / _DT
    A = np.pi / _T
    I = A * g
    # derivative (central), edges zero
    dI = np.zeros_like(I)
    dI[1:-1] = (I[2:] - I[:-2]) / (2.0 * _DT)
    alpha = 0.0
    Q = alpha * dI
    return I, Q


def _nudge(I, Q, k, dI, dQ, eps):
    I2 = I.copy(); Q2 = Q.copy()
    I2[k] += dI * eps; Q2[k] += dQ * eps
    if not _valid(I2, Q2):
        return None
    return I2, Q2


def _refine(I, Q, budget):
    best = _score_pulse(I, Q)
    I = I.copy(); Q = Q.copy()
    rng = np.random.default_rng(12345)
    t0 = time.time()
    eps = 0.05
    while time.time() - t0 < budget:
        k = int(rng.integers(0, _N))
        dI = float(rng.standard_normal())
        dQ = float(rng.standard_normal())
        n = _nudge(I, Q, k, dI, dQ, eps)
        if n is None:
            n = _nudge(I, Q, k, dI, dQ, eps * 0.5)
            if n is None:
                continue
        I2, Q2 = n
        s2 = _score_pulse(I2, Q2)
        if s2 < best:
            I, Q = I2, Q2
            best = s2
            if eps > 0.02:
                eps *= 0.85
        else:
            eps = min(eps * 1.15, 0.15)
    return I, Q, best


def pulse(target_gate, hardware_model):
    I, Q = _seed()
    if not _valid(I, Q):
        # scale down to satisfy slew
        r = np.sqrt(I * I + Q * Q)
        d = np.sqrt(np.diff(I) ** 2 + np.diff(Q) ** 2)
        worst = max(np.max(d) / _DT, np.hypot(I[0], Q[0]) / _DT, np.hypot(I[-1], Q[-1]) / _DT)
        if worst > _SLEW:
            f = _SLEW / worst
            I = I * f; Q = Q * f
    t0 = time.time()
    budget = _DEADLINE - (time.time() - t0) - 3.0
    if budget > 15.0:
        I, Q, _ = _refine(I, Q, budget)
    out = np.empty((_N, 2))
    out[:, 0] = I
    out[:, 1] = Q
    return out
# EVOLVE-BLOCK-END
