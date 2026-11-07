"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1)
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Precompute drive operator matrices
_A_AP = _A + _ADAG          # a + a†
_A_AM = _A - _ADAG          # a - a†

# Lindblad decay (static)
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T
_DECAY = 0.5 * (_L1ADAG @ _L1 + _L2ADAG @ _L2)

# Static Hamiltonian part (detuning + anharmonicity)
def _H0(det):
    return det * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - np.eye(3)))

# States
_S0 = np.array([1.0, 0.0, 0.0], dtype=complex)
_S1 = np.array([0.0, 1.0, 0.0], dtype=complex)
_Sp = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Sm = (np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Spi = (np.array([1.0, 1j, 0.0]) / np.sqrt(2.0)).astype(complex)
_Smi = (np.array([1.0, -1j, 0.0]) / np.sqrt(2.0)).astype(complex)
_INIT_STATES = [_S0, _S1, _Sp, _Sm, _Spi, _Smi]

_T0 = np.array([0.0, 1.0, 0.0], dtype=complex)
_T1 = np.array([1.0, 0.0, 0.0], dtype=complex)
_Tp = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tm = (np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tpi = (np.array([1j, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tmi = (np.array([-1j, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_TARGET_STATES = [_T0, _T1, _Tp, _Tm, _Tpi, _Tmi]

def _make_gaussian_drag(alpha, amp, sigma=None):
    t = np.arange(_N) * _DT
    if sigma is None:
        sigma = 40.0 / 6.0
    g = np.exp(-(t - _T/2.0)**2 / (2.0 * sigma**2))
    area = np.sum(g) * _DT
    I = amp * g * (np.pi / area)
    dIdt = np.zeros(_N)
    dIdt[1:-1] = (I[2:] - I[:-2]) / (2.0 * _DT)
    dIdt[0] = (I[1] - I[0]) / _DT
    dIdt[-1] = (I[-1] - I[-2]) / _DT
    Q = alpha * dIdt
    return np.column_stack([I, Q])

def _project_feasible(pulse):
    """Project pulse onto feasible region: amplitude <= _AMP, slew <= _SLEW."""
    p = pulse.copy()
    # Amplitude constraint: scale down if any step exceeds _AMP
    r = np.sqrt(np.sum(p**2, axis=1))
    max_r = np.max(r)
    if max_r > _AMP:
        p *= _AMP / max_r
    # Slew constraint: iteratively clamp differences
    for _ in range(10):
        # Check from zero to step 0
        diff0 = np.linalg.norm(p[0]) / _DT
        if diff0 > _SLEW:
            p[0] *= _SLEW * _DT / np.linalg.norm(p[0])
        # Check between consecutive steps
        for i in range(_N - 1):
            d = p[i+1] - p[i]
            dn = np.linalg.norm(d) / _DT
            if dn > _SLEW:
                # Reduce p[i+1] toward p[i]
                excess = dn - _SLEW
                p[i+1] -= d * (excess / dn)
        # Check from last step to zero
        dlast = np.linalg.norm(p[-1]) / _DT
        if dlast > _SLEW:
            p[-1] *= _SLEW * _DT / np.linalg.norm(p[-1])
    # Final amplitude check
    r = np.sqrt(np.sum(p**2, axis=1))
    max_r = np.max(r)
    if max_r > _AMP:
        p *= _AMP / max_r
    return p

def _valid(pulse_arr):
    r = np.sqrt(np.sum(pulse_arr**2, axis=1))
    if np.max(r) > _AMP + 1e-9:
        return False
    p0 = np.zeros(2)
    for i in range(_N):
        if np.linalg.norm(pulse_arr[i] - p0) / _DT > _SLEW + 1e-9:
            return False
        p0 = pulse_arr[i]
    if np.linalg.norm(p0) / _DT > _SLEW + 1e-9:
        return False
    return True

def _propagate(pulse_arr, det, scale):
    """Return 3x3 evolution matrix using precomputed static parts."""
    H0 = _H0(det)
    U = np.eye(3, dtype=complex)
    for i in range(_N):
        I = pulse_arr[i, 0]
        Q = pulse_arr[i, 1]
        H = H0 + 0.5 * scale * (I * _A_AP + 1j * Q * _A_AM)
        A_gen = -1j * H - _DECAY
        U = expm(A_gen * _DT) @ U
    return U

def _fidelity(U, psi_init, psi_target):
    psi_final = U @ psi_init
    overlap = np.vdot(psi_target, psi_final)
    return np.abs(overlap)**2

def _score(pulse_arr):
    if not _valid(pulse_arr):
        return None
    scores = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            U = _propagate(pulse_arr, det, scale)
            fids = []
            for psi_i, psi_t in zip(_INIT_STATES, _TARGET_STATES):
                fids.append(_fidelity(U, psi_i, psi_t))
            scores.append(1.0 - np.mean(fids))
    return float(np.mean(scores) + np.max(scores))

def _numerical_gradient(pulse, step=1e-4):
    """Compute gradient of score w.r.t. pulse parameters via finite differences."""
    grad = np.zeros_like(pulse)
    base_score = _score(pulse)
    if base_score is None:
        return None
    for i in range(_N):
        for j in range(2):
            p_plus = pulse.copy()
            p_plus[i, j] += step
            p_plus = _project_feasible(p_plus)
            s_plus = _score(p_plus)
            if s_plus is not None:
                grad[i, j] = (s_plus - base_score) / step
            else:
                grad[i, j] = 0.0
    return grad

def _gradient_descent(pulse_init, max_iter=200, lr=0.01):
    """Adaptive gradient descent with feasibility projection."""
    pulse = _project_feasible(pulse_init.copy())
    best_pulse = pulse.copy()
    best_score = _score(pulse)
    if best_score is None:
        return None, None

    lr = 0.01
    for it in range(max_iter):
        if _time_remaining() < 15.0:
            break
        grad = _numerical_gradient(pulse)
        if grad is None:
            break
        # Gradient descent step
        pulse_new = pulse - lr * grad
        pulse_new = _project_feasible(pulse_new)
        s_new = _score(pulse_new)
        if s_new is not None and s_new < best_score:
            best_score = s_new
            best_pulse = pulse_new.copy()
            pulse = pulse_new
            # Reduce learning rate if improvement is small
            if best_score < 0.01:
                lr *= 0.9
        else:
            # Reduce learning rate and try again
            lr *= 0.7
            if lr < 1e-6:
                break
    return best_pulse, best_score

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    # Multiple starting points
    starts = []
    # Gaussian-DRAG seeds
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for amp in [0.9, 1.0, 1.1]:
            for sigma in [3.0, 4.0, 5.0, 6.0]:
                p = _make_gaussian_drag(alpha, amp, sigma=sigma)
                p = _project_feasible(p)
                starts.append(p)
    # A few random feasible pulses
    rng = np.random.RandomState(42)
    for _ in range(5):
        p = rng.randn(_N, 2) * 0.1
        p = _project_feasible(p)
        starts.append(p)

    best_pulse = None
    best_score = np.inf

    for start in starts:
        if _time_remaining() < 30.0:
            break
        s = _score(start)
        if s is not None and s < best_score:
            best_score = s
            best_pulse = start.copy()

    # Gradient descent from the best start
    if best_pulse is not None and _time_remaining() > 60.0:
        gp, gs = _gradient_descent(best_pulse, max_iter=150)
        if gp is not None and gs is not None and gs < best_score:
            best_score = gs
            best_pulse = gp

    # Additional gradient descent from second-best start if time permits
    if _time_remaining() > 120.0:
        # Find second-best start
        scores = []
        for start in starts:
            s = _score(start)
            if s is not None:
                scores.append((s, start))
        scores.sort(key=lambda x: x[0])
        if len(scores) > 1:
            s2, p2 = scores[1]
            if s2 < best_score * 1.5:  # Only if close
                gp, gs = _gradient_descent(p2, max_iter=100)
                if gp is not None and gs is not None and gs < best_score:
                    best_score = gs
                    best_pulse = gp

    if best_pulse is None:
        # Fallback: simple Gaussian
        best_pulse = _project_feasible(_make_gaussian_drag(0.5, 1.0))

    return best_pulse
# EVOLVE-BLOCK-END
