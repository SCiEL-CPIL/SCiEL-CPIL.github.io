"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = 320.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25   # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1); a_dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute static parts
_L1ADAG_L1 = _L1ADAG @ _L1
_L2ADAG_L2 = _L2ADAG @ _L2
_DECAY = 0.5 * (_L1ADAG_L1 + _L2ADAG_L2)

# States for fidelity (3D, padded with 0 for |2>)
# Inputs
_S0 = np.array([1.0, 0.0, 0.0], dtype=complex)
_S1 = np.array([0.0, 1.0, 0.0], dtype=complex)
_Sp = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Sm = (np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Spi = (np.array([1.0, 1j, 0.0]) / np.sqrt(2.0)).astype(complex)
_Smi = (np.array([1.0, -1j, 0.0]) / np.sqrt(2.0)).astype(complex)
_INIT_STATES = [_S0, _S1, _Sp, _Sm, _Spi, _Smi]

# Targets (X gate applied to inputs)
_T0 = np.array([0.0, 1.0, 0.0], dtype=complex)
_T1 = np.array([1.0, 0.0, 0.0], dtype=complex)
_Tp = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tm = (np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tpi = (np.array([1j, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tmi = (np.array([-1j, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_TARGET_STATES = [_T0, _T1, _Tp, _Tm, _Tpi, _Tmi]

def _make_gaussian_pulse(alpha, amp, sigma=40.0/6.0):
    """Generate a Gaussian DRAG pulse for initialization."""
    t = np.arange(_N) * _DT
    g = np.exp(-(t - _T/2.0)**2 / (2.0 * sigma**2))
    area = np.sum(g) * _DT
    I = amp * g * (np.pi / area)
    dIdt = np.zeros(_N)
    dIdt[1:-1] = (I[2:] - I[:-2]) / (2.0 * _DT)
    dIdt[0] = (I[1] - I[0]) / _DT
    dIdt[-1] = (I[-1] - I[-2]) / _DT
    Q = alpha * dIdt
    return np.column_stack([I, Q])

def _check_validity(pulse_2d):
    """Returns True if pulse satisfies amplitude and slew constraints."""
    r = np.sqrt(np.sum(pulse_2d**2, axis=1))
    if np.max(r) > _AMP + 1e-9:
        return False
    p0 = np.zeros(2)
    for i in range(_N):
        if np.linalg.norm(pulse_2d[i] - p0) / _DT > _SLEW + 1e-9:
            return False
        p0 = pulse_2d[i]
    if np.linalg.norm(p0) / _DT > _SLEW + 1e-9:
        return False
    return True

def _propagate(pulse_2d, det, scale):
    """Return 3x3 evolution matrix for given pulse, detuning, scale."""
    H0 = det * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - np.eye(3)))
    A_AP = _A + _ADAG
    A_AM = _A - _ADAG
    
    U = np.eye(3, dtype=complex)
    for i in range(_N):
        I = pulse_2d[i, 0]
        Q = pulse_2d[i, 1]
        H = H0 + 0.5 * scale * (I * A_AP + 1j * Q * A_AM)
        A_gen = -1j * H - _DECAY
        U = expm(A_gen * _DT) @ U
    return U

def _fidelity(U, psi_init, psi_target):
    psi_final = U @ psi_init
    overlap = np.vdot(psi_target, psi_final)
    return np.abs(overlap)**2

def _calculate_score(pulse_2d):
    """Calculate the raw score (mean + max infidelity). Returns None if invalid."""
    if not _check_validity(pulse_2d):
        return None
    
    scores = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            U = _propagate(pulse_2d, det, scale)
            fids = []
            for psi_i, psi_t in zip(_INIT_STATES, _TARGET_STATES):
                fids.append(_fidelity(U, psi_i, psi_t))
            scores.append(1.0 - np.mean(fids))
    return float(np.mean(scores) + np.max(scores))

def _penalized_score(pulse_flat):
    """Objective function for optimizer: Score + Penalty for constraint violations."""
    pulse_2d = pulse_flat.reshape(_N, 2)
    
    # Calculate raw score
    raw_score = _calculate_score(pulse_2d)
    if raw_score is None:
        # Invalid, compute penalty
        penalty = 0.0
        r = np.sqrt(np.sum(pulse_2d**2, axis=1))
        amp_viol = np.sum(np.maximum(0, r - _AMP)**2)
        
        p0 = np.zeros(2)
        slew_viol = 0.0
        for i in range(_N):
            diff = pulse_2d[i] - p0
            norm_diff = np.linalg.norm(diff)
            if norm_diff / _DT > _SLEW:
                slew_viol += (norm_diff / _DT - _SLEW)**2
            p0 = pulse_2d[i]
        last_diff = np.linalg.norm(p0)
        if last_diff / _DT > _SLEW:
            slew_viol += (last_diff / _DT - _SLEW)**2
            
        # Return a large number to push optimizer away
        return 10.0 + 100.0 * amp_viol + 100.0 * slew_viol
    
    # Add soft penalty to encourage staying within bounds even if valid
    r = np.sqrt(np.sum(pulse_2d**2, axis=1))
    amp_margin = np.max(np.maximum(0, r - _AMP * 0.95))
    
    p0 = np.zeros(2)
    slew_margin = 0.0
    for i in range(_N):
        diff = pulse_2d[i] - p0
        norm_diff = np.linalg.norm(diff)
        if norm_diff / _DT > _SLEW * 0.95:
            slew_margin += (norm_diff / _DT - _SLEW * 0.95)**2
        p0 = pulse_2d[i]
    last_diff = np.linalg.norm(p0)
    if last_diff / _DT > _SLEW * 0.95:
        slew_margin += (last_diff / _DT - _SLEW * 0.95)**2
        
    return raw_score + 10.0 * amp_margin + 10.0 * slew_margin

def _gradient(pulse_flat):
    """Calculate gradient using adjoint method for the nominal case (det=0, scale=1)."""
    pulse_2d = pulse_flat.reshape(_N, 2)
    
    H0 = 0.5 * _ANH * (_NOP @ (_NOP - np.eye(3)))
    A_AP = _A + _ADAG
    A_AM = _A - _ADAG
    scale = 1.0
    det = 0.0
    
    # Forward propagation
    U_prev = np.eye(3, dtype=complex)
    U_steps = [U_prev]
    A_steps = []
    for i in range(_N):
        I = pulse_2d[i, 0]
        Q = pulse_2d[i, 1]
        H = H0 + 0.5 * scale * (I * A_AP + 1j * Q * A_AM)
        A_gen = -1j * H - _DECAY
        A_steps.append(A_gen)
        U_next = expm(A_gen * _DT) @ U_prev
        U_steps.append(U_next)
        U_prev = U_next
        
    U_final = U_steps[-1]
    
    # Cost gradient w.r.t U_final
    # J = 1 - mean_fid
    dJ_dU = np.zeros((3,3), dtype=complex)
    for psi_i, psi_t in zip(_INIT_STATES, _TARGET_STATES):
        z = np.vdot(psi_t, U_final @ psi_i)
        term1 = np.conj(z) * np.outer(psi_t, psi_i)
        term2 = z * np.outer(psi_i, psi_t)
        dJ_dU += -(term1 + term2) / 6.0
        
    # Adjoint propagation
    lambda_next = dJ_dU
    grad = np.zeros_like(pulse_flat)
    
    dA_dI = -1j * 0.5 * scale * A_AP
    dA_dQ = 0.5 * scale * A_AM
    
    for k in range(_N - 1, -1, -1):
        U_k_minus_1 = U_steps[k]
        
        dJ_dI = np.trace(np.conj(lambda_next).T @ (_DT * dA_dI @ U_k_minus_1))
        dJ_dQ = np.trace(np.conj(lambda_next).T @ (_DT * dA_dQ @ U_k_minus_1))
        
        grad[k*2] = np.real(dJ_dI)
        grad[k*2+1] = np.real(dJ_dQ)
        
        A_k = A_steps[k]
        lambda_next = expm(A_k.conj().T * _DT) @ lambda_next
        
    return grad

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    # Initialize with a good Gaussian DRAG pulse
    init_pulse_2d = _make_gaussian_pulse(0.5, 1.0)
    init_pulse_flat = init_pulse_2d.flatten()
    
    # Check if initial is valid
    if not _check_validity(init_pulse_2d):
        max_r = np.max(np.sqrt(np.sum(init_pulse_2d**2, axis=1)))
        if max_r > 0:
            scale_factor = (_AMP * 0.9) / max_r
            init_pulse_2d = init_pulse_2d * scale_factor
            init_pulse_flat = init_pulse_2d.flatten()
            
    best_pulse = init_pulse_2d
    best_score = _calculate_score(best_pulse)
    if best_score is None:
        return np.zeros((_N, 2))

    # Optimization using L-BFGS-B with analytic gradient (nominal case)
    options = {
        'maxiter': 50,
        'ftol': 1e-6,
        'gtol': 1e-5,
        'disp': False
    }
    
    try:
        res = minimize(
            _penalized_score, 
            init_pulse_flat, 
            jac=_gradient,
            method='L-BFGS-B',
            options=options
        )
        
        if res.success or res.fun < (_penalized_score(init_pulse_flat)):
            opt_pulse_flat = res.x
            opt_pulse_2d = opt_pulse_flat.reshape(_N, 2)
            
            if _check_validity(opt_pulse_2d):
                opt_score = _calculate_score(opt_pulse_2d)
                if opt_score is not None and opt_score < best_score:
                    best_pulse = opt_pulse_2d
                    best_score = opt_score
            else:
                max_r = np.max(np.sqrt(np.sum(opt_pulse_2d**2, axis=1)))
                if max_r > 0:
                    scale_factor = min(1.0, (_AMP * 0.9) / max_r)
                    shrunk_pulse = opt_pulse_2d * scale_factor
                    if _check_validity(shrunk_pulse):
                        sh_score = _calculate_score(shrunk_pulse)
                        if sh_score is not None and sh_score < best_score:
                            best_pulse = shrunk_pulse
                            best_score = sh_score
                            
    except Exception:
        pass

    # Final check and return
    if not _check_validity(best_pulse):
        return np.zeros((_N, 2))
        
    return best_pulse

# EVOLVE-BLOCK-END
