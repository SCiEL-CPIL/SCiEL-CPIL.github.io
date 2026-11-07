import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE = 310.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Hardware constants
_N = 80
_DT = 0.5
_T = _N * _DT
_ANH = -2.0 * np.pi * 0.25
_AMP_BOUND = 0.4
_SLEW_BOUND = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# Operators
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A
_NN1 = _NOP @ (_NOP - np.eye(3))

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)

def _liouvillian_matrix(delta, scale, I, Q):
    drive = 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))
    H = delta * _NOP + 0.5 * _ANH * _NN1 + drive
    
    L_hamil = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
    
    def diss_term(L):
        L_dag = L.conj().T
        term1 = np.kron(L.conj(), L)
        LL_dag = L_dag @ L
        term2 = 0.5 * (np.kron(np.eye(3), LL_dag) + np.kron(LL_dag.T, np.eye(3)))
        return term1 - term2
    
    L_diss = diss_term(_L1) + diss_term(_L2)
    
    return L_hamil + L_diss

def _simulate_pulse(pulse, delta, scale):
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    U = np.eye(9)
    
    for i in range(_N):
        L_mat = _liouvillian_matrix(delta, scale, I[i], Q[i])
        U_step = expm(L_mat * _DT)
        U = U_step @ U
        
    return U

def _infidelity_for_case(pulse, delta, scale):
    U = _simulate_pulse(pulse, delta, scale)
    
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    
    psi_p = (psi_0 + psi_1) / np.sqrt(2)
    psi_m = (psi_0 - psi_1) / np.sqrt(2)
    psi_p_i = (psi_0 + 1j*psi_1) / np.sqrt(2)
    psi_m_i = (psi_0 - 1j*psi_1) / np.sqrt(2)
    
    targets = [psi_1, psi_0, psi_p, psi_m, psi_m_i, psi_p_i]
    inits = [psi_0, psi_1, psi_p, psi_m, psi_p_i, psi_m_i]
    
    fids = []
    for psi_t, psi_i in zip(targets, inits):
        rho_init = np.outer(psi_i, psi_i.conj()).flatten()
        rho_final = (U @ rho_init).reshape((3, 3))
        fid = np.real(np.vdot(psi_t, rho_final @ psi_t))
        fids.append(fid)
        
    return 1.0 - np.mean(fids)

def _score(pulse):
    infids = []
    for delta in _DET_GRID:
        for scale in _SCALE_GRID:
            infids.append(_infidelity_for_case(pulse, delta, scale))
    infids = np.array(infids)
    return np.mean(infids) + np.max(infids)

def _make_pulse(k_drag, sigma=40.0/6.0):
    t = np.arange(_N) * _DT + _DT/2.0
    t_center = _T / 2.0
    
    A = np.exp(-(t - t_center)**2 / (2.0 * sigma**2))
    A = A * (np.pi / (np.sum(A) * _DT))
    
    dA = np.gradient(A, _DT)
    
    I = A
    Q = k_drag * dA
    
    pulse = np.column_stack((I, Q))
    
    amps = np.sqrt(I**2 + Q**2)
    if np.max(amps) > _AMP_BOUND:
        pulse *= _AMP_BOUND / np.max(amps)
        
    diffs = np.diff(pulse, axis=0)
    norms = np.sqrt(np.sum(diffs**2, axis=1))
    max_slew = np.max(norms) / _DT
    if max_slew > _SLEW_BOUND:
        pulse *= _SLEW_BOUND / max_slew
        
    return pulse

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
        
    if _time_remaining() < 30:
        return _make_pulse(0.0)
        
    # Optimization
    def objective(k):
        if _time_remaining() < 10:
            return 1.0
        p = _make_pulse(k)
        return _score(p)
        
    # Grid search
    k_vals = np.linspace(-20, 20, 20)
    scores = [objective(k) for k in k_vals]
    idx = np.argmin(scores)
    k_best = k_vals[idx]
    
    if _time_remaining() < 20:
        return _make_pulse(k_best)
        
    # Local search
    res = minimize_scalar(objective, bounds=(k_best - 5, k_best + 5), method='bounded', options={'xatol': 0.1})
    if res.success:
        k_opt = res.x
    else:
        k_opt = k_best
        
    return _make_pulse(k_opt)
