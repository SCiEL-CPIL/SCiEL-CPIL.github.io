"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DEADLINE = 325.0

# --- Precomputed matrices ---
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_eye3 = np.eye(3, dtype=complex)

_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

def _build_bases():
    I3 = _eye3
    L_det = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    H_I = 0.5 * (_a + _adag)
    L_I = -1j * (np.kron(H_I, I3) - np.kron(I3, H_I.T))
    H_Q = 0.5j * (_a - _adag)
    L_Q = -1j * (np.kron(H_Q, I3) - np.kron(I3, H_Q.T))
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkd = Lk.conj().T
        LdL = Lkd @ Lk
        L_diss += np.kron(Lkd.T, Lk) - 0.5 * (np.kron(I3, LdL.T) + np.kron(LdL, I3))
    return L_det, L_I, L_Q, L_anh, L_diss

_L_DET, _L_I, _L_Q, _L_ANH, _L_DISS = _build_bases()

# 6 initial states as 9-vectors
_init_states = np.array([
    np.outer([1,0,0], [1,0,0]).flatten(),
    np.outer([0,1,0], [0,1,0]).flatten(),
    np.outer([(1,1,0)/np.sqrt(2)], [(1,1,0)/np.sqrt(2)].conj()).flatten(),
    np.outer([(1,-1,0)/np.sqrt(2)], [(1,-1,0)/np.sqrt(2)].conj()).flatten(),
    np.outer([(1,1j,0)/np.sqrt(2)], [(1,1j,0)/np.sqrt(2)].conj()).flatten(),
    np.outer([(1,-1j,0)/np.sqrt(2)], [(1,-1j,0)/np.sqrt(2)].conj()).flatten(),
], dtype=complex)

# 6 target 3-vectors for fidelity
_target_vecs = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [1j, 0, 1],
    [-1j, 0, 1],
], dtype=complex) / np.sqrt(2)
_target_vecs[0] = [0, 1, 0]
_target_vecs[1] = [1, 0, 0]

# 9 combos
_combos = [(d, s) for d in [-2*np.pi*0.002, 0.0, 2*np.pi*0.002] for s in [0.97, 1.0, 1.03]]
_det_vals = np.array([c[0] for c in _combos])
_scl_vals = np.array([c[1] for c in _combos])

# Precompute constant Liouvillian per combo
_L_const_all = np.array([d * _L_DET + _L_ANH + _L_DISS for d in _det_vals], dtype=complex)

def _evolve_all(pulse_I, pulse_Q):
    """Evolve all 6 states through all 9 combos. Returns (9, 6) fidelities."""
    fids = np.zeros((9, 6))
    for c in range(9):
        Lc = _L_const_all[c]
        scl = _scl_vals[c]
        rho = _init_states.copy()  # (6, 9)
        for i in range(N_STEPS):
            L = Lc + scl * (pulse_I[i] * _L_I + pulse_Q[i] * _L_Q)
            expL = sla.expm(L * DT)
            rho = rho @ expL.T  # (6,9) @ (9,9)
        for s in range(6):
            rho_mat = rho[s].reshape(3, 3)
            psi_t = _target_vecs[s]
            F = np.real(np.vdot(psi_t, rho_mat @ psi_t))
            fids[c, s] = F
    return fids

def _compute_score(pulse_I, pulse_Q):
    fids = _evolve_all(pulse_I, pulse_Q)
    mean_fids = np.mean(fids, axis=1)
    infids = 1.0 - mean_fids
    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amp > AMP_BOUND + 1e-12):
        return False
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-12:
            return False
    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND + 1e-12:
        return False
    return True

def _enforce(pulse_I, pulse_Q):
    pulse_I = pulse_I.copy()
    pulse_Q = pulse_Q.copy()
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        pulse_I[mask] *= AMP_BOUND / amp[mask]
        pulse_Q[mask] *= AMP_BOUND / amp[mask]
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale
    for idx in [0, N_STEPS-1]:
        s = np.sqrt(pulse_I[idx]**2 + pulse_Q[idx]**2) / DT
        if s > SLEW_BOUND:
            sc = SLEW_BOUND * DT / s
            pulse_I[idx] *= sc
            pulse_Q[idx] *= sc
    return pulse_I, pulse_Q

def _make_pulse(sigma, beta, i_scale=1.0, q_scale=1.0):
    t = np.arange(N_STEPS) * DT
    center = GATE_TIME / 2.0
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    dI = np.gradient(g, DT)
    I = i_scale * g
    Q = q_scale * beta * dI
    return _enforce(I, Q)

def _dop853_recheck(pulse_I, pulse_Q):
    def ham(t, state):
        idx = min(int(t / DT), N_STEPS - 1)
        I_val = pulse_I[idx]
        Q_val = pulse_Q[idx]
        rho = state.reshape(3, 3)
        H = (0.5 * ANHARMONICITY * _n @ (_n - _eye3)
             + 0.5 * I_val * (_a + _adag) + 0.5j * Q_val * (_a - _adag))
        comm = H @ rho - rho @ H
        diss = np.zeros((3, 3), dtype=complex)
        for Lk in [_L1, _L2]:
            Lkd = Lk.conj().T
            LdL = Lkd @ Lk
            diss += Lk @ rho @ Lkd - 0.5 * (LdL @ rho + rho @ LdL)
        return (-1j * comm + diss).flatten()
    
    rho0 = np.outer([1,0,0], [1,0,0]).flatten()
    sol = solve_ivp(ham, [0, GATE_TIME], rho0, method='DOP853', rtol=1e-10, atol=1e-12)
    if not sol.success:
        return None
    rho_f = sol.y[:, -1].reshape(3, 3)
    psi_t = np.array([0, 1, 0], dtype=complex)
    return np.real(np.vdot(psi_t, rho_f @ psi_t))

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    t0 = time.time()
    
    # Phase 1: Nelder-Mead on (sigma, beta, i_scale, q_scale)
    best_I, best_Q = _make_pulse(GATE_TIME/6.0, 0.15)
    best_score = _compute_score(best_I, best_Q)
    
    elapsed = time.time() - t0
    if DEADLINE - elapsed > 80:
        def obj(x):
            sigma, beta, is_, qs = x
            if sigma < 2 or sigma > 20 or abs(beta) > 1 or is_ < 0.5 or is_ > 1.5 or qs < 0.5 or qs > 1.5:
                return 100.0
            I, Q = _make_pulse(sigma, beta, is_, qs)
            if not _is_valid(I, Q):
                return 100.0
            return _compute_score(I, Q)
        
        x0 = [GATE_TIME/6.0, 0.15, 1.0, 1.0]
        res = minimize(obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 0.01, 'fatol': 1e-4, 'adaptive': True})
        
        if res.fun < best_score:
            I, Q = _make_pulse(res.x[0], res.x[1], res.x[2], res.x[3])
            if _is_valid(I, Q):
                best_I, best_Q, best_score = I, Q, res.fun
    
    # Phase 2: Local refinement — optimize central steps
    elapsed = time.time() - t0
    if DEADLINE - elapsed > 50:
        for trial in range(20):
            if time.time() - t0 > DEADLINE - 25:
                break
            I, Q = best_I.copy(), best_Q.copy()
            # Small random perturbation of central steps
            for s in range(20, 60):
                I[s] += 0.003 * (np.random.random() - 0.5)
                Q[s] += 0.003 * (np.random.random() - 0.5)
            I, Q = _enforce(I, Q)
            if _is_valid(I, Q):
                sc = _compute_score(I, Q)
                if sc < best_score:
                    best_I, best_Q, best_score = I, Q, sc
    
    # DOP853 recheck
    elapsed = time.time() - t0
    if DEADLINE - elapsed > 10:
        F = _dop853_recheck(best_I, best_Q)
    
    return np.column_stack((best_I, best_Q))
# EVOLVE-BLOCK-END
