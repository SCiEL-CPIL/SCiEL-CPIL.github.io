"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_START_TIME = time.time()
_DEADLINE = 310.0

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
_NOP = _ADAG @ _A  # n = a^dagger a

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute L^dag L terms for Liouvillian
_L1DL1 = _L1ADAG @ _L1
_L2DL2 = _L2ADAG @ _L2

def _make_seed_pulse():
    """DRAG-style Gaussian pulse with area pi."""
    t = np.linspace(0, _T, _N, endpoint=False) + _DT/2
    sigma = 40.0 / 6.0
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - _T/2) / sigma) ** 2)
    # Offset to make it zero at edges (approximately)
    # Actually, let's just use a simple Gaussian centered at T/2
    # Area should be pi
    area = np.sum(env) * _DT
    env = env * (np.pi / area)
    
    # DRAG correction: derivative of envelope
    deriv = np.gradient(env, _DT)
    # DRAG beta parameter - tune to reduce leakage
    beta = 0.05
    q = beta * deriv
    
    # Ensure constraints
    i = env
    q = q
    # Normalize to meet amplitude constraint
    amp = np.sqrt(i**2 + q**2)
    max_amp = np.max(amp)
    if max_amp > _AMP:
        scale = _AMP / max_amp
        i = i * scale
        q = q * scale
    
    # Ensure slew rate constraint
    # Check from 0 to first step
    for k in range(_N - 1):
        di = i[k+1] - i[k]
        dq = q[k+1] - q[k]
        slew = np.sqrt(di**2 + dq**2) / _DT
        if slew > _SLEW:
            # Scale down
            pass  # We'll handle this in enforcement
    
    # Check from 0 to first
    slew0 = np.sqrt(i[0]**2 + q[0]**2) / _DT
    if slew0 > _SLEW:
        scale = _SLEW * _DT / np.sqrt(i[0]**2 + q[0]**2)
        i[0] *= scale
        q[0] *= scale
    
    # Check from last to 0
    slewN = np.sqrt(i[-1]**2 + q[-1]**2) / _DT
    if slewN > _SLEW:
        scale = _SLEW * _DT / np.sqrt(i[-1]**2 + q[-1]**2)
        i[-1] *= scale
        q[-1] *= scale
    
    pulse = np.column_stack([i, q])
    return pulse

def _enforce_constraints(pulse):
    """Enforce amplitude and slew rate constraints."""
    i = pulse[:, 0].copy()
    q = pulse[:, 1].copy()
    
    # Enforce amplitude constraint
    for k in range(_N):
        amp = np.sqrt(i[k]**2 + q[k]**2)
        if amp > _AMP:
            scale = _AMP / amp
            i[k] *= scale
            q[k] *= scale
    
    # Enforce slew rate constraint (including from 0)
    # From 0 to step 0
    amp0 = np.sqrt(i[0]**2 + q[0]**2)
    if amp0 > _SLEW * _DT:
        scale = _SLEW * _DT / amp0
        i[0] *= scale
        q[0] *= scale
    
    # Between steps
    for k in range(_N - 1):
        di = i[k+1] - i[k]
        dq = q[k+1] - q[k]
        slew = np.sqrt(di**2 + dq**2)
        if slew > _SLEW * _DT:
            # Reduce the step
            scale = _SLEW * _DT / slew
            i[k+1] = i[k] + di * scale
            q[k+1] = q[k] + dq * scale
            # Re-enforce amplitude
            amp = np.sqrt(i[k+1]**2 + q[k+1]**2)
            if amp > _AMP:
                scale2 = _AMP / amp
                i[k+1] *= scale2
                q[k+1] *= scale2
    
    # From last step to 0
    ampN = np.sqrt(i[-1]**2 + q[-1]**2)
    if ampN > _SLEW * _DT:
        scale = _SLEW * _DT / ampN
        i[-1] *= scale
        q[-1] *= scale
    
    return np.column_stack([i, q])

def _liouvillian_step_fast(detuning, scale, i, q):
    """Compute one Liouvillian step efficiently."""
    # H = detuning * n + 0.5 * anh * n(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    # n(n-1) = n^2 - n
    n2 = _NOP @ _NOP
    n_minus_1 = _NOP - 3*np.eye(3)  # n - I for 3x3
    
    # Actually n(n-1) = n^2 - n
    nn1 = n2 - _NOP
    
    H = detuning * _NOP + 0.5 * _ANH * nn1
    # Drive term
    drive = 0.5 * scale * (i * (_A + _ADAG) + 1j * q * (_A - _ADAG))
    H = H + drive
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k† - 0.5{L_k† L_k, rho})
    # In superoperator form: L = -i * (H ⊗ I - I ⊗ H†) + ...
    # For small dt, we can use: rho_{k+1} = (I - i*dt*H + ...) rho (I + i*dt*H† + ...) + decay terms
    # But the task says "exponentiates the Liouvillian", so we need expm
    
    # Build the 9x9 Liouvillian superoperator
    # Vectorization: vec(rho) is 9x1
    # L = -i (H ⊗ I - I ⊗ H.T) + sum_k (L_k ⊗ L_k*.T - 0.5 (L_k†L_k ⊗ I + I ⊗ (L_k†L_k).T))
    
    I9 = np.eye(9)
    Hc = H.conj()
    
    # Coherent part: -i(H ⊗ I - I ⊗ H.T)
    # Note: for vec(AB) = (B^T ⊗ A) vec(C)
    # d(rho)/dt = -i H rho + i rho H†
    # vec: -i (H^T ⊗ I) vec(rho) + i (I ⊗ H) vec(rho)  ... wait
    # Actually: vec(H rho) = (I ⊗ H) vec(rho) if we use column-major
    # Let's use: vec(A rho B) = (B^T ⊗ A) vec(rho)
    # d(rho)/dt = -i H rho + i rho H†
    # vec(d rho) = -i (I ⊗ H) vec(rho) + i ((H†)^T ⊗ I) vec(rho)
    #            = -i (I ⊗ H) vec(rho) + i (H^* ⊗ I) vec(rho)
    
    L_coh = -1j * (np.eye(3) @ H) + 1j * (H.conj().T @ np.eye(3))
    # This is getting messy. Let me build it properly.
    
    # Superoperator for -i[H, rho]:
    # vec(-i(H rho - rho H)) = -i ((H^T ⊗ I) - (I ⊗ H)) vec(rho)  ... no
    # vec(H rho) = (I ⊗ H) vec(rho)  [column-stacking, so vec(A X B) = (B^T ⊗ A) vec(X)]
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    # So vec(-i[H,rho]) = -i [(I ⊗ H) - (H^T ⊗ I)] vec(rho)
    
    I3 = np.eye(3)
    L_coh_super = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipative part
    L_diss_super = np.zeros((9, 9), dtype=complex)
    for L in [_L1, _L2]:
        LdagL = L.conj().T @ L
        # L rho L† -> vec = (L†^T ⊗ L) vec(rho) = (L* ⊗ L) vec(rho)
        L_diss_super += np.kron(L.conj(), L)
        # -0.5 {L†L, rho} -> vec = -0.5 [(L†L)^T ⊗ I + I ⊗ (L†L)] vec(rho)
        L_diss_super += -0.5 * (np.kron(LdagL.T, I3) + np.kron(I3, LdagL))
    
    L_super = L_coh_super + L_diss_super
    
    # Exponentiate
    Ldt = L_super * _DT
    expLdt = expm(Ldt)
    return expLdt

def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return final density matrix."""
    # Start from |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    # Vectorize
    rho_vec = rho.flatten()
    
    i_vals = pulse[:, 0]
    q_vals = pulse[:, 1]
    
    for k in range(_N):
        i_k = i_vals[k]
        q_k = q_vals[k]
        
        # Build Liouvillian for this step
        n2 = _NOP @ _NOP
        nn1 = n2 - _NOP
        
        H = detuning * _NOP + 0.5 * _ANH * nn1
        drive = 0.5 * scale * (i_k * (_A + _ADAG) + 1j * q_k * (_A - _ADAG))
        H = H + drive
        
        I3 = np.eye(3)
        L_coh_super = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
        
        L_diss_super = np.zeros((9, 9), dtype=complex)
        for L in [_L1, _L2]:
            LdagL = L.conj().T @ L
            L_diss_super += np.kron(L.conj(), L)
            L_diss_super += -0.5 * (np.kron(LdagL.T, I3) + np.kron(I3, LdagL))
        
        L_super = L_coh_super + L_diss_super
        Ldt = L_super * _DT
        expLdt = expm(Ldt)
        
        rho_vec = expLdt @ rho_vec
    
    rho_final = rho_vec.reshape(3, 3)
    return rho_final

def _fidelity_with_X(rho):
    """Compute fidelity to X gate. X|0> = |1>, X|1> = |0>."""
    # X matrix
    X = np.zeros((3, 3), dtype=complex)
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    
    # Fidelity = Tr(X† rho X) for pure state target... but we need mean fidelity
    # The task says: infidelity = 1 - mean fidelity to X over |0>, |1>, (|0>±|1>)/sqrt(2), (|0>±i|1>)/sqrt(2)
    # So we compute fidelity for each of the 6 initial states, where fidelity is |<psi_target|U|psi_init>|^2
    # But we have density matrix. Fidelity = Tr(|psi_target><psi_target| rho)
    
    # Actually, for a unitary U, fidelity of state |psi> under U is |<psi|U|psi>|^2? No.
    # The fidelity to the target gate X means: for each input state |psi>, the output should be X|psi>.
    # Fidelity = |<X psi | U psi>|^2 = |<psi| X† U |psi>|^2
    
    # But we're working with density matrices. Let me think...
    # If we start with |psi><psi|, evolve to rho, then fidelity = <psi| X† rho X |psi>
    
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),
    ]
    
    fidelities = []
    for psi in states:
        # Target state: X|psi>
        psi_target = X @ psi
        # Projector
        P = np.outer(psi_target, psi_target.conj())
        # Fidelity = Tr(P rho)
        fid = np.trace(P @ rho).real
        fidelities.append(fid)
    
    return np.mean(fidelities)

def _score_pulse(pulse):
    """Compute the score for a pulse."""
    infidelities = []
    max_infid = 0
    
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            rho = _simulate_pulse(pulse, det, scale)
            fid = _fidelity_with_X(rho)
            infid = 1.0 - fid
            infidelities.append(infid)
            if infid > max_infid:
                max_infid = infid
    
    mean_infid = np.mean(infidelities)
    score = mean_infid + max_infid
    return score

def pulse(target_gate, hardware_model):
    """Generate the control pulse."""
    if _time_remaining() < 5.0:
        # Not enough time, return seed
        return _enforce_constraints(_make_seed_pulse())
    
    # Start with DRAG seed
    best_pulse = _make_seed_pulse()
    best_pulse = _enforce_constraints(best_pulse)
    best_score = _score_pulse(best_pulse)
    
    # Try local optimization using gradient-free methods
    # Use Nelder-Mead or similar, but with constraint enforcement
    
    from scipy.optimize import minimize
    
    def objective(params):
        p = params.reshape(_N, 2)
        p = _enforce_constraints(p)
        return _score_pulse(p)
    
    # Flatten the pulse
    x0 = best_pulse.flatten()
    
    # Try a few optimization methods
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        if _time_remaining() < 10.0:
            break
        
        try:
            result = minimize(objective, x0, method=method, 
                             options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_pulse = _enforce_constraints(result.x.reshape(_N, 2))
                best_score = result.fun
                x0 = result.x
        except Exception:
            pass
    
    return best_pulse
# EVOLVE-BLOCK-END
