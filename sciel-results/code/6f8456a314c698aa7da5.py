"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution

# --- Constants ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_VALUES = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALE_VALUES = [0.97, 1.0, 1.03]
DEADLINE = 300.0  # safety margin before 330s kill

# Precompute matrices
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_eye3 = np.eye(3, dtype=complex)

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian building blocks
def _precompute_liouvillian_bases():
    I3 = _eye3
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh_base = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkd = Lk.conj().T
        LdL = Lkd @ Lk
        term1 = np.kron(Lkd.T, Lk)
        term2 = -0.5 * (np.kron(I3, LdL.T) + np.kron(LdL, I3))
        L_diss += term1 + term2
    return L_det_base, L_I_base, L_Q_base, L_anh_base, L_diss

_L_DET, _L_I, _L_Q, _L_ANH, _L_DISS = _precompute_liouvillian_bases()

# Initial states
_psi0 = np.array([1, 0, 0], dtype=complex)
_psi1 = np.array([0, 1, 0], dtype=complex)
_psi_plus = (_psi0 + _psi1) / np.sqrt(2)
_psi_minus = (_psi0 - _psi1) / np.sqrt(2)
_psi_plus_i = (_psi0 + 1j*_psi1) / np.sqrt(2)
_psi_minus_i = (_psi0 - 1j*_psi1) / np.sqrt(2)

# Ideal outputs (X applied to each)
_psi_outs = [
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 0, 0], dtype=complex),
    (_psi0 + _psi1) / np.sqrt(2),
    (_psi0 - _psi1) / np.sqrt(2),
    (_psi1 + 1j*_psi0) / np.sqrt(2),
    (_psi1 - 1j*_psi0) / np.sqrt(2),
]

# Precompute all initial density matrices as vectors
_rhos_init = [np.outer(p, p.conj()).flatten() for p in 
              [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]

def _evolve_unitary(pulse_I, pulse_Q, detuning, scale):
    """Evolve using unitary approximation (no dissipation) for speed during optimization."""
    L_const = detuning * _L_DET + _L_ANH
    U = _eye3.copy()
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        L = L_const + scale * I_val * _L_I + scale * Q_val * _L_Q
        expL = sla.expm(L * DT)
        U = expL @ U
    return U

def _fidelity_from_unitary(U):
    """Compute mean fidelity to X gate from a 9x9 superoperator (unitary approx)."""
    # Extract the 3x3 unitary from the superoperator acting on |0><0|
    # Actually, for unitary evolution, we can just track state vectors
    # Let's compute fidelity by evolving each initial state
    fidelities = []
    for psi_in, psi_out in zip(
        [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i],
        _psi_outs
    ):
        # Evolve psi_in through the same sequence
        psi = psi_in.copy()
        L_const = 0.0 * _L_DET + _L_ANH  # We need detuning/scale, but for unitary we skip
        # Actually, let's just use the full Liouvillian approach but track state vectors
        pass
    # This approach is getting complex. Let me use a simpler method.
    return 0.0  # placeholder

def _simulate_score_fast(pulse_I, pulse_Q):
    """Fast score computation using unitary evolution (no dissipation for speed)."""
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            # Build effective Hamiltonian evolution
            H_det = det * _n + 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
            
            # Evolve each initial state
            fidelities = []
            for psi_in, psi_out in zip(
                [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i],
                _psi_outs
            ):
                psi = psi_in.copy()
                for i in range(N_STEPS):
                    H_drive = 0.5 * scale * (pulse_I[i] * (_a + _adag) + 
                                              1j * pulse_Q[i] * (_a - _adag))
                    H_total = H_det + H_drive
                    U_step = sla.expm(-1j * H_total * DT)
                    psi = U_step @ psi
                F = np.abs(np.vdot(psi_out, psi))**2
                F = max(0.0, min(1.0, F))
                fidelities.append(F)
            F_mean = np.mean(fidelities)
            infids.append(1.0 - F_mean)
    
    return np.mean(infids) + np.max(infids)

def _simulate_score_full(pulse_I, pulse_Q):
    """Full score computation with dissipation (for final evaluation)."""
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            L_const = det * _L_DET + _L_ANH + _L_DISS
            
            fidelities = []
            for rho_init, psi_out in zip(_rhos_init, _psi_outs):
                rho = rho_init.copy()
                for i in range(N_STEPS):
                    L = L_const + scale * pulse_I[i] * _L_I + scale * pulse_Q[i] * _L_Q
                    expL = sla.expm(L * DT)
                    rho = expL @ rho
                rho_mat = rho.reshape(3, 3)
                F = np.real(np.vdot(psi_out, rho_mat @ psi_out))
                F = max(0.0, min(1.0, F))
                fidelities.append(F)
            
            F_mean = np.mean(fidelities)
            infids.append(1.0 - F_mean)
    
    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amp > AMP_BOUND + 1e-12):
        return False
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-12:
            return False
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND + 1e-12:
        return False
    return True

def _enforce_constraints(pulse_I, pulse_Q):
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
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slew0
        pulse_I[0] *= scale
        pulse_Q[0] *= scale
    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slewl
        pulse_I[-1] *= scale
        pulse_Q[-1] *= scale
    return pulse_I, pulse_Q

def _seed_pulse():
    """Generate DRAG-style Gaussian seed."""
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    center = GATE_TIME / 2.0
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    beta = 0.08
    dI_dt = np.gradient(g, DT)
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    return pulse_I, pulse_Q

def _dop853_recheck(pulse_I, pulse_Q):
    """Recheck nominal case with DOP853."""
    det = 0.0
    scale = 1.0
    L_const = det * _L_DET + _L_ANH + _L_DISS
    
    rhos_ours = []
    for rho_init in _rhos_init:
        rho = rho_init.copy()
        for i in range(N_STEPS):
            L = L_const + scale * pulse_I[i] * _L_I + scale * pulse_Q[i] * _L_Q
            expL = sla.expm(L * DT)
            rho = expL @ rho
        rhos_ours.append(rho.reshape(3, 3))
    
    def liouvillian_vec(t, rho_vec):
        idx = int(t / DT)
        if idx >= N_STEPS:
            idx = N_STEPS - 1
        I_val = pulse_I[idx]
        Q_val = pulse_Q[idx]
        L = L_const + scale * I_val * _L_I + scale * Q_val * _L_Q
        return L @ rho_vec
    
    max_diff = 0.0
    for idx, rho_init in enumerate(_rhos_init):
        sol = solve_ivp(liouvillian_vec, [0, GATE_TIME], rho_init,
                       method='DOP853', rtol=1e-12, atol=1e-14, max_step=0.1)
        rho_dop = sol.y[:, -1].reshape(3, 3)
        diff = np.max(np.abs(rho_dop - rhos_ours[idx]))
        max_diff = max(max_diff, diff)
    
    return max_diff

def _bfgs_optimize(pulse_I, pulse_Q, time_limit=200):
    """Direct BFGS optimization on full pulse vector (160 params)."""
    start_time = time.time()
    x0 = np.concatenate([pulse_I, pulse_Q])
    best_x = x0.copy()
    best_score = _simulate_score_fast(x0[:N_STEPS], x0[N_STEPS:])
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e6
        I = x[:N_STEPS]
        Q = x[N_STEPS:]
        I, Q = _enforce_constraints(I, Q)
        if not _is_valid(I, Q):
            return 1e6
        return _simulate_score_fast(I, Q)
    
    try:
        result = minimize(objective, x0, method='BFGS',
                         options={'maxiter': 500, 'gtol': 1e-6, 'disp': False})
        if result.fun < best_score:
            I, Q = _enforce_constraints(result.x[:N_STEPS], result.x[N_STEPS:])
            if _is_valid(I, Q):
                best_x = result.x.copy()
                best_score = result.fun
    except Exception:
        pass
    
    # Second pass with tighter tolerance
    remaining = time_limit - (time.time() - start_time)
    if remaining > 30:
        try:
            result2 = minimize(objective, best_x, method='BFGS',
                             options={'maxiter': 300, 'gtol': 1e-7, 'disp': False})
            if result2.fun < best_score:
                I, Q = _enforce_constraints(result2.x[:N_STEPS], result2.x[N_STEPS:])
                if _is_valid(I, Q):
                    best_x = result2.x.copy()
                    best_score = result2.fun
        except Exception:
            pass
    
    return best_x[:N_STEPS], best_x[N_STEPS:], best_score

def _random_restart_optimize(pulse_I, pulse_Q, time_limit=100):
    """Try several random perturbations of the seed, keep best."""
    start_time = time.time()
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_score = _simulate_score_fast(best_I, best_Q)
    
    np.random.seed(42)
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Create perturbed seed
        I = pulse_I.copy()
        Q = pulse_Q.copy()
        
        # Add small random noise
        noise_I = np.random.randn(N_STEPS) * 0.02
        noise_Q = np.random.randn(N_STEPS) * 0.01
        I += noise_I
        Q += noise_Q
        
        # Randomly shift center
        shift = np.random.uniform(-2, 2)
        t = np.arange(N_STEPS) * DT
        center = GATE_TIME / 2.0 + shift
        sigma = GATE_TIME / 6.0 * np.random.uniform(0.8, 1.2)
        g = np.exp(-0.5 * (t - center)**2 / sigma**2)
        g[0] = 0.0
        g[-1] = 0.0
        area = np.sum(g) * DT
        if area > 0:
            g *= np.pi / area
        dI_dt = np.gradient(g, DT)
        beta = np.random.uniform(-0.15, 0.15)
        
        I = g + noise_I * 0.5
        Q = beta * dI_dt + noise_Q * 0.5
        
        I, Q = _enforce_constraints(I, Q)
        if not _is_valid(I, Q):
            continue
        
        score = _simulate_score_fast(I, Q)
        if score < best_score:
            best_I = I.copy()
            best_Q = Q.copy()
            best_score = score
    
    return best_I, best_Q, best_score

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    
    # Generate seed pulse
    pulse_I, pulse_Q = _seed_pulse()
    seed_score = _simulate_score_fast(pulse_I, pulse_Q)
    
    # Phase 1: Random restarts to find good starting point
    restart_limit = min(60, DEADLINE - (time.time() - start_time) - 150)
    if restart_limit > 15:
        pulse_I, pulse_Q, _ = _random_restart_optimize(pulse_I, pulse_Q, time_limit=restart_limit)
    
    # Phase 2: BFGS direct optimization on full pulse
    bfgs_limit = min(180, DEADLINE - (time.time() - start_time) - 40)
    if bfgs_limit > 20:
        pulse_I, pulse_Q, _ = _bfgs_optimize(pulse_I, pulse_Q, time_limit=bfgs_limit)
    
    # DOP853 recheck
    if time.time() - start_time < DEADLINE - 10.0:
        diff = _dop853_recheck(pulse_I, pulse_Q)
        if diff > 2e-7:
            # If DOP853 disagrees, do one more refinement
            remaining = DEADLINE - (time.time() - start_time) - 5.0
            if remaining > 15:
                pulse_I, pulse_Q, _ = _bfgs_optimize(pulse_I, pulse_Q, time_limit=remaining)
    
    # Final validity check
    if not _is_valid(pulse_I, pulse_Q):
        pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    result = np.column_stack((pulse_I, pulse_Q))
    return result
# EVOLVE-BLOCK-END
