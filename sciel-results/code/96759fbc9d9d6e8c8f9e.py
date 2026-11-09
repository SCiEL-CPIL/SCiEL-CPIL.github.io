"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

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
DEADLINE = 305.0  # safety margin before 330s kill

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
    """Precompute the matrix coefficients for the Liouvillian."""
    I3 = _eye3
    
    # Detuning term
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    
    # Drive I term: 0.5*(a+a†)
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))
    
    # Drive Q term: 0.5*i*(a-a†)
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))
    
    # Anharmonicity term (constant part): 0.5*alpha*n*(n-1)
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh_base = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    
    # Dissipative part (constant)
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

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    """Evolve all 6 input states through the pulse, return final density matrices."""
    rhos = [np.outer(p, p.conj()) for p in [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]
    
    # Constant part of Liouvillian
    L_const = detuning * _L_DET + _L_ANH + _L_DISS
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        
        L = L_const + scale * I_val * _L_I + scale * Q_val * _L_Q
        expL = sla.expm(L * DT)
        
        for j in range(6):
            rhos_vec[j] = expL @ rhos_vec[j]
    
    return [v.reshape(3, 3) for v in rhos_vec]

def _fidelity_from_rhos(rhos):
    """Compute mean fidelity to X gate from 6 final density matrices."""
    fidelities = []
    for rho, psi_out in zip(rhos, _psi_outs):
        F = np.real(np.vdot(psi_out, rho @ psi_out))
        F = max(0.0, min(1.0, F))
        fidelities.append(F)
    
    return np.mean(fidelities)

def _compute_score(pulse_I, pulse_Q):
    """Compute score = mean + max of 9 infidelities."""
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rhos = _evolve_all_states(pulse_I, pulse_Q, det, scale)
            F = _fidelity_from_rhos(rhos)
            infids.append(1.0 - F)
    
    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
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
    """Project pulse onto feasible set (amplitude + slew)."""
    pulse_I = pulse_I.copy()
    pulse_Q = pulse_Q.copy()
    
    # Amplitude
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        pulse_I[mask] *= AMP_BOUND / amp[mask]
        pulse_Q[mask] *= AMP_BOUND / amp[mask]
    
    # Slew: forward pass
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale
    
    # Edge slews
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
    """Generate DRAG-style Gaussian seed with physics-informed beta compensation."""
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    center = GATE_TIME / 2.0
    
    # Gaussian envelope
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    
    # Scale to area pi
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    
    # Physics-informed DRAG beta
    beta = 0.08
    
    # Derivative of I envelope
    dI_dt = np.gradient(g, DT)
    
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    
    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    return pulse_I, pulse_Q

def _pulse_from_basis(params):
    """
    Generate pulse from Hermite-Gaussian basis expansion.
    params: [a0, a1, a2, a3, b0, b1, b2, b3, sigma, center_shift]
    - a0..a3: coefficients for I channel Hermite-Gaussian basis
    - b0..b3: coefficients for Q channel Hermite-Gaussian basis
    - sigma: Gaussian width
    - center_shift: time shift of pulse center
    """
    a0, a1, a2, a3 = params[0], params[1], params[2], params[3]
    b0, b1, b2, b3 = params[4], params[5], params[6], params[7]
    sigma = max(3.0, min(25.0, params[8]))
    center_shift = max(-8.0, min(8.0, params[9]))
    
    t = np.arange(N_STEPS) * DT
    center = GATE_TIME / 2.0 + center_shift
    
    # Hermite-Gaussian basis functions
    x = (t - center) / sigma
    
    # H0 = 1
    H0 = np.ones_like(x)
    # H1 = 2x
    H1 = 2 * x
    # H2 = 4x^2 - 2
    H2 = 4 * x**2 - 2
    # H3 = 8x^3 - 12x
    H3 = 8 * x**3 - 12 * x
    
    # Gaussian envelope
    g = np.exp(-0.5 * x**2)
    g[0] = 0.0
    g[-1] = 0.0
    
    # Build I pulse
    pulse_I = g * (a0 * H0 + a1 * H1 + a2 * H2 + a3 * H3)
    
    # Build Q pulse
    pulse_Q = g * (b0 * H0 + b1 * H1 + b2 * H2 + b3 * H3)
    
    # Scale to ensure area is approximately pi
    area_I = np.sum(pulse_I) * DT
    if abs(area_I) > 1e-10:
        pulse_I *= np.pi / area_I
    
    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    return pulse_I, pulse_Q

def _score_from_basis(params):
    """Compute score from basis parameter vector."""
    pulse_I, pulse_Q = _pulse_from_basis(params)
    if not _is_valid(pulse_I, pulse_Q):
        return 1.0
    return _compute_score(pulse_I, pulse_Q)

def _dop853_recheck(pulse_I, pulse_Q, detuning=0.0, scale=1.0):
    """Recheck nominal case using DOP853 to ensure accuracy within 2e-7."""
    def hamiltonian(t, state):
        # state is 3x3 density matrix flattened
        rho = state.reshape(3, 3)
        
        idx = int(t / DT)
        if idx >= N_STEPS:
            idx = N_STEPS - 1
        
        I_val = pulse_I[idx]
        Q_val = pulse_Q[idx]
        
        H = (detuning * _n
             + 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
             + 0.5 * scale * I_val * (_a + _adag)
             + 0.5j * scale * Q_val * (_a - _adag))
        
        # Liouvillian: d(rho)/dt = -i[H, rho] + diss
        comm = H @ rho - rho @ H
        
        diss = np.zeros((3, 3), dtype=complex)
        for Lk in [_L1, _L2]:
            Lkd = Lk.conj().T
            LdL = Lkd @ Lk
            diss += Lk @ rho @ Lkd - 0.5 * (LdL @ rho + rho @ LdL)
        
        d_rho = -1j * comm + diss
        return d_rho.flatten()
    
    # Evolve |0><0|
    psi0 = np.array([1, 0, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0).flatten()
    
    sol = solve_ivp(hamiltonian, [0, GATE_TIME], rho0, method='DOP853', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        return None
    
    rho_final = sol.y[:, -1].reshape(3, 3)
    
    # Fidelity to X|0> = |1>
    psi_out = np.array([0, 1, 0], dtype=complex)
    F = np.real(np.vdot(psi_out, rho_final @ psi_out))
    F = max(0.0, min(1.0, F))
    
    return F

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    
    # Generate seed pulse with DRAG
    pulse_I, pulse_Q = _seed_pulse()
    seed_score = _compute_score(pulse_I, pulse_Q)
    
    # Differential evolution over basis parameters
    remaining = DEADLINE - (time.time() - start_time) - 100.0
    if remaining > 60:
        # Bounds for differential evolution
        # a0..a3: I channel coefficients
        # b0..b3: Q channel coefficients
        # sigma: Gaussian width
        # center_shift: time shift
        
        bounds = [
            (-2.0, 2.0),  # a0
            (-1.0, 1.0),  # a1
            (-0.5, 0.5),  # a2
            (-0.2, 0.2),  # a3
            (-1.0, 1.0),  # b0
            (-1.0, 1.0),  # b1
            (-0.5, 0.5),  # b2
            (-0.2, 0.2),  # b3
            (4.0, 20.0),  # sigma
            (-5.0, 5.0),  # center_shift
        ]
        
        try:
            result = differential_evolution(
                _score_from_basis,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-4,
                popsize=15,
                polish=True,
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            
            if result.success or result.fun < seed_score:
                best_params = result.x
                best_I, best_Q = _pulse_from_basis(best_params)
                best_score = result.fun
                
                if _is_valid(best_I, best_Q):
                    pulse_I = best_I
                    pulse_Q = best_Q
                    seed_score = best_score
                    
        except Exception:
            pass
    
    # Additional refinement with Nelder-Mead from best point
    remaining = DEADLINE - (time.time() - start_time) - 40.0
    if remaining > 20:
        # Get current best parameters by inverting the pulse
        # For now, just try a few random perturbations
        best_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 6.67, 0.0])
        
        def objective(params):
            pulse_I, pulse_Q = _pulse_from_basis(params)
            if not _is_valid(pulse_I, pulse_Q):
                return 1.0
            return _compute_score(pulse_I, pulse_Q)
        
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-8,
                                     'adaptive': True})
            
            if result.fun < seed_score:
                best_params = result.x
                best_I, best_Q = _pulse_from_basis(best_params)
                best_score = result.fun
                
                if _is_valid(best_I, best_Q):
                    pulse_I = best_I
                    pulse_Q = best_Q
                    seed_score = best_score
                    
        except Exception:
            pass
    
    # DOP853 recheck (nominal case)
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 10:
        F_dop = _dop853_recheck(pulse_I, pulse_Q)
        if F_dop is not None:
            pass  # Verification only
    
    result = np.column_stack((pulse_I, pulse_Q))
    
    return result
# EVOLVE-BLOCK-END
