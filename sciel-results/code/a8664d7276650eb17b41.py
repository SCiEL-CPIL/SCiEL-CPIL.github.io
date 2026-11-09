"""Parameterized Gaussian DRAG pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import differential_evolution

# --- Constants from task ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
DETUNING_VALUES = np.array([-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002])
SCALE_VALUES = np.array([0.97, 1.0, 1.03])
AMP_BOUND = 0.4
SLEW_BOUND = 0.12

# Precompute constant matrices
A_MAT = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
AD_MAT = A_MAT.conj().T
N_MAT = AD_MAT @ A_MAT
I3 = np.eye(3, dtype=complex)

# Lindblad operators
G1 = A_MAT / np.sqrt(T1)
G2 = N_MAT * np.sqrt(2.0 / Tphi)
G1DAG = G1.conj().T
G2DAG = G2.conj().T
G1DAG_G1 = G1DAG @ G1
G2DAG_G2 = G2DAG @ G2

# Precompute Liouvillian terms that don't depend on H
DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))

def build_liouvillian_full(H):
    """Build full Liouvillian for given Hamiltonian."""
    coh_L = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    return coh_L + DISS_L

def build_hamiltonian(detuning, scale, I, Q):
    """Build the 3-level Hamiltonian matrix."""
    NN1 = N_MAT @ (N_MAT - I3)
    H = (detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 +
         0.5 * scale * (I * (A_MAT + AD_MAT) + 1j * Q * (A_MAT - AD_MAT)))
    return H

def evolve_all_states(pulse, detuning, scale):
    """Evolve all 6 test states and return mean fidelity."""
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    X3 = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 1]], dtype=complex)
    
    # Precompute exp(L*dt) for each step
    exp_mats = []
    for i in range(N_STEPS):
        I_val, Q_val = pulse[i]
        H = build_hamiltonian(detuning, scale, I_val, Q_val)
        L = build_liouvillian_full(H)
        exp_mats.append(sla.expm(L * DT))
    
    fidelities = []
    for psi0 in states:
        rho0 = np.outer(psi0, psi0.conj())
        rho_vec = rho0.reshape(-1).copy()
        
        for i in range(N_STEPS):
            rho_vec = exp_mats[i] @ rho_vec
        
        rho = rho_vec.reshape(3, 3)
        psi_target = X3 @ psi0
        f = np.vdot(psi_target, rho @ psi_target).real
        f = np.clip(f, 0, 1)
        fidelities.append(f)
    
    return np.mean(fidelities)

def score_only(pulse):
    """Compute score only (no gradient)."""
    case_fidelities = []
    for detuning in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            f = evolve_all_states(pulse, detuning, scale)
            case_fidelities.append(f)
    
    case_fidelities = np.array(case_fidelities)
    case_infidelities = 1.0 - case_fidelities
    
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    score = mean_inf + max_inf
    
    return score

def check_validity(pulse):
    """Check amplitude and slew constraints."""
    for i in range(len(pulse)):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND + 1e-9:
            return False
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse)):
        curr = pulse[i]
        diff = np.linalg.norm(curr - prev) / DT
        if diff > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    diff = np.linalg.norm(prev - np.array([0.0, 0.0])) / DT
    if diff > SLEW_BOUND + 1e-9:
        return False
    return True

def enforce_constraints(pulse):
    """Project pulse onto feasible set."""
    pulse = pulse.copy()
    
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    # Enforce slew bound forward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Enforce slew bound backward (from last to zero)
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS - 1, -1, -1):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Re-enforce amplitude after slew projection
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    return pulse

def generate_pulse_from_params(params):
    """
    Generate pulse from parameters:
    params = [amp, sigma, drag, center, phase]
    amp: peak amplitude (rad/ns)
    sigma: Gaussian width (ns)
    drag: DRAG coefficient (relative to amp)
    center: time center (ns)
    phase: Q phase (radians)
    """
    amp = params[0]
    sigma = params[1]
    drag = params[2]
    center = params[3]
    phase = params[4]
    
    t = np.arange(N_STEPS) * DT
    # Gaussian envelope
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset to 0 at edges to satisfy slew constraint more easily
    edge_val = g[0]
    g = g - edge_val
    
    # Normalize area to pi for X_pi gate (approximate)
    area = np.sum(g) * DT
    if area > 0:
        g = g * (np.pi / area)
    
    # Apply DRAG: Q component proportional to derivative of I
    # d/dt g ~ (t-center)/sigma^2 * g
    dg = (t - center) / (sigma**2) * g
    # Normalize dg to have similar scale to g for DRAG coefficient
    if np.max(np.abs(dg)) > 0:
        dg = dg / np.max(np.abs(dg))
    
    I = amp * g
    Q = amp * drag * dg * np.sin(phase)
    
    pulse = np.column_stack((I, Q))
    return pulse

def seed_params():
    """Generate seed parameters."""
    # amp ~ 0.2, sigma ~ 5, drag ~ -0.1, center ~ 20, phase ~ 0
    return np.array([0.2, 5.0, -0.1, 20.0, 0.0])

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Start with seed
    p0 = seed_params()
    current_pulse = generate_pulse_from_params(p0)
    current_pulse = enforce_constraints(current_pulse)
    best_pulse = current_pulse.copy()
    best_score = score_only(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Define objective for DE
    def objective(params):
        p = generate_pulse_from_params(params)
        p = enforce_constraints(p)
        return score_only(p)
    
    # Bounds for parameters
    # amp: [0.1, 0.4]
    # sigma: [2, 15]
    # drag: [-0.5, 0.5]
    # center: [10, 30]
    # phase: [0, 2*pi]
    bounds = [
        (0.1, 0.4),
        (2.0, 15.0),
        (-0.5, 0.5),
        (10.0, 30.0),
        (0.0, 2 * np.pi)
    ]
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-4,
            atol=1e-6,
            polish=False, # Disable polish to save time, we'll do local refine
            init='latinhypercube'
        )
        
        if result.fun < best_score:
            opt_pulse = generate_pulse_from_params(result.x)
            opt_pulse = enforce_constraints(opt_pulse)
            opt_score = score_only(opt_pulse)
            if opt_score < best_score:
                best_pulse = opt_pulse
                best_score = opt_score
                
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pulse
    
    # Local refinement with Nelder-Mead on the parameter space
    x0 = result.x if 'result' in locals() and result.success else p0
    
    def objective_params(params):
        p = generate_pulse_from_params(params)
        p = enforce_constraints(p)
        return score_only(p)
        
    try:
        from scipy.optimize import minimize
        result_nm = minimize(
            objective_params,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
        )
        
        if result_nm.fun < best_score:
            opt_pulse = generate_pulse_from_params(result_nm.x)
            opt_pulse = enforce_constraints(opt_pulse)
            opt_score = score_only(opt_pulse)
            if opt_score < best_score:
                best_pulse = opt_pulse
                best_score = opt_score
                
    except Exception:
        pass
        
    return best_pulse
# EVOLVE-BLOCK-END
