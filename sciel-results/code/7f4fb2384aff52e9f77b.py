"""Analytic DRAG pulse synthesis with Gaussian shaping and edge tapering."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
    """Compute score only."""
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
    
    # Enforce slew bound backward
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
    """Generate a DRAG pulse from parameters: [sigma, alpha, beta, gamma, delta]"""
    sigma, alpha, beta, gamma, delta = params
    t = np.arange(N_STEPS) * DT
    center = 20.0
    
    # Gaussian envelope
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((t[0] - center) / sigma)**2)
    g = g - edge_val
    
    # Normalize area to pi
    area = np.sum(g) * DT
    if area > 0:
        g = g * (np.pi / area)
    
    # DRAG term: derivative of envelope scaled by anharmonicity
    # The derivative of a Gaussian is proportional to (t-center)/sigma^2 * g
    drag = alpha * (t - center) / sigma**2 * g
    
    # Add additional shape terms for robustness
    # beta: overall amplitude scaling
    # gamma: detuning compensation term (linear in t)
    # delta: quadratic correction
    I = beta * g + gamma * (t - center) * g + delta * (t - center)**2 * g
    Q = drag
    
    pulse = np.zeros((N_STEPS, 2))
    pulse[:, 0] = I
    pulse[:, 1] = Q
    
    return pulse

def seed_params():
    """Generate seed parameters."""
    return np.array([40.0/6.0, 0.0, 0.0, 0.0, 0.0])

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Start with seed
    params = seed_params()
    current_pulse = generate_pulse_from_params(params)
    current_pulse = enforce_constraints(current_pulse)
    best_pulse = current_pulse.copy()
    best_score = score_only(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Coarse grid search over key parameters
    # sigma: [4, 5, 6, 7, 8]
    # alpha: [-0.5, -0.25, 0, 0.25, 0.5]
    # beta: [0.8, 0.9, 1.0, 1.1, 1.2]
    # gamma: [-0.1, -0.05, 0, 0.05, 0.1]
    # delta: [-0.01, -0.005, 0, 0.005, 0.01]
    
    sigma_vals = [4.0, 5.0, 6.0, 7.0, 8.0]
    alpha_vals = [-0.5, -0.25, 0.0, 0.25, 0.5]
    beta_vals = [0.8, 0.9, 1.0, 1.1, 1.2]
    gamma_vals = [-0.1, -0.05, 0.0, 0.05, 0.1]
    delta_vals = [-0.01, -0.005, 0.0, 0.005, 0.01]
    
    # Full grid is 5^5 = 3125 evaluations, each ~0.1s = 312s. Too many.
    # Do a smarter search: first vary sigma and alpha, then refine.
    
    # Phase 1: vary sigma and alpha (25 combos)
    for sigma in sigma_vals:
        for alpha in alpha_vals:
            if time.time() > deadline:
                break
            p = np.array([sigma, alpha, 1.0, 0.0, 0.0])
            pl = generate_pulse_from_params(p)
            pl = enforce_constraints(pl)
            s = score_only(pl)
            if s < best_score:
                best_score = s
                best_pulse = pl.copy()
    
    if time.time() > deadline:
        return best_pulse
    
    # Phase 2: refine beta and gamma around best
    best_sigma = best_pulse is not None
    # Extract best params from the grid search
    # Actually, let's just do a finer local search
    
    # Phase 2: local refinement with Nelder-Mead on 5 parameters
    from scipy.optimize import minimize
    
    def objective(params):
        p = generate_pulse_from_params(params)
        p = enforce_constraints(p)
        return score_only(p)
    
    # Find current best params by checking which grid point gave best score
    # For simplicity, restart from seed and do local optimization
    x0 = seed_params()
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 100,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True,
                'disp': False
            }
        )
        
        if result.fun < best_score:
            opt_pulse = generate_pulse_from_params(result.x)
            opt_pulse = enforce_constraints(opt_pulse)
            opt_score = score_only(opt_pulse)
            if opt_score < best_score:
                best_pulse = opt_pulse
                best_score = opt_score
                
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_pulse
    
    # Phase 3: try a few more starting points for Nelder-Mead
    for start_params in [
        [5.0, 0.3, 1.0, 0.0, 0.0],
        [6.0, -0.3, 1.0, 0.0, 0.0],
        [5.0, 0.0, 1.1, 0.05, 0.0],
        [6.0, 0.2, 0.9, -0.05, 0.0],
    ]:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                np.array(start_params),
                method='Nelder-Mead',
                options={
                    'maxiter': 50,
                    'xatol': 1e-4,
                    'fatol': 1e-6,
                    'adaptive': True,
                    'disp': False
                }
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
    
    return best_pulse
# EVOLVE-BLOCK-END
