"""
Low-dimensional basis expansion pulse optimization for 3-level transmon X-pi gate.
Uses Differential Evolution for global search over 8 parameters,
then L-BFGS-B for local refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import differential_evolution, minimize

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

# Target operator X
X3 = np.array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 1]], dtype=complex)

# Test states
STATES = [
    np.array([1, 0, 0], dtype=complex),
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
    np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
    np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
    np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
]

# Precompute static Liouvillian parts for each detuning/scale case
# L = L_static + 0.5 * scale * (I * K_I + Q * K_Q)
# H_static = detuning * N + 0.5 * Anhar * N(N-1)
# L_static = -1j * (kron(H_static, I) - kron(I, H_static.T)) + DISS_L
# K_I = -1j * (kron(X_op, I) - kron(I, X_op.T))
# K_Q = -1j * (kron(Y_op, I) - kron(I, Y_op.T))

X_OP = A_MAT + AD_MAT
Y_OP = A_MAT - AD_MAT
K_I = -1j * (np.kron(X_OP, I3) - np.kron(I3, X_OP.T))
K_Q = -1j * (np.kron(Y_OP, I3) - np.kron(I3, Y_OP.T))

CASES = []
for detuning in DETUNING_VALUES:
    for scale in SCALE_VALUES:
        H_static = detuning * N_MAT + 0.5 * ANHARMONICITY * (N_MAT @ (N_MAT - I3))
        L_static = -1j * (np.kron(H_static, I3) - np.kron(I3, H_static.T)) + DISS_L
        CASES.append((detuning, scale, L_static))

def build_liouvillian_fast(L_static, scale, I, Q):
    return L_static + 0.5 * scale * (I * K_I + Q * K_Q)

def check_validity(pulse):
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
    pulse = pulse.copy()
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
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
    
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
            
    return pulse

def generate_pulse_from_params(params):
    """
    params: [amp, sigma, center, drag_coeff, phase, taper_left, taper_right, offset]
    Generates pulse using a Gaussian-DRAG basis.
    """
    t = np.arange(N_STEPS) * DT
    amp = params[0]
    sigma = params[1]
    center = params[2]
    drag_coeff = params[3]
    phase = params[4]
    taper_left = params[5]
    taper_right = params[6]
    offset = params[7]
    
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    dg = (t - center) / (sigma**2) * g
    
    # Tapering to ensure edges are close to 0
    edge_width = 5.0
    taper_mask_left = (t < edge_width)
    taper_mask_right = (t > GATE_TIME - edge_width)
    
    g[taper_mask_left] *= (t[taper_mask_left] / edge_width) ** taper_left
    g[taper_mask_right] *= ((GATE_TIME - t[taper_mask_right]) / edge_width) ** taper_right
    
    # Apply phase rotation
    I = amp * g * np.cos(phase)
    Q = amp * (g * np.sin(phase) + drag_coeff * dg)
    
    pulse = np.column_stack((I, Q))
    
    return enforce_constraints(pulse)

def compute_score(pulse):
    """
    Computes the score (mean + max infidelity) for a given pulse.
    Uses the precomputed static Liouvillians for speed.
    """
    case_infideltas = []
    
    for detuning, scale, L_static in CASES:
        exp_mats = []
        for i in range(N_STEPS):
            I_val, Q_val = pulse[i]
            L = build_liouvillian_fast(L_static, scale, I_val, Q_val)
            exp_mats.append(sla.expm(L * DT))
            
        infidelities = []
        for psi0 in STATES:
            rho0 = np.outer(psi0, psi0.conj())
            rho_vec = rho0.reshape(-1).copy()
            
            for i in range(N_STEPS):
                rho_vec = exp_mats[i] @ rho_vec
            
            rho = rho_vec.reshape(3, 3)
            psi_target = X3 @ psi0
            f = np.vdot(psi_target, rho @ psi_target).real
            f = np.clip(f, 0, 1)
            infidelities.append(1.0 - f)
            
        mean_inf = np.mean(infidelities)
        case_infideltas.append(mean_inf)
        
    mean_inf = np.mean(case_infideltas)
    max_inf = np.max(case_infideltas)
    score = mean_inf + max_inf
    return score

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Initial guess
    x0 = np.array([0.2, 5.0, 20.0, -0.1, 0.0, 1.0, 1.0, 0.0])
    
    bounds = [
        (0.05, 0.35),
        (3.0, 8.0),
        (15.0, 25.0),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (0.5, 3.0),
        (0.5, 3.0),
        (-0.1, 0.1)
    ]
    
    best_params = x0.copy()
    best_pulse = generate_pulse_from_params(best_params)
    best_score = compute_score(best_pulse)
    
    def objective(params):
        if time.time() > deadline:
            return best_score
        p = generate_pulse_from_params(params)
        s = compute_score(p)
        return s
        
    # Run DE
    try:
        result_de = differential_evolution(
            objective, bounds, 
            seed=42, 
            maxiter=30, 
            popsize=15,
            tol=1e-5,
            polish=False
        )
        if result_de.fun < best_score:
            best_params = result_de.x
            best_pulse = generate_pulse_from_params(best_params)
            best_score = compute_score(best_pulse)
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if result_nm.fun < best_score:
                best_params = result_nm.x
                best_pulse = generate_pulse_from_params(best_params)
                best_score = compute_score(best_pulse)
        except Exception:
            pass
            
    return best_pulse
# EVOLVE-BLOCK-END
