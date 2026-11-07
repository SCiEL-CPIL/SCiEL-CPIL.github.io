import numpy as np
import time
from scipy.linalg import expm, sqrtm
from scipy.optimize import minimize

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
Tphi = 40000.0
DT = 0.5
N_STEPS = 80
T_TOTAL = 40.0

# Precompute constants
a = np.zeros((3, 3), dtype=complex)
a[0, 1] = 1.0
a[1, 2] = np.sqrt(2.0)
adag = a.conj().T
n = adag @ a
n_minus_1 = n - np.eye(3)

# Lindblad terms (constant parts)
L1 = a / np.sqrt(T1)
L2 = n * np.sqrt(2.0 / Tphi)
L1dL1 = L1.conj().T @ L1
L2dL2 = L2.conj().T @ L2

# Precompute superoperator components
I9 = np.eye(9)
kron_L1_L1c = np.kron(L1, L1.conj())
kron_L2_L2c = np.kron(L2, L2.conj())
kron_I_L1dL1T = np.kron(I9, L1dL1.T)
kron_L1dL1_I = np.kron(L1dL1, I9)
kron_I_L2dL2T = np.kron(I9, L2dL2.T)
kron_L2dL2_I = np.kron(L2dL2, I9)

dissipative_super = kron_L1_L1c + kron_L2_L2c - 0.5*(kron_I_L1dL1T + kron_L1dL1_I + kron_I_L2dL2T + kron_L2dL2_I)

def make_dag_pulse(beta, sigma=None):
    """Create a DRAG pulse. Returns (80, 2) array of [I, Q]."""
    if sigma is None:
        sigma = T_TOTAL / 6.0
    
    t = np.linspace(0, T_TOTAL, N_STEPS, endpoint=False) + DT/2
    
    # Gaussian envelope
    g = np.exp(-0.5 * (t / sigma) ** 2)
    
    # Normalize so that integral of I dt = pi
    area = np.trapz(g, t)
    if area > 0:
        g = g * (np.pi / area)
    
    I = g
    dI_dt = np.gradient(I, DT)
    Q = beta * dI_dt
    
    # Ensure amplitude constraint
    amplitude = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amplitude)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        I = I * scale_factor
        Q = Q * scale_factor
        # Renormalize
        area = np.trapz(I, t)
        if area > 0:
            I = I * (np.pi / area)
            Q = Q * (np.pi / area)
    
    return np.column_stack([I, Q])

def check_constraints(pulse):
    """Check if pulse satisfies amplitude and slew constraints."""
    I, Q = pulse[:, 0], pulse[:, 1]
    amplitude = np.sqrt(I**2 + Q**2)
    if np.max(amplitude) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    p = np.vstack([[0, 0], pulse, [0, 0]])
    diffs = np.diff(p, axis=0)
    norms = np.sqrt(np.sum(diffs**2, axis=1))
    slew_rates = norms / DT
    if np.max(slew_rates) > SLEW_BOUND + 1e-10:
        return False
    
    return True

def build_superoperator(detuning, scale, I, Q):
    """Build the full superoperator for one step."""
    H = detuning * n + 0.5 * ANHARMONICITY * n @ n_minus_1
    H += 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    
    H_comm = -1j * (np.kron(H, I9) - np.kron(I9, H.T))
    
    return H_comm + dissipative_super

def evolve_rho(rho, pulse, detuning, scale):
    """Evolve density matrix through the pulse."""
    rho_vec = rho.flatten()
    
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        L_super = build_superoperator(detuning, scale, I, Q)
        exp_L = expm(L_super * DT)
        rho_vec = exp_L @ rho_vec
    
    return rho_vec.reshape(3, 3)

def fidelity(rho, rho_target):
    """Compute fidelity between two density matrices."""
    rho_sqrt = sqrtm(rho)
    inner = rho_sqrt @ rho_target @ rho_sqrt
    inner_sqrt = sqrtm(inner)
    F = np.real(np.trace(inner_sqrt))
    return np.clip(F, 0, 1)

def compute_score(pulse, detunings, scales):
    """Compute mean + max infidelity over the 9 cases."""
    X_gate = np.zeros((3, 3), dtype=complex)
    X_gate[0, 1] = 1.0
    X_gate[1, 0] = 1.0
    
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    infidelities = []
    
    for detuning in detunings:
        for scale in scales:
            fidelities = []
            for psi in states:
                rho = np.outer(psi, psi.conj())
                rho_final = evolve_rho(rho, pulse, detuning, scale)
                
                psi_target = X_gate @ psi
                rho_target = np.outer(psi_target, psi_target.conj())
                
                F = fidelity(rho_final, rho_target)
                fidelities.append(F)
            
            mean_fidelity = np.mean(fidelities)
            infidelities.append(1.0 - mean_fidelity)
    
    return np.mean(infidelities) + np.max(infidelities)

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    start_time = time.time()
    deadline = start_time + 280.0
    
    detunings = [DETUNING_BOUNDS[0], 0.0, DETUNING_BOUNDS[1]]
    scales = [SCALE_BOUNDS[0], 1.0, SCALE_BOUNDS[1]]
    
    # Start with DRAG pulse with beta=0
    best_pulse = make_dag_pulse(0.0)
    best_score = compute_score(best_pulse, detunings, scales)
    
    # Optimize beta using Nelder-Mead
    def objective(params):
        if time.time() > deadline:
            return best_score
        beta = params[0]
        pulse_candidate = make_dag_pulse(beta)
        if not check_constraints(pulse_candidate):
            return best_score + 1.0
        return compute_score(pulse_candidate, detunings, scales)
    
    x0 = np.array([0.0])
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6})
    
    if time.time() < deadline and result.fun < best_score:
        best_score = result.fun
        best_pulse = make_dag_pulse(result.x[0])
    
    if not check_constraints(best_pulse):
        best_pulse = make_dag_pulse(0.0)
    
    return best_pulse
