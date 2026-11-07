"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp

# Constants
DT = 0.5
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns
ANHARMONICITY = -2 * np.pi * 0.25
T1 = 25000.0
T_PHI = 40000.0
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
SCALE_BOUNDS = [0.97, 1.03]

# Hardware model parameters
HARDWARE_MODEL = {
    'anharmonicity': ANHARMONICITY,
    'amplitude_bound': AMPLITUDE_BOUND,
    'slew_bound': SLEW_BOUND,
    'detuning_bounds': DETUNING_BOUNDS,
    'drive_scale_bounds': SCALE_BOUNDS,
    'T1': T1,
    'Tphi': T_PHI
}

# Time grid
t_grid = np.linspace(0, T_TOTAL, N_STEPS + 1)
t_mid = 0.5 * (t_grid[:-1] + t_grid[1:])  # midpoints for constant steps


def make_lindblad_ops():
    """Create Lindblad operators for the 3-level system."""
    # a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.array([
        [0, np.sqrt(1), 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    # n = a^dag a
    n = a.conj().T @ a
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2 / T_PHI)
    
    return a, n, L1, L2


A_OP, N_OP, L1, L2 = make_lindblad_ops()


def liouvillian(detuning, scale, I, Q):
    """Construct the Liouvillian superoperator."""
    # Hamiltonian in the rotating frame
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    
    a_dag = A_OP.conj().T
    n_minus_1 = N_OP - np.eye(3)
    
    H = (detuning * N_OP + 
         0.5 * ANHARMONICITY * N_OP @ n_minus_1 + 
         0.5 * scale * (I * (A_OP + a_dag) + 1j * Q * (A_OP - a_dag)))
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    def L(rho):
        vec = rho.reshape(9)
        result = -1j * (H @ rho - rho @ H).reshape(9)
        
        # Lindblad terms
        for L_op in [L1, L2]:
            L_dag = L_op.conj().T
            result += (L_op @ rho @ L_dag - 0.5 * (L_dag @ L_op @ rho + rho @ L_dag @ L_op)).reshape(9)
        
        return result
    
    # Precompute the superoperator matrix
    L_mat = np.zeros((9, 9), dtype=complex)
    for j in range(9):
        e_j = np.zeros(9, dtype=complex)
        e_j[j] = 1
        L_mat[:, j] = L(e_j.reshape(3, 3))
    
    return L_mat


def simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    # Initial state: rho(0) = |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1
    
    # Precompute Liouvillian for each step
    L_mats = []
    for i in range(N_STEPS):
        L_mats.append(liouvillian(detuning, scale, I_vals[i], Q_vals[i]))
    
    # Integrate step by step
    for i in range(N_STEPS):
        L_mat = L_mats[i]
        
        def ode(t, y):
            return L_mat @ y
        
        y0 = rho.reshape(9)
        sol = solve_ivp(ode, [0, DT], y0, method='DOP853', rtol=1e-10, atol=1e-12)
        rho = sol.y[:, -1].reshape(3, 3)
    
    # Renormalize
    tr = np.trace(rho).real
    if abs(tr) > 1e-10:
        rho = rho / tr
    
    return rho


def compute_fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure state."""
    # Fidelity for pure state: F = <psi|rho|psi>
    psi = target_state.reshape(-1)
    F = np.real(np.conj(psi) @ rho @ psi)
    return max(0, min(1, F))


def compute_infidelity(pulse, detuning, scale):
    """Compute mean infidelity over the 5 test states."""
    # Test states
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 0, 0]) + np.array([0, 1, 0])) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        (np.array([1, 0, 0]) - np.array([0, 1, 0])) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 0, 0]) + 1j * np.array([0, 1, 0])) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, 0, 0]) - 1j * np.array([0, 1, 0])) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    rho = simulate_pulse(pulse, detuning, scale)
    
    fidelities = []
    for state in states:
        F = compute_fidelity(rho, state)
        fidelities.append(F)
    
    mean_F = np.mean(fidelities)
    return 1 - mean_F


def compute_score(pulse):
    """Compute the score over all 9 cases."""
    detunings = [DETUNING_BOUNDS[0], 0, DETUNING_BOUNDS[1]]
    scales = [SCALE_BOUNDS[0], 1.0, SCALE_BOUNDS[1]]
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            inf = compute_infidelity(pulse, det, scale)
            infidelities.append(inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf


def check_validity(pulse):
    """Check if pulse satisfies constraints."""
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    # Amplitude bound
    amplitudes = np.sqrt(I_vals**2 + Q_vals**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND + 1e-9:
        return False
    
    # Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero to step 1
    diff = np.sqrt(I_vals[0]**2 + Q_vals[0]**2)
    if diff / DT > SLEW_BOUND + 1e-9:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        dI = I_vals[i+1] - I_vals[i]
        dQ = Q_vals[i+1] - Q_vals[i]
        diff = np.sqrt(dI**2 + dQ**2)
        if diff / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    diff = np.sqrt(I_vals[-1]**2 + Q_vals[-1]**2)
    if diff / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def generate_seed_pulse():
    """Generate DRAG-style Gaussian seed pulse."""
    # Gaussian with sigma = 40/6 ns
    sigma = T_TOTAL / 6
    t = t_mid
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - T_TOTAL/2) / sigma)**2)
    
    # Offset to 0 at edges
    gauss = gauss - gauss[0]
    gauss = gauss - gauss[-1]
    
    # Normalize area to pi
    area = np.sum(gauss) * DT
    if abs(area) > 1e-10:
        gauss = gauss * (np.pi / area)
    
    # DRAG term: proportional to derivative of envelope
    # dI/dt approximated by finite difference
    dI_dt = np.zeros_like(gauss)
    dI_dt[1:-1] = (gauss[2:] - gauss[:-2]) / (2 * DT)
    dI_dt[0] = (gauss[1] - gauss[0]) / DT
    dI_dt[-1] = (gauss[-1] - gauss[-2]) / DT
    
    # DRAG coefficient (typically beta ~ anharmonicity/omega_d, but we'll use a reasonable value)
    beta = 0.1  # Will be optimized
    
    Q_vals = beta * dI_dt
    I_vals = gauss
    
    pulse = np.column_stack([I_vals, Q_vals])
    
    # Clip to satisfy amplitude bound
    amplitudes = np.sqrt(I_vals**2 + Q_vals**2)
    mask = amplitudes > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amplitudes[mask]
        I_vals[mask] *= scale_factor
        Q_vals[mask] *= scale_factor
    
    pulse = np.column_stack([I_vals, Q_vals])
    
    return pulse


def optimize_pulse(pulse, max_iterations=100, deadline=None):
    """Optimize the pulse using gradient-free method."""
    best_pulse = pulse.copy()
    best_score = compute_score(best_pulse)
    
    start_time = time.time()
    
    # Simple coordinate descent / Nelder-Mead style optimization
    # We'll use a simple iterative improvement approach
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        # Try small perturbations
        improved = False
        for i in range(N_STEPS):
            for coord in [0, 1]:
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    new_pulse = best_pulse.copy()
                    new_pulse[i, coord] += delta
                    
                    if not check_validity(new_pulse):
                        continue
                    
                    score = compute_score(new_pulse)
                    if score < best_score - 1e-8:
                        best_pulse = new_pulse
                        best_score = score
                        improved = True
        
        if not improved:
            break
    
    return best_pulse, best_score


def pulse(target_gate, hardware_model):
    """Main entry point: generate optimized pulse for the given target gate."""
    if target_gate != 'X_pi':
        # Return a zero pulse for invalid gates
        return np.zeros((80, 2))
    
    # Set deadline
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds total
    
    # Generate seed pulse
    seed_pulse = generate_seed_pulse()
    
    # Optimize
    optimized_pulse, score = optimize_pulse(seed_pulse, max_iterations=50, deadline=deadline)
    
    # Ensure validity
    if not check_validity(optimized_pulse):
        optimized_pulse = seed_pulse
    
    return optimized_pulse

# EVOLVE-BLOCK-END
