"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm
import warnings

warnings.filterwarnings('ignore')

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_GATE = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = 2 * np.pi * 0.002
T1 = 25000.0  # ns
T2 = 40000.0  # ns

# 3-level system: |0>, |1>, |2>
# a = diag(sqrt([1, 2]), 1) in the sense that a|n> = sqrt(n)|n-1>
# So a is a 3x3 matrix with sqrt(1) at (0,1) and sqrt(2) at (1,2)
a_mat = np.zeros((3, 3), dtype=complex)
a_mat[0, 1] = 1.0       # sqrt(1)
a_mat[1, 2] = np.sqrt(2)  # sqrt(2)
a_dag = a_mat.conj().T

# n = a^dag a
n_mat = a_dag @ a_mat
# n_mat should be diag(0, 1, 2)

def generate_pulse(params):
    """Generate pulse from parameters: [amp, sigma, beta, phi]
    DRAG pulse: I(t) = amp * exp(-t^2/(2*sigma^2)) * cos(phi)
                Q(t) = amp * beta * d/dt[exp(-t^2/(2*sigma^2))]
    Offset to ensure zero at edges.
    """
    amp, sigma, beta, phi = params
    
    t = np.arange(N_STEPS) * DT - T_GATE / 2.0
    
    # Gaussian envelope
    gauss = np.exp(-t**2 / (2.0 * sigma**2))
    
    # Offset so that edges are zero
    edge_val = gauss[0]
    gauss = gauss - edge_val
    
    # Ensure non-negative and scale to have area pi
    gauss = np.maximum(gauss, 0)
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    # Derivative of Gaussian
    dgauss = -t / (sigma**2) * gauss
    
    # I and Q components
    I = amp * gauss * np.cos(phi)
    Q = amp * beta * dgauss
    
    # Apply amplitude bound
    magnitude = np.sqrt(I**2 + Q**2)
    mask = magnitude > AMPLITUDE_BOUND
    if np.any(mask):
        scale = AMPLITUDE_BOUND / magnitude[mask]
        I[mask] *= scale
        Q[mask] *= scale
    
    # Apply slew bound
    for i in range(1, N_STEPS):
        dI = I[i] - I[i-1]
        dQ = Q[i] - Q[i-1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            I[i] = I[i-1] + dI * scale
            Q[i] = Q[i-1] + dQ * scale
    
    # Check slew from zero to first step
    slew0 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew0 > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.sqrt(I[0]**2 + Q[0]**2)
        I[0] *= scale
        Q[0] *= scale
    
    # Check slew from last step to zero
    sleven = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if sleven > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.sqrt(I[-1]**2 + Q[-1]**2)
        I[-1] *= scale
        Q[-1] *= scale
    
    return np.column_stack([I, Q])


def simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the process matrix (or state evolution).
    Returns fidelity for each initial state.
    """
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Build Liouvillian for each step
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    
    a_adag = a_mat + a_dag
    a_minus_adag = a_mat - a_dag
    
    # n*(n-1)
    n_nm1 = n_mat @ (n_mat - np.eye(3))
    
    # Lindblad operators
    L1 = a_mat / np.sqrt(T1)
    L2 = n_mat * np.sqrt(2.0 / T2)
    
    # Precompute Lindblad superoperator parts
    # For Liouvillian: L(rho) = -i[H,rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    
    def liouvillian_step(I_val, Q_val):
        H = (detuning * n_mat + 
             0.5 * ANHARMONICITY * n_nm1 + 
             0.5 * scale * (I_val * a_adag + 1j * Q_val * a_minus_adag))
        
        # Build Liouvillian as 9x9 matrix
        # rho is 3x3, flatten to 9 vector
        # L(rho) = -i(H rho - rho H) + L1 rho L1^dag - 0.5(L1^dag L1 rho + rho L1^dag L1) + L2 rho L2^dag - 0.5(L2^dag L2 rho + rho L2^dag L2)
        
        # Use vectorization: vec(A rho B) = (B^T ⊗ A) vec(rho)
        # -i(H rho - rho H) = -i((I ⊗ H) - (H^T ⊗ I)) vec(rho) ... actually:
        # vec(H rho) = (I ⊗ H) vec(rho)
        # vec(rho H) = (H^T ⊗ I) vec(rho)
        
        L_mat = np.zeros((9, 9), dtype=complex)
        
        # Coherent part
        L_mat += -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
        
        # Dissipative part for L1
        L1_dag = L1.conj().T
        L1_dag_L1 = L1_dag @ L1
        L_mat += np.kron(L1, L1_dag) - 0.5 * np.kron(np.eye(3), L1_dag_L1) - 0.5 * np.kron(L1_dag_L1.T, np.eye(3))
        
        # Dissipative part for L2
        L2_dag = L2.conj().T
        L2_dag_L2 = L2_dag @ L2
        L_mat += np.kron(L2, L2_dag) - 0.5 * np.kron(np.eye(3), L2_dag_L2) - 0.5 * np.kron(L2_dag_L2.T, np.eye(3))
        
        return L_mat
    
    # Simulate all 8 steps
    # Start with identity superoperator (track how each basis state evolves)
    # Actually, let's track the evolution of the density matrix for each initial state
    
    # Better: compute the process matrix by evolving basis states
    # For fidelity calculation, we need U such that rho_f = U rho_0 U^dag (approximately, ignoring dissipation for now)
    # But with dissipation, we need the full superoperator
    
    # Let's compute the full 9x9 evolution operator
    U_super = np.eye(9, dtype=complex)
    
    for i in range(N_STEPS):
        L_step = liouvillian_step(I[i], Q[i])
        # Exponentiate: exp(L * dt)
        E = expm(L_step * DT)
        U_super = E @ U_super
    
    # Now compute fidelity for each initial state
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    
    def state_to_vec(state):
        return state.flatten()
    
    def vec_to_state(vec):
        return vec.reshape(3, 3)
    
    # Target: X gate on the 2-level subspace {|0>, |1>}
    # X|0> = |1>, X|1> = |0>
    # So target states:
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2), so fidelity uses |<target|final>|^2
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2) ... let me just compute properly
    
    # Define initial and target states
    s0 = np.array([1, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0], dtype=complex)
    
    states_init = [
        s0,
        s1,
        (s0 + s1) / np.sqrt(2),
        (s0 - s1) / np.sqrt(2),
        (s0 + 1j * s1) / np.sqrt(2),
        (s0 - 1j * s1) / np.sqrt(2),
    ]
    
    states_target = [
        s1,
        s0,
        (s1 + s0) / np.sqrt(2),
        (s1 - s0) / np.sqrt(2),
        (s1 + 1j * s0) / np.sqrt(2),
        (s1 - 1j * s0) / np.sqrt(2),
    ]
    
    fidelities = []
    for psi0, psi_t in zip(states_init, states_target):
        rho0 = np.outer(psi0, psi0.conj())
        rho0_vec = rho0.flatten()
        rho_f_vec = U_super @ rho0_vec
        rho_f = rho_f_vec.reshape(3, 3)
        
        # Fidelity = |<psi_t|rho_f|psi_t>|
        fid = np.abs(np.vdot(psi_t, rho_f @ psi_t))
        # Clip to [0, 1]
        fid = np.clip(fid, 0, 1)
        fidelities.append(fid)
    
    return fidelities


def score_params(params):
    """Compute the score (mean + max infidelity over 9 cases)."""
    pulse = generate_pulse(params)
    
    detunings = [-DETUNING_BOUNDS, 0.0, DETUNING_BOUNDS]
    scales = [0.97, 1.0, 1.03]
    
    all_infidelities = []
    for det in detunings:
        for scl in scales:
            fids = simulate_pulse(pulse, det, scl)
            infid = 1.0 - np.mean(fids)
            all_infidelities.append(infid)
    
    all_infidelities = np.array(all_infidelities)
    score = np.mean(all_infidelities) + np.max(all_infidelities)
    return score


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit, keep safety margin
    
    # Initial guess: DRAG pulse parameters
    # amp ~ 0.3, sigma ~ 40/6 ~ 6.67 ns, beta ~ 0.5, phi ~ 0
    x0 = np.array([0.3, 6.67, 0.5, 0.0])
    
    best_params = x0.copy()
    best_score = score_params(x0)
    
    # Try a few random starts
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline - 60.0:
            break
        
        if trial == 0:
            x_try = x0
        else:
            x_try = np.array([
                np.random.uniform(0.15, 0.38),
                np.random.uniform(4.0, 10.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-0.3, 0.3),
            ])
        
        try:
            result = minimize(
                score_params,
                x_try,
                method='L-BFGS-B',
                bounds=[(0.1, 0.39), (3.0, 12.0), (-2.0, 2.0), (-0.5, 0.5)],
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Generate final pulse
    final_pulse = generate_pulse(best_params)
    
    # Validate
    for i in range(N_STEPS):
        mag = np.sqrt(final_pulse[i, 0]**2 + final_pulse[i, 1]**2)
        if mag > AMPLITUDE_BOUND + 1e-9:
            # Scale down
            scale = AMPLITUDE_BOUND / mag
            final_pulse[i] *= scale
    
    # Re-check slew
    for i in range(1, N_STEPS):
        dI = final_pulse[i, 0] - final_pulse[i-1, 0]
        dQ = final_pulse[i, 1] - final_pulse[i-1, 1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-9:
            scale = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            final_pulse[i, 0] = final_pulse[i-1, 0] + dI * scale
            final_pulse[i, 1] = final_pulse[i-1, 1] + dQ * scale
    
    # Check edge slews
    slew0 = np.sqrt(final_pulse[0, 0]**2 + final_pulse[0, 1]**2) / DT
    if slew0 > SLEW_BOUND + 1e-9:
        scale = SLEW_BOUND * DT / np.sqrt(final_pulse[0, 0]**2 + final_pulse[0, 1]**2)
        final_pulse[0] *= scale
    
    sleven = np.sqrt(final_pulse[-1, 0]**2 + final_pulse[-1, 1]**2) / DT
    if sleven > SLEW_BOUND + 1e-9:
        scale = SLEW_BOUND * DT / np.sqrt(final_pulse[-1, 0]**2 + final_pulse[-1, 1]**2)
        final_pulse[-1] *= scale
    
    return final_pulse

# EVOLVE-BLOCK-END
