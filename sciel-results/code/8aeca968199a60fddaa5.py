"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T2 = 40000.0

N_STEPS = 80
DT = 0.5

# Build ladder operators for 3-level system
def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

def hamiltonian(detuning, scale, I, Q):
    """Build the Hamiltonian for given parameters."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    H += 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    return H

def liouvillian_step(rho, H):
    """Evolve density matrix through one Liouvillian step (matrix exponential)."""
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    
    # Build the Liouvillian superoperator
    dim = 3
    L = np.zeros((dim * dim, dim * dim), dtype=complex)
    
    # Coherent evolution: -i[H, rho]
    for i in range(dim):
        for j in range(dim):
            # H rho term
            for k in range(dim):
                L[i * dim + j, k * dim + i] -= 1j * H[i, k]
            # rho H term
            for k in range(dim):
                L[i * dim + j, k * dim + j] += 1j * H[k, j]
    
    # Dissipative terms
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for m in range(dim):
                        # L rho L^dagger term
                        L[i * dim + j, k * dim + m] += L_op[i, k] * L_dag[m, j]
                        # 1/2 {L^dagger L, rho} terms
                        L[i * dim + j, k * dim + m] -= 0.5 * (L_dag @ L_op)[i, k] * (np.eye(dim)[j, m] + np.eye(dim)[m, j])
    
    # Exponentiate the Liouvillian
    from scipy.linalg import expm
    U = expm(L * DT)
    rho_vec = rho.flatten()
    rho_new_vec = U @ rho_vec
    return rho_new_vec.reshape(dim, dim)

def evolve_pulse(pulse, detuning, scale):
    """Evolve the density matrix through the full pulse."""
    dim = 3
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        H = hamiltonian(detuning, scale, I, Q)
        rho = liouvillian_step(rho, H)
    
    return rho

def fidelity_with_X(rho):
    """Compute fidelity with X gate applied to |0>."""
    # X gate: |0> -> |1>, |1> -> |0>
    # Fidelity = Tr(rho * rho_X) where rho_X is the target state
    # For |0> -> |1>: target is |1><1|
    target = np.zeros((3, 3), dtype=complex)
    target[1, 1] = 1.0
    return np.real(np.trace(rho @ target))

def compute_score(pulse):
    """Compute the 9-case infidelity score."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    fidelities = []
    for det in detunings:
        for scale in scales:
            # Test on 4 initial states
            states = [
                np.array([1, 0, 0], dtype=complex),  # |0>
                np.array([0, 1, 0], dtype=complex),  # |1>
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
                (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
                (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
                (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
            ]
            
            case_fidelities = []
            for state in states:
                rho_init = np.outer(state, state.conj())
                
                dim = 3
                rho = rho_init.copy()
                for i in range(N_STEPS):
                    I = pulse[i, 0]
                    Q = pulse[i, 1]
                    H = hamiltonian(det, scale, I, Q)
                    rho = liouvillian_step(rho, H)
                
                # Target: X applied to state
                # X: |0><0| -> |1><1|, |1><1| -> |0><0|
                # For general state, X|psi> = |psi> with 0 and 1 swapped
                target_state = np.zeros(3, dtype=complex)
                target_state[0] = state[1]
                target_state[1] = state[0]
                target_state[2] = state[2]
                target_rho = np.outer(target_state, target_state.conj())
                
                fid = np.real(np.trace(rho @ target_rho))
                case_fidelities.append(fid)
            
            mean_fid = np.mean(case_fidelities)
            fidelities.append(mean_fid)
    
    infidelities = [1.0 - f for f in fidelities]
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def check_validity(pulse):
    """Check if the pulse satisfies all constraints."""
    # Amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND + 1e-10:
            return False
    
    # Slew bound
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-10:
            return False
        prev = curr
    # From last step to zero
    if np.linalg.norm(prev - np.array([0.0, 0.0])) / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True

def seed_pulse():
    """DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, N_STEPS * DT, N_STEPS)
    sigma = 40.0 / 6.0
    center = N_STEPS * DT / 2.0
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset to make edges zero
    edge_offset = env[0]
    env = env - edge_offset
    env = np.maximum(env, 0)
    
    # Scale to area pi
    area = np.sum(env) * DT
    if area > 0:
        env = env * (np.pi / area)
    
    # DRAG: add derivative term
    denv = np.gradient(env, DT)
    drag_coeff = 0.1  # Small DRAG coefficient
    
    I = env
    Q = drag_coeff * denv
    
    pulse = np.column_stack([I, Q])
    
    # Ensure validity
    if not check_validity(pulse):
        # Scale down
        max_amp = np.max(np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            pulse *= AMPLITUDE_BOUND / max_amp
        
        # Check slew
        for i in range(1, N_STEPS):
            delta = np.linalg.norm(pulse[i] - pulse[i-1]) / DT
            if delta > SLEW_BOUND:
                scale = SLEW_BOUND / delta
                pulse[i:] *= scale
    
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin before 330s kill
    
    # Start with seed
    best_pulse = seed_pulse()
    best_score = compute_score(best_pulse)
    
    if time.time() > deadline - 60:
        return best_pulse
    
    # Try to improve with optimization
    def objective(params):
        pulse_arr = params.reshape(N_STEPS, 2)
        if not check_validity(pulse_arr):
            return 1e10
        return compute_score(pulse_arr)
    
    # Use Nelder-Mead for derivative-free optimization
    x0 = best_pulse.flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        
        if result.fun < best_score:
            candidate = result.x.reshape(N_STEPS, 2)
            if check_validity(candidate):
                best_pulse = candidate
                best_score = result.fun
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_pulse
    
    # Try a few random perturbations
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        
        perturbation = np.random.randn(N_STEPS, 2) * 0.01
        candidate = best_pulse + perturbation
        
        # Clip to amplitude bound
        for i in range(N_STEPS):
            amp = np.sqrt(candidate[i, 0]**2 + candidate[i, 1]**2)
            if amp > AMPLITUDE_BOUND:
                candidate[i] *= AMPLITUDE_BOUND / amp
        
        if check_validity(candidate):
            score = compute_score(candidate)
            if score < best_score:
                best_pulse = candidate
                best_score = score
    
    return best_pulse
# EVOLVE-BLOCK-END
