"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# --- Constants ---
N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT  # 40 ns

# Hardware parameters from the task description
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
SCALE_MIN = 0.97
SCALE_MAX = 1.03
T1 = 25000.0
Tphi = 40000.0

# --- Quantum mechanics helpers ---

def _build_hamiltonian(detuning, scale, I, Q):
    """
    Build the 3x3 Hamiltonian for the transmon system.
    
    H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    
    where a = diag(sqrt([1, 2]), 1) in the |0>, |1>, |2> basis.
    """
    # Ladder operator a in the |0>, |1>, |2> basis
    # a|0> = 0, a|1> = sqrt(1)|0>, a|2> = sqrt(2)|1>
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    
    adag = a.conj().T
    
    # Number operator n = a† a
    n = adag @ a
    
    # Number operator n(n-1)
    nn_minus_1 = n @ (n - np.eye(3))
    
    # Drive term: 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    drive = 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    
    # Full Hamiltonian
    H = detuning * n + 0.5 * ANHARMONICITY * nn_minus_1 + drive
    
    return H

def _build_liouvillian(H):
    """
    Build the Liouvillian superoperator for the Lindblad equation.
    
    Lindblad operators:
    L1 = a / sqrt(T1)  (relaxation)
    L2 = n * sqrt(2 / Tphi)  (dephasing)
    
    The master equation is: dρ/dt = -i[H, ρ] + sum_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ})
    """
    # Ladder operator a
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    
    adag = a.conj().T
    n = adag @ a
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Build the Liouvillian as a 9x9 matrix acting on vec(ρ)
    # vec(ρ) is a 9-element vector
    # The Liouvillian L acts as: L[vec(ρ)] = vec(dρ/dt)
    
    # Term 1: -i[H, ρ] -> -i(H ⊗ I - I ⊗ H^T) in vec form
    # Actually, vec(AρB) = (B^T ⊗ A) vec(ρ)
    # So vec(Hρ) = (I ⊗ H) vec(ρ)
    # And vec(ρH) = (H^T ⊗ I) vec(ρ)
    # Thus vec(Hρ - ρH) = (I ⊗ H - H^T ⊗ I) vec(ρ)
    
    I3 = np.eye(3)
    
    # Coherent part: -i[H, ρ]
    L_coherent = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipative part
    L_dissipative = np.zeros((9, 9), dtype=complex)
    for L in [L1, L2]:
        # L ρ L† term: vec(L ρ L†) = (L†^T ⊗ L) vec(ρ)
        L_dissipative += np.kron(L.conj().T, L)
        # -0.5 {L† L, ρ} term:
        # vec(0.5 L† L ρ) = (I ⊗ 0.5 L† L) vec(ρ)
        # vec(0.5 ρ L† L) = ((0.5 L† L)^T ⊗ I) vec(ρ)
        LL = L.conj().T @ L
        L_dissipative -= 0.5 * (np.kron(I3, LL) + np.kron(LL.T, I3))
    
    L_total = L_coherent + L_dissipative
    
    return L_total

def _exponentiate_liouvillian(L, dt):
    """
    Exponentiate the Liouvillian: exp(L * dt)
    """
    # Use scipy.linalg.expm for numerical stability
    from scipy.linalg import expm
    return expm(L * dt)

def _simulate_pulse(I, Q, detuning, scale):
    """
    Simulate the pulse and return the final density matrix.
    
    Parameters:
    - I, Q: arrays of shape (80,) with I and Q amplitudes
    - detuning: detuning parameter
    - scale: drive scale parameter
    
    Returns:
    - Final density matrix (3x3)
    """
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    rho_vec = rho.flatten()
    
    # Precompute Liouvillian exponentials for each step
    for i in range(N_STEPS):
        H = _build_hamiltonian(detuning, scale, I[i], Q[i])
        L = _build_liouvillian(H)
        U = _exponentiate_liouvillian(L, DT)
        rho_vec = U @ rho_vec
    
    rho_final = rho_vec.reshape((3, 3))
    
    # Ensure Hermiticity and trace preservation (small numerical errors)
    rho_final = 0.5 * (rho_final + rho_final.conj().T)
    trace = np.trace(rho_final)
    if abs(trace) > 1e-15:
        rho_final /= trace
    
    return rho_final

def _fidelity_to_X(rho, psi_initial):
    """
    Compute fidelity to the target state after applying X gate.
    
    The target gate is X, which maps |0> -> |1> and |1> -> |0>.
    So if we start in state |ψ>, the target final state is X|ψ>.
    
    Fidelity = |<Xψ|ρ|Xψ>|
    """
    # X gate matrix
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    # Target state
    psi_target = X @ psi_initial
    
    # Fidelity: |<ψ_target|ρ|ψ_target>|
    fidelity = np.abs(np.vdot(psi_target, rho @ psi_target)) ** 2
    
    return fidelity

def _evaluate_pulse(I, Q):
    """
    Evaluate the pulse across all 9 cases and return the score.
    
    Score = mean + max of infidelities over the 9 cases.
    """
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [SCALE_MIN, 1.0, SCALE_MAX]
    
    # Initial states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    initial_states = [psi_0, psi_1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    infidelities = []
    
    for det in detunings:
        for scl in scales:
            # Simulate once to get the final density matrix
            # But wait - the final density matrix depends on the initial state!
            # We need to simulate for each initial state.
            
            case_infidelities = []
            for psi in initial_states:
                # Initialize rho = |psi><psi|
                rho = np.outer(psi, psi.conj())
                rho_vec = rho.flatten()
                
                # Simulate
                for i in range(N_STEPS):
                    H = _build_hamiltonian(det, scl, I[i], Q[i])
                    L = _build_liouvillian(H)
                    U = _exponentiate_liouvillian(L, DT)
                    rho_vec = U @ rho_vec
                
                rho_final = rho_vec.reshape((3, 3))
                rho_final = 0.5 * (rho_final + rho_final.conj().T)
                trace = np.trace(rho_final)
                if abs(trace) > 1e-15:
                    rho_final /= trace
                
                # Compute fidelity
                fid = _fidelity_to_X(rho_final, psi)
                infid = 1.0 - fid
                case_infidelities.append(infid)
            
            # Mean fidelity over the 6 initial states
            mean_infid = np.mean(case_infidelities)
            infidelities.append(mean_infid)
    
    # Score = mean + max of the 9 infidelities
    score = np.mean(infidelities) + np.max(infidelities)
    
    return score

def _check_validity(I, Q):
    """
    Check if the pulse satisfies the constraints:
    1. sqrt(I^2 + Q^2) <= 0.4 each step
    2. ||p[i+1] - p[i]|| / dt <= 0.12, including from zero into step 1 and from last step to zero
    """
    # Amplitude constraint
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.any(amplitudes > AMPLITUDE_BOUND + 1e-10):
        return False
    
    # Slew rate constraint
    # From zero to step 0
    p0 = np.array([I[0], Q[0]])
    if np.linalg.norm(p0) / DT > SLEW_BOUND + 1e-10:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        p_i = np.array([I[i], Q[i]])
        p_ip1 = np.array([I[i+1], Q[i+1]])
        if np.linalg.norm(p_ip1 - p_i) / DT > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    p_last = np.array([I[-1], Q[-1]])
    if np.linalg.norm(p_last) / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True

def _generate_drag_pulse(alpha=1.0, beta=0.0):
    """
    Generate a DRAG-style pulse.
    
    The seed is a Gaussian with sigma = 40/6 ns, offset to 0 at edges, area pi.
    DRAG adds a derivative term to suppress leakage.
    
    Parameters:
    - alpha: scaling factor for the main Gaussian
    - beta: DRAG parameter (derivative weight)
    """
    t = np.arange(N_STEPS) * DT
    t_center = T_TOTAL / 2.0
    sigma = T_TOTAL / 6.0  # 40/6 ns
    
    # Main Gaussian
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Offset to 0 at edges: subtract the value at t=0
    gauss_offset = gauss - gauss[0]
    
    # Ensure non-negative and normalize area to pi
    gauss_offset = np.maximum(gauss_offset, 0)
    area = np.sum(gauss_offset) * DT
    if area > 0:
        gauss_offset = gauss_offset * (np.pi / area)
    
    # Derivative for DRAG
    # d/dt of the Gaussian
    deriv = -((t - t_center) / sigma**2) * gauss_offset
    
    # I and Q components
    I = alpha * gauss_offset
    Q = beta * deriv
    
    # Normalize to have area pi for I component
    area_I = np.sum(I) * DT
    if area_I > 0:
        I = I * (np.pi / area_I)
    
    # Ensure validity by scaling down if needed
    # Check amplitude
    amplitudes = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amplitudes)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        I *= scale_factor
        Q *= scale_factor
    
    # Check slew rate and scale down if needed
    # We'll do a simple iterative scaling
    for _ in range(10):
        if _check_validity(I, Q):
            break
        
        # Scale down
        I *= 0.95
        Q *= 0.95
    
    return I, Q

def pulse(target_gate, hardware_model):
    """
    Main entry point.
    
    Returns an (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second deadline
    
    if target_gate != 'X_pi':
        # Return a zero pulse for invalid gates
        return np.zeros((N_STEPS, 2))
    
    # Start with the DRAG pulse
    I, Q = _generate_drag_pulse(alpha=1.0, beta=0.0)
    
    best_I = I.copy()
    best_Q = Q.copy()
    best_score = _evaluate_pulse(best_I, best_Q)
    
    # Try different DRAG parameters
    for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        if time.time() > deadline - 30.0:
            break
        
        I, Q = _generate_drag_pulse(alpha=1.0, beta=beta)
        if _check_validity(I, Q):
            score = _evaluate_pulse(I, Q)
            if score < best_score:
                best_score = score
                best_I = I.copy()
                best_Q = Q.copy()
    
    # Try different alpha values
    for alpha in [0.9, 0.95, 1.0, 1.05, 1.1]:
        if time.time() > deadline - 30.0:
            break
        
        for beta in [0.0, 0.1, 0.2, 0.3]:
            if time.time() > deadline - 30.0:
                break
            
            I, Q = _generate_drag_pulse(alpha=alpha, beta=beta)
            if _check_validity(I, Q):
                score = _evaluate_pulse(I, Q)
                if score < best_score:
                    best_score = score
                    best_I = I.copy()
                    best_Q = Q.copy()
    
    # Return the best pulse found
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
