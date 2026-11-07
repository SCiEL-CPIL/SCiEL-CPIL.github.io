"""Rewritten program: Legendre-basis pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import eval_legendre

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
SCALE_MIN, SCALE_MAX = 0.97, 1.03
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
N_LEVELS = 3
N_BASIS = 8  # Legendre polynomials

# Hardware model defaults
HARDWARE = {
    'anharmonicity': ANHARMONICITY,
    'amplitude_bound': AMPLITUDE_BOUND,
    'slew_bound': SLEW_BOUND,
    'detuning_bounds': (-DETUNING_BOUND, DETUNING_BOUND),
    'drive_scale_bounds': (SCALE_MIN, SCALE_MAX),
    'T1': T1,
    'Tphi': Tphi,
}


def _legendre_basis(t):
    """Return (N_STEPS, N_BASIS) array of Legendre polynomials evaluated at t in [0,1]."""
    # Map t from [0, 1] to [-1, 1] for standard Legendre
    x = 2 * t - 1
    basis = np.zeros((N_STEPS, N_BASIS))
    for k in range(N_BASIS):
        basis[:, k] = eval_legendre(k, x)
    return basis


def _coeffs_to_pulse(coeffs):
    """Convert Legendre coefficients to (N_STEPS, 2) pulse array [I, Q]."""
    t = np.linspace(0, 1, N_STEPS, endpoint=False) + 0.5 / N_STEPS  # midpoints
    basis = _legendre_basis(t)
    # coeffs = [I_c0, I_c1, ..., I_c7, Q_c0, Q_c1, ..., Q_c7]
    I_coeffs = coeffs[:N_BASIS]
    Q_coeffs = coeffs[N_BASIS:2 * N_BASIS]
    I = basis @ I_coeffs
    Q = basis @ Q_coeffs
    return np.column_stack([I, Q])


def _pulse_to_coeffs(pulse):
    """Convert pulse to Legendre coefficients (least squares)."""
    t = np.linspace(0, 1, N_STEPS, endpoint=False) + 0.5 / N_STEPS
    basis = _legendre_basis(t)
    I = pulse[:, 0]
    Q = pulse[:, 1]
    I_coeffs, _, _, _ = np.linalg.lstsq(basis, I, rcond=None)
    Q_coeffs, _, _, _ = np.linalg.lstsq(basis, Q, rcond=None)
    return np.concatenate([I_coeffs, Q_coeffs])


def _seed_pulse():
    """DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, T_TOTAL, N_STEPS, endpoint=False) + DT / 2
    # Gaussian with sigma = 40/6 ns
    sigma = T_TOTAL / 6.0
    center = T_TOTAL / 2.0
    # Area = pi for X-pi gate
    # Integral of A * exp(-(t-center)^2 / (2*sigma^2)) dt = pi
    # A * sigma * sqrt(2*pi) = pi => A = 1 / (sigma * sqrt(2*pi))
    A = 1.0 / (sigma * np.sqrt(2 * np.pi))
    I = A * np.exp(-(t - center) ** 2 / (2 * sigma ** 2))
    # DRAG component: proportional to derivative of envelope
    Q = -0.1 * A * sigma * np.exp(-(t - center) ** 2 / (2 * sigma ** 2)) * (t - center) / sigma
    # Offset to make edges zero
    edge_offset = (I[0] + I[-1]) / 2
    I = I - edge_offset
    Q = Q - (Q[0] + Q[-1]) / 2
    # Ensure amplitude bound
    amp = np.sqrt(I ** 2 + Q ** 2)
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        I *= AMPLITUDE_BOUND / max_amp
        Q *= AMPLITUDE_BOUND / max_amp
    return np.column_stack([I, Q])


def _build_hamiltonian(detuning, scale, I, Q, anharmonicity=ANHARMONICITY):
    """Build the 3-level Hamiltonian matrix."""
    # a = diag(sqrt([1, 2]), 1) - ladder operator
    a = np.zeros((N_LEVELS, N_LEVELS), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    adag = a.conj().T
    n = adag @ a  # number operator

    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
    H = detuning * n + 0.5 * anharmonicity * (n @ (n - np.eye(N_LEVELS)))
    H += 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    return H


def _liouvillian_step(H, dt, T1, Tphi):
    """Compute the superoperator for one Liouvillian step.
    
    Returns the 9x9 matrix representing the map rho -> rho' for one step.
    """
    N = N_LEVELS
    dim = N * N
    # Coherent evolution: U = exp(-i H dt)
    U = expm(-1j * H * dt)
    
    # Build the superoperator for coherent part: rho' = U rho U^dag
    # In vectorized form: vec(U rho U^dag) = (U^T ⊗ U) vec(rho)
    U_super = np.kron(U.T, U)
    
    # Dissipative part: Lindblad operators
    a = np.zeros((N, N), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    adag = a.conj().T
    n = adag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Lindblad superoperator for each L:
    # D[L] rho = L rho L^dag - 0.5 {L^dag L, rho}
    # In vectorized form:
    # vec(L rho L^dag) = (L^T ⊗ L) vec(rho)
    # vec(L^dag L rho) = (I ⊗ L^dag L) vec(rho)
    # vec(rho L^dag L) = ((L^dag L)^T ⊗ I) vec(rho)
    
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    # Total Liouvillian superoperator
    # rho' = (U rho U^dag) + dt * sum_k [L_k rho L_k^dag - 0.5(L_k^dag L_k rho + rho L_k^dag L_k)]
    # Note: We should use the full Liouvillian exponentiation, but for small dt, first order is ok.
    # For better accuracy, let's build the full Liouvillian and exponentiate.
    
    # Actually, let's build the full Liouvillian L such that d rho/dt = L(rho)
    # Then rho' = expm(L * dt) vec(rho)
    
    # Build the Liouvillian matrix
    L_mat = np.zeros((dim, dim), dtype=complex)
    
    # Coherent part: -i [H, rho]
    H_vec = np.kron(H.T, H)  # This gives vec(H rho)
    H_vec2 = np.kron(np.eye(N), H)  # vec(rho H)
    # -i[H, rho] = -i(H rho - rho H)
    L_mat += -1j * (np.kron(H.T, np.eye(N)) - np.kron(np.eye(N), H))
    
    # Wait, let me redo this properly.
    # vec(A rho B) = (B^T ⊗ A) vec(rho)
    # vec(H rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    # vec([H, rho]) = vec(H rho - rho H) = ((I ⊗ H) - (H^T ⊗ I)) vec(rho)
    
    L_mat = -1j * (np.kron(np.eye(N), H) - np.kron(H.T, np.eye(N)))
    
    # Dissipative part
    for L_op in [L1, L2]:
        LdL = L_op.conj().T @ L_op
        # vec(L rho L^dag) = ((L^dag)^T ⊗ L) vec(rho) = (L.conj() ⊗ L) vec(rho)
        L_mat += np.kron(L_op.conj(), L_op)
        # -0.5 vec(L^dag L rho) = -0.5 (I ⊗ L^dag L) vec(rho)
        L_mat += -0.5 * np.kron(np.eye(N), LdL)
        # -0.5 vec(rho L^dag L) = -0.5 ((L^dag L)^T ⊗ I) vec(rho)
        L_mat += -0.5 * np.kron(LdL.T, np.eye(N))
    
    # Exponentiate
    return expm(L_mat * dt)


def _evolve_pure_state(pulse, detuning, scale, anharmonicity=ANHARMONICITY, T1=T1, Tphi=Tphi):
    """Evolve the system for each initial state and return fidelities."""
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    # Target: X gate on |0> and |1>
    # X|0> = |1>, X|1> = |0>
    # For superpositions, X applies the same transformation
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
        np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>-|1>)/sqrt(2) = (-|1>+|0>)/sqrt(2)
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+i|1>)/sqrt(2) = (i|1>+|0>)/sqrt(2)
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>-i|1>)/sqrt(2) = (-i|1>+|0>)/sqrt(2)
    ]
    
    fidelities = []
    for state, target in zip(states, target_states):
        rho = np.outer(state, state.conj())
        # Vectorize rho
        rho_vec = rho.flatten()
        
        # Evolve step by step
        for i in range(N_STEPS):
            I_val = pulse[i, 0]
            Q_val = pulse[i, 1]
            H = _build_hamiltonian(detuning, scale, I_val, Q_val, anharmonicity)
            S = _liouvillian_step(H, DT, T1, Tphi)
            rho_vec = S @ rho_vec
        
        # De-vectorize
        rho_final = rho_vec.reshape((N_LEVELS, N_LEVELS))
        
        # Fidelity for pure states: F = |<target|psi>|^2 = <target|rho|target>
        fid = np.real(np.vdot(target, rho_final @ target))
        fidelities.append(fid)
    
    return np.mean(fidelities)


def _score_pulse(pulse):
    """Compute the score for a given pulse."""
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [SCALE_MIN, 1.0, SCALE_MAX]
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            fid = _evolve_pure_state(pulse, det, scale)
            infidelities.append(1.0 - fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)


def _check_validity(pulse):
    """Check if the pulse satisfies amplitude and slew constraints."""
    # Amplitude bound
    amp = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    # Slew bound: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # From zero to step 1
    diff = np.sqrt(pulse[0, 0] ** 2 + pulse[0, 1] ** 2)
    if diff / DT > SLEW_BOUND + 1e-10:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        dI = pulse[i + 1, 0] - pulse[i, 0]
        dQ = pulse[i + 1, 1] - pulse[i, 1]
        norm = np.sqrt(dI ** 2 + dQ ** 2)
        if norm / DT > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    dI = -pulse[-1, 0]
    dQ = -pulse[-1, 1]
    norm = np.sqrt(dI ** 2 + dQ ** 2)
    if norm / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True


def _project_to_feasible(coeffs):
    """Project coefficients to satisfy constraints."""
    pulse = _coeffs_to_pulse(coeffs)
    
    # Amplitude constraint
    amp = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / max_amp
        pulse *= scale
    
    # Slew constraint: need to smooth the pulse
    # Iteratively reduce high-frequency components
    for _ in range(10):
        # Check slew
        valid = True
        # From zero to step 1
        if np.sqrt(pulse[0, 0] ** 2 + pulse[0, 1] ** 2) / DT > SLEW_BOUND:
            valid = False
        for i in range(N_STEPS - 1):
            dI = pulse[i + 1, 0] - pulse[i, 0]
            dQ = pulse[i + 1, 1] - pulse[i, 1]
            if np.sqrt(dI ** 2 + dQ ** 2) / DT > SLEW_BOUND:
                valid = False
                break
        if np.sqrt(pulse[-1, 0] ** 2 + pulse[-1, 1] ** 2) / DT > SLEW_BOUND:
            valid = False
        
        if valid:
            break
        
        # Apply low-pass filter to reduce slew
        # Simple boxcar filter
        kernel = np.ones(3) / 3.0
        pulse[:, 0] = np.convolve(pulse[:, 0], kernel, mode='same')
        pulse[:, 1] = np.convolve(pulse[:, 1], kernel, mode='same')
    
    # Re-check amplitude after smoothing
    amp = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / max_amp
        pulse *= scale
    
    return _pulse_to_coeffs(pulse)


def pulse(target_gate, hardware_model):
    """Main entry point: return optimized pulse."""
    start_time = time.time()
    DEADLINE = 300.0  # seconds, with safety margin
    
    if target_gate != 'X_pi':
        # Return seed pulse for invalid gate
        return _seed_pulse()
    
    # Start with seed pulse
    seed = _seed_pulse()
    best_coeffs = _pulse_to_coeffs(seed)
    best_pulse = seed
    best_score = _score_pulse(seed)
    
    # Check validity of seed
    if _check_validity(seed):
        valid_seed = True
    else:
        valid_seed = False
        # Project to feasible
        best_coeffs = _project_to_feasible(best_coeffs)
        best_pulse = _coeffs_to_pulse(best_coeffs)
        best_score = _score_pulse(best_pulse)
    
    # Define objective function for optimization
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:  # Safety margin
            return best_score  # Return current best
        pulse_candidate = _coeffs_to_pulse(coeffs)
        
        # Check amplitude constraint
        amp = np.sqrt(pulse_candidate[:, 0] ** 2 + pulse_candidate[:, 1] ** 2)
        if np.max(amp) > AMPLITUDE_BOUND:
            return best_score + 1.0  # Penalty
        
        # Check slew constraint
        if np.sqrt(pulse_candidate[0, 0] ** 2 + pulse_candidate[0, 1] ** 2) / DT > SLEW_BOUND:
            return best_score + 1.0
        for i in range(N_STEPS - 1):
            dI = pulse_candidate[i + 1, 0] - pulse_candidate[i, 0]
            dQ = pulse_candidate[i + 1, 1] - pulse_candidate[i + 1, 1]
            if np.sqrt(dI ** 2 + dQ ** 2) / DT > SLEW_BOUND:
                return best_score + 1.0
        if np.sqrt(pulse_candidate[-1, 0] ** 2 + pulse_candidate[-1, 1] ** 2) / DT > SLEW_BOUND:
            return best_score + 1.0
        
        return _score_pulse(pulse_candidate)
    
    # Optimize using L-BFGS-B
    # Bounds for coefficients: keep them reasonable
    coeff_bounds = [(-1.0, 1.0)] * (2 * N_BASIS)
    
    result = minimize(
        objective,
        best_coeffs,
        method='L-BFGS-B',
        bounds=coeff_bounds,
        options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 100}
    )
    
    elapsed = time.time() - start_time
    if result.success and elapsed < DEADLINE - 30:
        candidate_coeffs = result.x
        candidate_pulse = _coeffs_to_pulse(candidate_coeffs)
        if _check_validity(candidate_pulse):
            candidate_score = _score_pulse(candidate_pulse)
            if candidate_score < best_score:
                best_score = candidate_score
                best_pulse = candidate_pulse
                best_coeffs = candidate_coeffs
    
    # Final validity check
    if not _check_validity(best_pulse):
        best_pulse = _seed_pulse()
    
    return best_pulse
# EVOLVE-BLOCK-END
