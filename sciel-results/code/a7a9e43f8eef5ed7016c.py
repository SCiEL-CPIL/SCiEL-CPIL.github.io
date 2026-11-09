"""Robust DRAG pulse for X-pi gate with gradient-free optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Hardware constants
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Hamiltonian parameters
ANHARMONICITY = -2 * np.pi * 0.25  # -pi/2
T1 = 25000.0
T_PHI = 40000.0

# Bounds
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)

# Build operators
def _build_operators():
    """Build the 3-level operators."""
    # a = diag(sqrt([1,2]), 1) in the |0>,|1>,|2> basis
    # Actually, a is the lowering operator: a|1> = |0>, a|2> = sqrt(2)|1>
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    
    # n = a^dagger a
    n_op = a_dag @ a
    
    return a, a_dag, n_op

A_OP, A_DAG, N_OP = _build_operators()

def _hamiltonian(detuning, scale, I, Q):
    """Build the Hamiltonian for given parameters."""
    H = (detuning * N_OP + 
         0.5 * ANHARMONICITY * N_OP @ (N_OP - np.eye(3)) +
         0.5 * scale * (I * (A_OP + A_DAG) + 1j * Q * (A_OP - A_DAG)))
    return H

def _liouvillian(H, T1, T_phi):
    """Build the Liouvillian superoperator."""
    # Lindblad operators: L1 = a/sqrt(T1), L2 = n*sqrt(2/T_phi)
    L1 = A_OP / np.sqrt(T1)
    L2 = N_OP * np.sqrt(2.0 / T_phi)
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # In superoperator form, we need to vectorize
    # Using the commutator form: L(rho) = -i(H ⊗ I - I ⊗ H^T) vec(rho) + dissipative terms
    
    dim = 3
    I_mat = np.eye(dim)
    
    # Coherent part: -i(H rho - rho H) = -i(H ⊗ I - I ⊗ H^T) vec(rho)
    # Actually: vec(H rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    # So vec(H rho - rho H) = (I ⊗ H - H^T ⊗ I) vec(rho)
    # And -i times that
    
    coherent = -1j * (np.kron(I_mat, H) - np.kron(H.T, I_mat))
    
    # Dissipative part
    def _dissipator(L):
        """Build the superoperator for a single Lindblad term."""
        L_dag = L.conj().T
        # L rho L^dag - 0.5(L^dag L rho + rho L^dag L)
        term1 = np.kron(L.conj(), L)  # vec(L rho L^dag) = (L^* ⊗ L) vec(rho)
        L_dag_L = L_dag @ L
        term2 = -0.5 * (np.kron(I_mat, L_dag_L) + np.kron(L_dag_L.T, I_mat))
        return term1 + term2
    
    dissipative = _dissipator(L1) + _dissipator(L2)
    
    return coherent + dissipative

def _evolve_state(pulse_I, pulse_Q, detuning, scale):
    """Evolve the density matrix through all steps."""
    dim = 3
    # Initial state: |0><0|
    rho = np.zeros((dim * dim, dim * dim), dtype=complex)
    rho[0, 0] = 1.0  # |0><0| vectorized
    
    for i in range(N_STEPS):
        H = _hamiltonian(detuning, scale, pulse_I[i], pulse_Q[i])
        L = _liouvillian(H, T1, T_phi)
        # Evolve: rho(t+dt) = expm(L * dt) rho(t)
        U = expm(L * DT)
        rho = U @ rho
    
    return rho

def _fidelity_to_X(rho_final, initial_state):
    """Compute fidelity for a specific initial state."""
    dim = 3
    # Apply X gate to initial state to get target
    # X gate swaps |0> and |1>, leaves |2> unchanged
    X = np.zeros((dim, dim), dtype=complex)
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    X[2, 2] = 1.0
    
    # Target state
    psi_target = X @ initial_state
    rho_target = np.outer(psi_target, psi_target.conj())
    
    # Fidelity: F = Tr(rho_final @ rho_target) for pure target
    # More precisely: F = |<psi_target|rho_final|psi_target>| for pure states
    # But rho_final may be mixed
    F = np.real(np.trace(rho_final.reshape(dim, dim) @ rho_target))
    return max(0, min(1, F))

def _compute_infidelity(pulse_I, pulse_Q):
    """Compute the mean + max infidelity over the 9 cases."""
    # Initial states: |0>, |1>, (|0>±|1>)/sqrt(2), (|0>±i|1>)/sqrt(2)
    states = []
    e0 = np.array([1, 0, 0], dtype=complex)
    e1 = np.array([0, 1, 0], dtype=complex)
    states.append(e0)
    states.append(e1)
    states.append((e0 + e1) / np.sqrt(2))
    states.append((e0 - e1) / np.sqrt(2))
    states.append((e0 + 1j * e1) / np.sqrt(2))
    states.append((e0 - 1j * e1) / np.sqrt(2))
    
    # 9 cases: detuning in (-2pi*0.002, 0, 2pi*0.002), scale in (0.97, 1, 1.03)
    detunings = [DETUNING_BOUNDS[0], 0.0, DETUNING_BOUNDS[1]]
    scales = [SCALE_BOUNDS[0], 1.0, SCALE_BOUNDS[1]]
    
    fidelities = []
    for det in detunings:
        for sc in scales:
            rho_final = _evolve_state(pulse_I, pulse_Q, det, sc)
            for state in states:
                F = _fidelity_to_X(rho_final, state)
                fidelities.append(F)
    
    fidelities = np.array(fidelities)
    mean_fid = np.mean(fidelities)
    max_fid = np.max(fidelities)
    # Score = mean + max of infidelities = (1 - mean_fid) + (1 - max_fid)
    # Wait, re-reading: "Score = mean + max of the 9"
    # infidelity = 1 - mean fidelity
    # Score = mean of infidelities + max of infidelities? Or mean infidelity + max infidelity?
    # "infidelity = 1 - mean fidelity to X over |0>, |1>, ..."
    # So for each of the 9 cases, we compute infidelity = 1 - mean_fidelity_for_that_case
    # Then Score = mean(infidelities) + max(infidelities)
    
    # Let me recompute: for each case, compute mean fidelity over the 6 states
    case_infidelities = []
    for det in detunings:
        for sc in scales:
            rho_final = _evolve_state(pulse_I, pulse_Q, det, sc)
            case_fids = []
            for state in states:
                F = _fidelity_to_X(rho_final, state)
                case_fids.append(F)
            mean_fid = np.mean(case_fids)
            infid = 1.0 - mean_fid
            case_infidelities.append(infid)
    
    case_infidelities = np.array(case_infidelities)
    score = np.mean(case_infidelities) + np.max(case_infidelities)
    return score

def _check_validity(pulse_I, pulse_Q):
    """Check amplitude and slew rate constraints."""
    # Amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMP_BOUND + 1e-10:
            return False
    
    # Slew rate: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero into step 1 and from last step to zero
    # Step 0 to step 1
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
    
    # From zero to step 0
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-10:
        return False
    
    # From last step to zero
    slewend = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewend > SLEW_BOUND + 1e-10:
        return False
    
    return True

def _seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT - TOTAL_TIME / 2  # Centered time
    sigma = 40 / 6.0  # ~6.67 ns
    
    # Gaussian envelope
    gauss = np.exp(-t**2 / (2 * sigma**2))
    
    # Offset to make it zero at edges
    edge_val = np.exp(-(TOTAL_TIME/2)**2 / (2 * sigma**2))
    gauss = gauss - edge_val
    gauss = np.maximum(gauss, 0)
    
    # Scale to have area pi (for X-pi gate, the integral of the drive should give pi)
    # Area = sum(gauss) * DT
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    pulse_I = gauss
    pulse_Q = np.zeros(N_STEPS)
    
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = pulse_I[i]
        if amp > AMP_BOUND:
            pulse_I[i] = AMP_BOUND
    
    # Enforce slew rate
    # This is tricky; we may need to iteratively reduce
    for _ in range(100):
        valid = True
        for i in range(N_STEPS):
            amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
            if amp > AMP_BOUND + 1e-10:
                valid = False
                break
        if valid:
            break
        for i in range(N_STEPS):
            amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
            if amp > AMP_BOUND:
                scale = AMP_BOUND / amp
                pulse_I[i] *= scale
                pulse_Q[i] *= scale
    
    # Check slew rate and reduce if needed
    for _ in range(100):
        max_slew = 0
        # From zero to step 0
        slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
        max_slew = max(max_slew, slew0)
        for i in range(N_STEPS - 1):
            dI = pulse_I[i+1] - pulse_I[i]
            dQ = pulse_Q[i+1] - pulse_Q[i]
            slew = np.sqrt(dI**2 + dQ**2) / DT
            max_slew = max(max_slew, slew)
        slewend = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
        max_slew = max(max_slew, slewend)
        
        if max_slew <= SLEW_BOUND + 1e-10:
            break
        # Scale down the pulse
        scale = SLEW_BOUND / max_slew * 0.99
        pulse_I *= scale
        pulse_Q *= scale
    
    return pulse_I, pulse_Q

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes for a 40 ns X-pi gate.
    Uses gradient-free optimization to find a robust DRAG pulse.
    """
    if target_gate != 'X_pi':
        # Return seed for non-X_pi gates
        I, Q = _seed_pulse()
        return np.column_stack([I, Q])
    
    start_time = time.time()
    deadline = start_time + 320  # 10s safety margin
    
    # Start with seed
    pulse_I, pulse_Q = _seed_pulse()
    best_I, best_Q = pulse_I.copy(), pulse_Q.copy()
    best_score = _compute_infidelity(best_I, best_Q)
    
    # Optimization using Nelder-Mead on a reduced parameter set
    # We'll optimize the Gaussian parameters: sigma, center, DRAG beta
    # Or we can optimize a subset of pulse parameters
    
    # Let's try optimizing the full pulse with constraints handled by projection
    # But 80*2 = 160 parameters is too many for Nelder-Mead
    
    # Instead, let's use a structured parameterization:
    # Gaussian with sigma, center, amplitude, and DRAG component
    # DRAG: Q component is proportional to derivative of I
    
    def _param_pulse(params):
        """Generate pulse from parameters: [sigma, center, amp, beta_drag]"""
        sigma = np.abs(params[0]) + 1.0
        center = params[1]
        amp = np.abs(params[2])
        beta = params[3]
        
        t = np.arange(N_STEPS) * DT - TOTAL_TIME / 2
        
        # Gaussian envelope
        gauss = np.exp(-(t - center)**2 / (2 * sigma**2))
        edge_val = np.exp(-(TOTAL_TIME/2 - center)**2 / (2 * sigma**2))
        gauss = gauss - edge_val
        gauss = np.maximum(gauss, 0)
        
        # Scale to target area
        area = np.sum(gauss) * DT
        if area > 0:
            gauss = gauss * (np.pi / area)
        
        pulse_I = amp * gauss
        
        # DRAG: Q = beta * dI/dt
        dI = np.gradient(pulse_I, DT)
        pulse_Q = beta * dI
        
        # Enforce amplitude bound
        for i in range(N_STEPS):
            a = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
            if a > AMP_BOUND:
                scale = AMP_BOUND / a
                pulse_I[i] *= scale
                pulse_Q[i] *= scale
        
        # Enforce slew rate
        for _ in range(200):
            max_slew = 0
            slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
            max_slew = max(max_slew, slew0)
            for i in range(N_STEPS - 1):
                dI = pulse_I[i+1] - pulse_I[i]
                dQ = pulse_Q[i+1] - pulse_Q[i]
                slew = np.sqrt(dI**2 + dQ**2) / DT
                max_slew = max(max_slew, slew)
            slewend = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
            max_slew = max(max_slew, slewend)
            
            if max_slew <= SLEW_BOUND + 1e-10:
                break
            scale = SLEW_BOUND / max_slew * 0.99
            pulse_I *= scale
            pulse_Q *= scale
        
        return pulse_I, pulse_Q
    
    def _objective(params):
        """Objective function for optimization."""
        pulse_I, pulse_Q = _param_pulse(params)
        if not _check_validity(pulse_I, pulse_Q):
            return 100.0  # Penalty for invalid
        return _compute_infidelity(pulse_I, pulse_Q)
    
    # Initial parameters: sigma ~ 6.67, center = 0, amp ~ 0.3, beta ~ 0.1
    x0 = np.array([40/6.0, 0.0, 0.3, 0.1])
    
    # Try Nelder-Mead
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if result.fun < best_score:
            pulse_I, pulse_Q = _param_pulse(result.x)
            if _check_validity(pulse_I, pulse_Q):
                best_I, best_Q = pulse_I.copy(), pulse_Q.copy()
                best_score = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return np.column_stack([best_I, best_Q])
    
    # Try Powell method
    try:
        result2 = minimize(_objective, x0, method='Powell',
                          options={'maxiter': 100, 'xtol': 1e-4, 'ftol': 1e-6})
        
        if result2.fun < best_score:
            pulse_I, pulse_Q = _param_pulse(result2.x)
            if _check_validity(pulse_I, pulse_Q):
                best_I, best_Q = pulse_I.copy(), pulse_Q.copy()
                best_score = result2.fun
    except Exception:
        pass
    
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
