"""Rewritten program: robust DRAG pulse for 3-level transmon X-pi gate."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.integrate import solve_ivp
import time

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPL_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = np.array([-2 * np.pi * 0.002, 2 * np.pi * 0.002])
SCALE_BOUNDS = [0.97, 1.0, 1.03]
T1 = 25000.0  # ns
Tphi = 40000.0  # ns

# Hardware model cases: (detuning, scale)
HARDWARE_CASES = [
    (DETUNING_BOUNDS[0], SCALE_BOUNDS[0]),
    (DETUNING_BOUNDS[0], SCALE_BOUNDS[1]),
    (DETUNING_BOUNDS[0], SCALE_BOUNDS[2]),
    (0.0, SCALE_BOUNDS[0]),
    (0.0, SCALE_BOUNDS[1]),
    (0.0, SCALE_BOUNDS[2]),
    (DETUNING_BOUNDS[1], SCALE_BOUNDS[0]),
    (DETUNING_BOUNDS[1], SCALE_BOUNDS[1]),
    (DETUNING_BOUNDS[1], SCALE_BOUNDS[2]),
]

# Target states for fidelity calculation
def get_target_states():
    """Return the 4 target states for fidelity calculation."""
    psi_0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi_1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_plus = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1.0, -1j, 0.0], dtype=complex) / np.sqrt(2)
    return [psi_0, psi_1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]

# X gate matrix (applied to 2-level subspace)
X_GATE = np.array([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=complex)

def build_hamiltonian(detuning, scale, I, Q):
    """
    Build the 3-level Hamiltonian.
    H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    """
    # a matrix (lowering operator)
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2)
    a_dag = a.conj().T
    
    # n = a† a
    n_op = a_dag @ a
    
    # Static part
    H_static = detuning * n_op + 0.5 * ANHARMONICITY * (n_op @ (n_op - np.eye(3)))
    
    # Drive part
    H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    return H_static + H_drive

def lindblad_operators():
    """Return Lindblad operators."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    return [L1, L2]

def liouvillian(H, Ls):
    """
    Build the Liouvillian superoperator (in vec form).
    L(rho) = -i[H, rho] + sum_k (L_k rho L_k† - 0.5 {L_k† L_k, rho})
    """
    dim = H.shape[0]
    dim2 = dim * dim
    
    # -i[H, rho] -> -i * (H ⊗ I - I ⊗ H^T) in vec form
    # vec(H rho H†) = (H ⊗ H*) vec(rho)
    # vec(H rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    
    I_dim = np.eye(dim, dtype=complex)
    
    # Coherent: -i[H, rho] = -i H rho + i rho H
    # vec(-i H rho) = -i (I ⊗ H) vec(rho)
    # vec(i rho H) = i (H^T ⊗ I) vec(rho)
    L_coh = -1j * (np.kron(I_dim, H) - np.kron(H.T, I_dim))
    
    # Dissipative
    L_diss = np.zeros((dim2, dim2), dtype=complex)
    for L in Ls:
        L_dag = L.conj().T
        # L rho L† -> (L† ⊗ L) vec(rho)
        L_diss += np.kron(L.conj().T, L)
        # -0.5 {L† L, rho} = -0.5 L† L rho - 0.5 rho L† L
        # vec(-0.5 L† L rho) = -0.5 (I ⊗ L† L) vec(rho)
        # vec(-0.5 rho L† L) = -0.5 ((L† L)^T ⊗ I) vec(rho)
        LL_dag = L_dag @ L
        L_diss += -0.5 * (np.kron(I_dim, LL_dag) + np.kron(LL_dag.T, I_dim))
    
    return L_coh + L_diss

def vec(rho):
    """Vectorize density matrix (column-major, matching np.kron convention)."""
    return rho.flatten(order='F')

def unvec(vec_rho):
    """Devectorize."""
    dim = int(np.sqrt(len(vec_rho)))
    return vec_rho.reshape((dim, dim), order='F')

def simulate_pulse(I_Q, detuning, scale, n_steps=80, dt=0.5):
    """
    Simulate the pulse and return the final density matrix.
    I_Q: (n_steps, 2) array of [I, Q]
    """
    Ls = lindblad_operators()
    dim = 3
    dim2 = dim * dim
    
    # Initial state: |0><0|
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    
    # Precompute Liouvillian for each step (H depends on I, Q)
    rho_vec = vec(rho)
    
    for i in range(n_steps):
        I = I_Q[i, 0]
        Q = I_Q[i, 1]
        H = build_hamiltonian(detuning, scale, I, Q)
        L = liouvillian(H, Ls)
        
        # Exponentiate: rho(t+dt) = exp(L * dt) rho(t)
        # Use matrix exponential
        from scipy.linalg import expm
        Ldt = L * dt
        expLdt = expm(Ldt)
        rho_vec = expLdt @ rho_vec
    
    rho_final = unvec(rho_vec)
    # Ensure trace = 1 and Hermiticity
    rho_final = (rho_final + rho_final.conj().T) / 2
    tr = np.trace(rho_final)
    if np.abs(tr) > 1e-15:
        rho_final /= tr
    return rho_final

def fidelity(rho, psi):
    """Calculate fidelity between density matrix and pure state."""
    # F = <psi|rho|psi>
    val = np.vdot(psi, rho @ psi)
    return np.real(val)

def compute_infidelity(I_Q, detuning, scale):
    """
    Compute infidelity for a given pulse and hardware configuration.
    Infidelity = 1 - mean fidelity to X gate over target states.
    """
    rho_final = simulate_pulse(I_Q, detuning, scale)
    
    target_states = get_target_states()
    fidelities = []
    for psi in target_states:
        # Apply X gate to target state
        psi_x = X_GATE @ psi
        # Fidelity = |<psi_x|rho_final|psi>|^2 ... wait, for pure state target
        # Fidelity = <psi_x|rho|psi_x>
        f = fidelity(rho_final, psi_x)
        fidelities.append(f)
    
    mean_fid = np.mean(fidelities)
    return 1.0 - mean_fid

def compute_score(I_Q):
    """
    Compute the overall score: mean + max of infidelities over 9 hardware cases.
    """
    infidelities = []
    for detuning, scale in HARDWARE_CASES:
        inf = compute_infidelity(I_Q, detuning, scale)
        infidelities.append(inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf

def generate_drag_pulse(params):
    """
    Generate a DRAG-style pulse from parameters.
    params: [amp, sigma, beta]
    - amp: peak amplitude of Gaussian
    - sigma: Gaussian width
    - beta: DRAG amplitude (Q component)
    """
    amp, sigma, beta = params
    t = np.arange(N_STEPS) * DT
    t_center = T_TOTAL / 2.0
    
    # Gaussian envelope
    gauss = amp * np.exp(-(t - t_center)**2 / (2 * sigma**2))
    
    # Offset to make area = pi
    # Area of Gaussian = amp * sigma * sqrt(2*pi)
    # We want integral of I dt = pi
    # But we have discrete steps, so approximate
    area = np.sum(gauss) * DT
    if area > 1e-10:
        gauss = gauss * (np.pi / area)
    
    I = gauss
    
    # DRAG: Q component proportional to derivative of I
    # Q = beta * dI/dt
    if N_STEPS > 1:
        dI_dt = np.gradient(I, DT)
    else:
        dI_dt = np.zeros_like(I)
    Q = beta * dI_dt
    
    I_Q = np.column_stack([I, Q])
    return I_Q

def check_constraints(I_Q):
    """
    Check amplitude and slew constraints.
    Returns (is_valid, max_violation)
    """
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amplitudes = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
    amp_max = np.max(amplitudes)
    amp_violation = max(0, amp_max - AMPL_BOUND)
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and last step to zero
    p = I_Q
    # Add zero at start and end
    p_ext = np.vstack([np.zeros((1, 2)), p, np.zeros((1, 2))])
    diffs = np.diff(p_ext, axis=0)
    norms = np.sqrt(np.sum(diffs**2, axis=1))
    slew_max = np.max(norms) / DT
    slew_violation = max(0, slew_max - SLEW_BOUND)
    
    is_valid = (amp_violation < 1e-10) and (slew_violation < 1e-10)
    return is_valid, max(amp_violation, slew_violation)

def penalty_score(params):
    """
    Objective function with penalty for constraint violations.
    """
    I_Q = generate_drag_pulse(params)
    is_valid, violation = check_constraints(I_Q)
    
    if not is_valid:
        # Add penalty
        return 1.0 + 10.0 * violation
    
    score = compute_score(I_Q)
    return score

def optimize_pulse(deadline):
    """
    Optimize the DRAG pulse parameters.
    """
    # Initial guess
    # Gaussian with area pi over 40ns
    # amp * sigma * sqrt(2*pi) ≈ pi => amp ≈ 1/(sigma*sqrt(2))
    # For sigma ~ 6ns: amp ≈ 1/(6*1.414) ≈ 0.118
    # But we need to account for the offset to edges
    
    # Let's try a few starting points
    initial_params = [
        [0.12, 6.0, 0.0],      # Basic DRAG
        [0.15, 5.0, 0.02],     # Wider DRAG
        [0.10, 7.0, -0.01],    # Narrower
        [0.13, 5.5, 0.03],     # More DRAG
        [0.11, 6.5, -0.02],    # Less DRAG
    ]
    
    best_score = np.inf
    best_params = initial_params[0]
    
    for x0 in initial_params:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                penalty_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception as e:
            continue
    
    # Generate final pulse
    I_Q = generate_drag_pulse(best_params)
    
    # Ensure constraints are satisfied
    is_valid, violation = check_constraints(I_Q)
    if not is_valid:
        # Scale down to satisfy constraints
        amplitudes = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amplitudes) > AMPL_BOUND:
            scale_factor = AMPL_BOUND / np.max(amplitudes) * 0.999
            I_Q = I_Q * scale_factor
        
        # Check slew
        p_ext = np.vstack([np.zeros((1, 2)), I_Q, np.zeros((1, 2))])
        diffs = np.diff(p_ext, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / DT > SLEW_BOUND:
            scale_factor = SLEW_BOUND * DT / np.max(norms) * 0.999
            I_Q = I_Q * scale_factor
    
    return I_Q

def pulse(target_gate, hardware_model):
    """
    Main entry point.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gate
        return np.zeros((N_STEPS, 2))
    
    # Set deadline: 300s total, keep 30s margin
    start_time = time.time()
    deadline = start_time + 270  # 270s for optimization
    
    # Quick seed pulse first
    seed_params = [0.12, 6.0, 0.0]
    I_Q_seed = generate_drag_pulse(seed_params)
    
    # Check if seed is valid
    is_valid, _ = check_constraints(I_Q_seed)
    if not is_valid:
        # Scale down
        amplitudes = np.sqrt(I_Q_seed[:, 0]**2 + I_Q_seed[:, 1]**2)
        if np.max(amplitudes) > AMPL_BOUND:
            scale_factor = AMPL_BOUND / np.max(amplitudes) * 0.999
            I_Q_seed = I_Q_seed * scale_factor
    
    # If time allows, optimize
    if time.time() < deadline - 60:
        try:
            I_Q_opt = optimize_pulse(deadline)
            # Check validity
            is_valid_opt, _ = check_constraints(I_Q_opt)
            if is_valid_opt:
                I_Q_seed = I_Q_opt
        except Exception:
            pass
    
    # Final constraint check
    is_valid, violation = check_constraints(I_Q_seed)
    if not is_valid:
        # Force validity by scaling
        amplitudes = np.sqrt(I_Q_seed[:, 0]**2 + I_Q_seed[:, 1]**2)
        if np.max(amplitudes) > AMPL_BOUND:
            scale_factor = AMPL_BOUND / np.max(amplitudes) * 0.999
            I_Q_seed = I_Q_seed * scale_factor
        
        p_ext = np.vstack([np.zeros((1, 2)), I_Q_seed, np.zeros((1, 2))])
        diffs = np.diff(p_ext, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / DT > SLEW_BOUND:
            scale_factor = SLEW_BOUND * DT / np.max(norms) * 0.999
            I_Q_seed = I_Q_seed * scale_factor
    
    return I_Q_seed
