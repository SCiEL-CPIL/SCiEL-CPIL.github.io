"""Rewritten program for robust control pulses for a three-level transmon.

Implements a spectral nulling representation: the pulse is defined as a sum of
two orthogonal Gaussian envelopes (main X-pulse and derivative-coupled DRAG
component) with independent amplitudes and time-shifts. The four parameters
(main amplitude, DRAG amplitude, main time-shift, DRAG time-shift) are optimized
to minimize leakage into |2> across the detuning range while satisfying all
hardware constraints.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import erf

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = N_STEPS * DT  # 40 ns

# Build the 3x3 ladder operators
# a = diag(sqrt([1, 2]), 1) in the basis |0>, |1>, |2>
# Actually, a is the lowering operator:
# a|0> = 0, a|1> = |0>, a|2> = sqrt(2)|1>
# So in matrix form (columns are images of basis vectors):
a_mat = np.zeros((3, 3), dtype=complex)
a_mat[0, 1] = 1.0
a_mat[1, 2] = np.sqrt(2.0)
a_dag = a_mat.conj().T

# Number operator n = a^dagger a
n_mat = a_dag @ a_mat
# n_mat should be diag(0, 1, 2)

def build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for one time step.
    
    H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    
    Lindblad operators: L1 = a/sqrt(T1), L2 = n*sqrt(2/Tphi)
    """
    # Hamiltonian
    H = (detuning * n_mat + 
         0.5 * ANHARMONICITY * (n_mat @ (n_mat - np.eye(3))) + 
         0.5 * scale * (I * (a_mat + a_dag) + 1j * Q * (a_mat - a_dag)))
    
    # Lindblad operators
    L1 = a_mat / np.sqrt(T1)
    L2 = n_mat * np.sqrt(2.0 / Tphi)
    
    # Liouvillian: dρ/dt = -i[H, ρ] + Σ (LρL† - 1/2{L†L, ρ})
    # In vectorized form (row-major or column-major?), let's use column-major (Fortran)
    # vec(LρL†) = (L† ⊗ L) vec(ρ)
    # vec({L†L, ρ}/2) = 1/2 ((L†L) ⊗ I + I ⊗ (L†L)^T) vec(ρ)
    
    I3 = np.eye(3, dtype=complex)
    
    def vec_op(op):
        """Vectorize operator acting on density matrix (column-major)."""
        # For AρB, vec(AρB) = (B^T ⊗ A) vec(ρ)
        return op
    
    # -i[H, ρ] = -i(Hρ - ρH)
    # vec(Hρ) = (I ⊗ H) vec(ρ)
    # vec(ρH) = (H^T ⊗ I) vec(ρ)
    L_ham = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipation terms
    L_diss = np.zeros((9, 9), dtype=complex)
    for L in [L1, L2]:
        Ldag = L.conj().T
        # LρL†: (L†^T ⊗ L) = (L ⊗ L†) since L†^T = L
        L_diss += np.kron(L, L.conj().T)
        # -1/2{L†L, ρ} = -1/2(L†L ρ + ρ L†L)
        LdagL = Ldag @ L
        L_diss += -0.5 * (np.kron(I3, LdagL) + np.kron(LdagL.T, I3))
    
    Liouvillian = L_ham + L_diss
    return Liouvillian

def simulate_pulse(pulse_I, pulse_Q, detuning, scale):
    """Simulate the full evolution and return the final density matrix.
    
    pulse_I, pulse_Q: arrays of shape (N_STEPS,) with I and Q values.
    """
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    # We'll simulate from |0> and also need to check other initial states.
    # Actually, we need fidelity to X gate for 5 initial states.
    # Let's simulate all 5 at once by tracking the full evolution operator.
    
    # Better: simulate the unitary (or quantum channel) evolution.
    # Since we have dissipation, we need to track the full map.
    # For simplicity, let's simulate each initial state separately.
    
    # Actually, let's just return the final state for |0> initial state for now,
    # and handle the full scoring in the objective function.
    
    return rho

def fidelity_to_X(rho_final, initial_state):
    """Compute fidelity between final state and X|initial_state>.
    
    For pure states: F = |<X|init>|^2
    For mixed states: F = Tr(rho_final X rho_initial X†) ... 
    Actually, fidelity for pure target: F = <psi_target|rho_final|psi_target>
    """
    X = np.zeros((3, 3), dtype=complex)
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    # X|0> = |1>, X|1> = |0>, X|2> = 0
    
    X_init = X @ initial_state
    # Fidelity = |<X_init|rho_final|X_init>| ... for pure target
    # Actually, if target is pure |psi>, F = <psi|rho|psi>
    return np.real(np.vdot(X_init, rho_final @ X_init))

def compute_score(pulse_I, pulse_Q):
    """Compute the score for a given pulse.
    
    Returns (mean_infidelity, max_infidelity, invalid_flag)
    """
    # Check constraints
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMPLITUDE_BOUND + 1e-9:
            return (np.inf, np.inf, True)
    
    # Slew rate check: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # From zero into step 1
    diff = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2)
    if diff / DT > SLEW_BOUND + 1e-9:
        return (np.inf, np.inf, True)
    
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        diff = np.sqrt(dI**2 + dQ**2)
        if diff / DT > SLEW_BOUND + 1e-9:
            return (np.inf, np.inf, True)
    
    # From last step to zero
    diff = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2)
    if diff / DT > SLEW_BOUND + 1e-9:
        return (np.inf, np.inf, True)
    
    # Initial states
    psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_plus = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2)
    psi_iplus = np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2)
    psi_iminus = np.array([1.0, -1j, 0.0], dtype=complex) / np.sqrt(2)
    
    initial_states = [psi0, psi1, psi_plus, psi_minus, psi_iplus, psi_iminus]
    
    detunings = [DETUNING_BOUNDS[0], 0.0, DETUNING_BOUNDS[1]]
    scales = [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]
    
    infidelities = []
    
    for det in detunings:
        for sc in scales:
            # Simulate for each initial state
            case_fidelities = []
            for psi_init in initial_states:
                rho = np.zeros((3, 3), dtype=complex)
                rho = np.outer(psi_init, psi_init.conj())
                
                # Evolve step by step
                for step in range(N_STEPS):
                    I_val = pulse_I[step]
                    Q_val = pulse_Q[step]
                    
                    # Build Liouvillian
                    L = build_liouvillian(det, sc, I_val, Q_val)
                    
                    # Exponentiate: exp(L * DT)
                    # Use matrix exponential
                    from scipy.linalg import expm
                    U_step = expm(L * DT)
                    
                    # rho_vec = vec(rho), rho_new_vec = U_step @ rho_vec
                    rho_vec = rho.flatten(order='F')
                    rho_new_vec = U_step @ rho_vec
                    rho = rho_new_vec.reshape((3, 3), order='F')
                
                # Compute fidelity
                X = np.zeros((3, 3), dtype=complex)
                X[0, 1] = 1.0
                X[1, 0] = 1.0
                psi_target = X @ psi_init
                fid = np.real(np.vdot(psi_target, rho @ psi_target))
                case_fidelities.append(fid)
            
            mean_fid = np.mean(case_fidelities)
            infid = 1.0 - mean_fid
            infidelities.append(infid)
    
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    score = mean_infid + max_infid
    return (score, False)

def generate_pulse_from_params(params):
    """Generate I and Q pulse arrays from 4 parameters.
    
    params: [A_main, A_drag, t_shift_main, t_shift_drag]
    """
    A_main = params[0]
    A_drag = params[1]
    t_shift_main = params[2]
    t_shift_drag = params[3]
    
    t = np.linspace(0, DURATION, N_STEPS, endpoint=False) + DT/2  # Step centers
    
    sigma_main = 40.0 / 6.0
    sigma_drag = 40.0 / 6.0
    
    # Main Gaussian: I component
    I_main = A_main * np.exp(-0.5 * ((t - t_shift_main) / sigma_main)**2)
    
    # DRAG component: Q component (derivative-coupled)
    # The DRAG term is proportional to the derivative of the envelope
    # d/dt of Gaussian = -(t - t0)/sigma^2 * Gaussian
    Q_drag = A_drag * (-(t - t_shift_drag) / sigma_drag**2) * np.exp(-0.5 * ((t - t_shift_drag) / sigma_drag)**2)
    
    # Combine
    I = I_main
    Q = Q_drag
    
    return I, Q

def objective(params):
    """Objective function for optimization."""
    I, Q = generate_pulse_from_params(params)
    
    # Check constraints quickly
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        return 1e6
    
    # Slew rate
    dI = np.diff(I, prepend=0.0)
    dQ = np.diff(Q, prepend=0.0)
    slew_in = np.sqrt(dI[0]**2 + dQ[0]**2) / DT
    if slew_in > SLEW_BOUND:
        return 1e6
    
    dI_mid = np.diff(I)
    dQ_mid = np.diff(Q)
    slew_mid = np.sqrt(dI_mid**2 + dQ_mid**2) / DT
    if np.max(slew_mid) > SLEW_BOUND:
        return 1e6
    
    slew_out = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slew_out > SLEW_BOUND:
        return 1e6
    
    # Compute score
    score, invalid = compute_score(I, Q)
    if invalid:
        return 1e6
    return score

def pulse(target_gate, hardware_model):
    """Main entry point.
    
    Returns an (80, 2) array of I and Q amplitudes.
    """
    if target_gate != 'X_pi':
        # Return zero pulse for unsupported gates
        return np.zeros((N_STEPS, 2))
    
    # Initial guess for parameters
    # Area of main Gaussian should be pi
    # Area = A_main * sigma_main * sqrt(2*pi)
    sigma_main = 40.0 / 6.0
    A_main_init = np.pi / (sigma_main * np.sqrt(2 * np.pi))
    
    # DRAG amplitude: typically small, proportional to anharmonicity
    # For a transmon with chi = -2*pi*0.25, the DRAG amplitude is roughly
    # A_drag ~ A_main * chi / (4*pi*f) ... let's start with a reasonable guess
    A_drag_init = -0.15  # Negative because anharmonicity is negative
    
    t_shift_main_init = DURATION / 2.0
    t_shift_drag_init = DURATION / 2.0
    
    x0 = [A_main_init, A_drag_init, t_shift_main_init, t_shift_drag_init]
    
    # Bounds
    bounds = [
        (0.01, 0.39),  # A_main
        (-0.3, 0.3),   # A_drag
        (5.0, 35.0),   # t_shift_main
        (5.0, 35.0),   # t_shift_drag
    ]
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      bounds=bounds,
                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    
    I, Q = generate_pulse_from_params(result.x)
    
    # Final constraint check
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        # Scale down
        scale = AMPLITUDE_BOUND / np.max(amp) * 0.99
        I *= scale
        Q *= scale
    
    # Ensure slew rate compliance
    # Simple check and clip if needed
    for i in range(N_STEPS):
        amp_i = np.sqrt(I[i]**2 + Q[i]**2)
        if amp_i > AMPLITUDE_BOUND:
            ratio = AMPLITUDE_BOUND / amp_i
            I[i] *= ratio
            Q[i] *= ratio
    
    # Recheck slew rate and adjust if necessary
    # This is a simple fallback; the optimizer should have handled this
    for i in range(N_STEPS):
        if i == 0:
            prev_I, prev_Q = 0.0, 0.0
        else:
            prev_I, prev_Q = I[i-1], Q[i-1]
        
        dI = I[i] - prev_I
        dQ = Q[i] - prev_Q
        diff = np.sqrt(dI**2 + dQ**2)
        max_diff = SLEW_BOUND * DT
        
        if diff > max_diff:
            ratio = max_diff / diff
            I[i] = prev_I + dI * ratio
            Q[i] = prev_Q + dQ * ratio
    
    # Check final step to zero
    last_amp = np.sqrt(I[-1]**2 + Q[-1]**2)
    max_last = SLEW_BOUND * DT
    if last_amp > max_last:
        ratio = max_last / last_amp
        I[-1] *= ratio
        Q[-1] *= ratio
    
    pulse_array = np.column_stack([I, Q])
    return pulse_array
