"""
Rewritten program: frequency-domain spectral shaping with projected gradient descent
for robust X-pi gate in a three-level transmon.

Strategy:
- Define pulse in frequency domain as a Gaussian envelope centered at the 0↔1 transition
  with an adjustable notch (second Gaussian subtracted) to suppress 1↔2 leakage.
- Inverse FFT to get time-domain I, Q.
- Apply amplitude and slew constraints via projection.
- Optimize spectral parameters (center, width, notch depth, notch width, phase)
  using projected gradient descent on the infidelity cost.
- Precompute Liouvillian base terms for speed.
- Deadline guard: track elapsed time, return best found if approaching limit.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Constants
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
Tphi = 40000.0
DT = 0.5
N_STEPS = 80
DURATION = 40.0

# Time grid: step centers
t_grid = (np.arange(N_STEPS) + 0.5) * DT  # 0.25 to 39.75 ns

# Basis operators (3x3)
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T
n_op = a_dag @ a_op  # diag(0, 1, 2)

# Precompute Liouvillian base terms
# H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
# For each (detuning, scale), build the Hamiltonian and Lindblad superoperators.

# Lindblad operators
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / Tphi)

def _comm_super(op):
    """Superoperator for commutator [op, .]"""
    return np.kron(op, np.eye(3)) - np.kron(np.eye(3), op.T)

def _dissipative_super(L):
    """Dissipative superoperator for Lindblad L: L rho L^dag - 0.5(L^dag L rho + rho L^dag L)"""
    LdagL = L.conj().T @ L
    return (np.kron(L, L.conj().T) - 0.5 * (np.kron(LdagL, np.eye(3)) + np.kron(np.eye(3), LdagL.T)))

# Precompute superoperators for the constant parts
S_n = _comm_super(n_op)  # detuning term
S_nnm1 = _comm_super(n_op @ (n_op - np.eye(3)))  # anharmonicity term
S_a = _comm_super(a_op)  # I drive: 0.5*(a + a_dag)
S_adag = _comm_super(a_dag)
S_a_minus_adag = _comm_super(a_op - a_dag)  # for i*Q term
D1 = _dissipative_super(L1)
D2 = _dissipative_super(L2)

# Precompute the constant part of Liouvillian (detuning + anharmonicity + dissipation)
# Liouvillian = detuning*S_n + 0.5*anh*S_nnm1 + 0.5*scale*(I*(S_a+S_adag) + i*Q*S_a_minus_adag) + D1 + D2
# Note: the drive part is time-dependent, so we build it per step.

def _build_hamiltonian(detuning, scale, I, Q):
    """Build Hamiltonian for given parameters."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * (n_op @ (n_op - np.eye(3)))
    H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
    return H

def _liouvillian_super(detuning, scale, I, Q):
    """Build full Liouvillian superoperator."""
    H = _build_hamiltonian(detuning, scale, I, Q)
    L = _comm_super(H) + D1 + D2
    return L

# Precompute the constant Liouvillian part (no drive)
L_const_base = D1 + D2  # dissipation only, since detuning/anhar are also constant per case

def _liouvillian_with_drive(detuning, scale, I, Q):
    """Build Liouvillian efficiently using precomputed parts."""
    # Constant part: detuning*S_n + 0.5*anh*S_nnm1 + D1 + D2
    L = detuning * S_n + 0.5 * ANHARMONICITY * S_nnm1 + D1 + D2
    # Drive part: 0.5*scale*(I*(S_a+S_adag) + i*Q*S_a_minus_adag)
    L += 0.5 * scale * (I * (S_a + S_adag) + 1j * Q * S_a_minus_adag)
    return L

def _vec_rho(rho):
    """Vectorize density matrix (column-major)."""
    return rho.flatten(order='F')

def _unvec_rho(vec):
    """Unvectorize."""
    return vec.reshape((3, 3), order='F')

def _simulate_pulse(I, Q, detuning=0.0, scale=1.0):
    """
    Simulate the pulse and return the final density matrix.
    I, Q: arrays of length N_STEPS.
    """
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    rho_vec = _vec_rho(rho)
    
    for i in range(N_STEPS):
        L = _liouvillian_with_drive(detuning, scale, I[i], Q[i])
        # Exponentiate: rho_{i+1} = exp(L * dt) @ rho_i
        # Use matrix exponential via eigendecomposition or scipy
        from scipy.linalg import expm
        Ldt = L * DT
        U = expm(Ldt)
        rho_vec = U @ rho_vec
    
    return _unvec_rho(rho_vec)

def _fidelity_to_X(rho):
    """
    Compute mean fidelity to X gate over the 5 test states.
    X gate maps |0><0| -> |1><1|, etc.
    Fidelity = Tr(rho_target^dag @ rho) for pure target, but for mixed we use Uhlmann.
    For pure target |psi>, fidelity = <psi|rho|psi>.
    """
    # Target states after X:
    # |0> -> |1>, |1> -> |0>, (|0>+|1>)/sqrt2 -> (|1>+|0>)/sqrt2, etc.
    # Actually, X |psi> for each input state.
    # Input states: |0>, |1>, (|0>±|1>)/sqrt2, (|0>±i|1>)/sqrt2
    # After X: |1>, |0>, (|1>±|0>)/sqrt2, (|1>±i|0>)/sqrt2
    
    # Fidelity for pure target |phi> is <phi|rho|phi>
    states_after_X = [
        np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
        np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+|1>)/sqrt2
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # X(|0>-|1>)/sqrt2
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # X(|0>+i|1>)/sqrt2
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # X(|0>-i|1>)/sqrt2
    ]
    
    fidelities = []
    for psi in states_after_X:
        f = np.real(np.vdot(psi, rho @ psi))
        fidelities.append(f)
    return np.mean(fidelities)

def _compute_score(I, Q):
    """
    Compute the score: mean + max infidelity over 9 cases.
    """
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for d in detunings:
        for s in scales:
            rho = _simulate_pulse(I, Q, d, s)
            fid = _fidelity_to_X(rho)
            infidelities.append(1.0 - fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _check_validity(I, Q):
    """Check amplitude and slew constraints."""
    # Amplitude
    amp = np.sqrt(I**2 + Q**2)
    if np.any(amp > AMPLITUDE_BOUND + 1e-9):
        return False
    # Slew: from zero to step 0, between steps, from last step to zero
    p = np.column_stack([I, Q])
    p_prev = np.zeros((2,))
    for i in range(N_STEPS):
        diff = np.linalg.norm(p[i] - p_prev) / DT
        if diff > SLEW_BOUND + 1e-9:
            return False
        p_prev = p[i]
    # Last step to zero
    diff = np.linalg.norm(p[-1] - np.zeros(2)) / DT
    if diff > SLEW_BOUND + 1e-9:
        return False
    return True

def _project_to_constraints(I, Q):
    """
    Project I, Q to satisfy amplitude and slew constraints.
    Simple approach: clip amplitude, then smooth slew.
    """
    I = I.copy()
    Q = Q.copy()
    
    # Amplitude constraint: scale down if exceeds bound
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amp[mask]
        I[mask] *= scale_factor
        Q[mask] *= scale_factor
    
    # Slew constraint: iterative smoothing
    # For simplicity, apply a simple low-pass to reduce slew
    # This is a heuristic; full projection is complex.
    for _ in range(5):
        p = np.column_stack([I, Q])
        # Check slew from zero
        p_ext = np.vstack([np.zeros((1, 2)), p, np.zeros((1, 2))])
        diffs = np.diff(p_ext, axis=0) / DT
        norms = np.linalg.norm(diffs, axis=1)
        # Where slew exceeds bound, reduce the step
        for i in range(len(norms)):
            if norms[i] > SLEW_BOUND:
                # Reduce the endpoint
                if i == 0:
                    # Reduce p[0]
                    factor = SLEW_BOUND / norms[i]
                    p[0] = p[0] * factor
                elif i == len(norms) - 1:
                    # Reduce p[-1]
                    factor = SLEW_BOUND / norms[i]
                    p[-1] = p[-1] * factor
                else:
                    # Reduce the step between p[i] and p[i+1]
                    # Simple: average them
                    p[i] = 0.5 * (p[i] + p[i+1])
                    p[i+1] = p[i]
        I = p[:, 0]
        Q = p[:, 1]
    
    return I, Q

def _spectral_to_time(params):
    """
    Convert spectral parameters to time-domain I, Q.
    params: [center_freq, width, notch_depth, notch_width, phase, q_phase]
    - center_freq: center of main Gaussian (rad/ns)
    - width: std of main Gaussian
    - notch_depth: depth of notch (0 to 1)
    - notch_width: std of notch Gaussian
    - phase: phase of I component
    - q_phase: phase of Q component (for DRAG)
    """
    center, width, notch_depth, notch_width, phase, q_phase = params
    
    # Frequency grid for FFT
    N_freq = N_STEPS
    freqs = 2 * np.pi * np.fft.fftfreq(N_freq, d=DT)
    
    # Main Gaussian spectrum
    S_main = np.exp(-0.5 * ((freqs - center) / width) ** 2)
    
    # Notch Gaussian (centered at 1->2 transition)
    # 1->2 transition is at omega_12 = -anharmonicity = 2*pi*0.25
    omega_12 = -ANHARMONICITY
    S_notch = np.exp(-0.5 * ((freqs - omega_12) / notch_width) ** 2)
    
    S = S_main - notch_depth * S_notch
    
    # Ensure non-negative? No, allow negative for phase control.
    # Apply phase
    S_I = S * np.cos(phase)
    S_Q = S * np.sin(q_phase)
    
    # Inverse FFT to time domain
    I = np.real(np.fft.ifft(S_I)) * N_freq * DT
    Q = np.real(np.fft.ifft(S_Q)) * N_freq * DT
    
    # Normalize area to pi
    area_I = np.sum(I) * DT
    area_Q = np.sum(Q) * DT
    area = np.sqrt(area_I**2 + area_Q**2)
    if area > 1e-10:
        scale = np.pi / area
        I *= scale
        Q *= scale
    
    # Offset to ensure zero at edges (approximately)
    # Subtract linear trend
    t = np.arange(N_STEPS)
    if N_STEPS > 1:
        I = I - np.polyval(np.polyfit(t, I, 1), t)
        Q = Q - np.polyval(np.polyfit(t, Q, 1), t)
    
    return I, Q

def _cost(params):
    """Cost function for optimization."""
    I, Q = _spectral_to_time(params)
    I, Q = _project_to_constraints(I, Q)
    if not _check_validity(I, Q):
        return 1e6
    return _compute_score(I, Q)

def pulse(target_gate, hardware_model):
    """
    Main entry point. Returns (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    DEADLINE = 300.0  # seconds
    
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gate
        return np.zeros((N_STEPS, 2))
    
    # Initial guess: DRAG-style Gaussian in time domain
    # sigma = 40/6 ns
    sigma_t = DURATION / 6.0
    t = np.arange(N_STEPS) * DT
    t_center = DURATION / 2.0
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - t_center) / sigma_t) ** 2)
    
    # Offset to zero at edges
    env = env - env[0]
    env = env - env[-1]
    env = env - 0.5 * (env[0] + env[-1])
    
    # Area = pi
    area = np.sum(env) * DT
    if area > 1e-10:
        env *= np.pi / area
    
    # DRAG: add Q component proportional to derivative of I
    dI = np.gradient(env, DT)
    Q_drag = -0.5 * dI  # DRAG coefficient ~ -0.5 for anharmonicity
    
    I_init = env
    Q_init = Q_drag
    
    # Project to constraints
    I_init, Q_init = _project_to_constraints(I_init, Q_init)
    
    # Initial score
    best_I = I_init.copy()
    best_Q = Q_init.copy()
    best_score = _compute_score(best_I, best_Q)
    
    # Spectral parameter optimization
    # Initial spectral params: center at 0, width ~ pi/sigma_t, notch at omega_12
    omega_12 = -ANHARMONICITY
    width_init = np.pi / sigma_t
    params_init = np.array([0.0, width_init, 0.5, width_init / 2, 0.0, 0.0])
    
    # Optimize spectral parameters
    try:
        result = minimize(_cost, params_init, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6,
                                  'adaptive': True})
        
        if result.fun < best_score:
            I_opt, Q_opt = _spectral_to_time(result.x)
            I_opt, Q_opt = _project_to_constraints(I_opt, Q_opt)
            if _check_validity(I_opt, Q_opt):
                score_opt = _compute_score(I_opt, Q_opt)
                if score_opt < best_score:
                    best_I = I_opt
                    best_Q = Q_opt
                    best_score = score_opt
    except Exception:
        pass
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        # Return best found
        return np.column_stack([best_I, best_Q])
    
    # Try additional optimization with different initial conditions
    for trial in range(3):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Perturb initial params
        params_trial = params_init + 0.1 * np.random.randn(len(params_init))
        
        try:
            result = minimize(_cost, params_trial, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6,
                                      'adaptive': True})
            
            if result.fun < best_score:
                I_opt, Q_opt = _spectral_to_time(result.x)
                I_opt, Q_opt = _project_to_constraints(I_opt, Q_opt)
                if _check_validity(I_opt, Q_opt):
                    score_opt = _compute_score(I_opt, Q_opt)
                    if score_opt < best_score:
                        best_I = I_opt
                        best_Q = Q_opt
                        best_score = score_opt
        except Exception:
            pass
    
    return np.column_stack([best_I, best_Q])
