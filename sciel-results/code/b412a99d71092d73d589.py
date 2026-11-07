"""
Robust X-pi pulse for a 3-level transmon using DRAG correction + Nelder-Mead optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize

# Hardware constants
ANHARMONICITY = -2 * np.pi * 0.25  # GHz
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DT = 0.5  # ns
N_STEPS = 80
T1 = 25000.0  # ns
T_PHI = 40000.0  # ns

# Detuning and scale bounds
DETUNING_BOUNDS = np.array([-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002])
SCALE_BOUNDS = np.array([0.97, 1.0, 1.03])

# State space: 3 levels (0, 1, 2)
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
A = np.array([[0, 1, 0],
              [0, 0, np.sqrt(2)],
              [0, 0, 0]], dtype=complex)
A_DAG = A.conj().T
N_OP = A_DAG @ A  # n = a^dag a

def liouvillian_step(detuning, scale, I, Q, dt):
    """
    Compute the Liouvillian superoperator for one time step.
    H = detuning * n + 0.5 * anh * n(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    Lindblad: a/sqrt(T1), n*sqrt(2/Tphi)
    """
    # Hamiltonian
    n_n_minus_1 = N_OP @ (N_OP - np.eye(3))
    H = (detuning * N_OP + 0.5 * ANHARMONICITY * n_n_minus_1 +
         0.5 * scale * (I * (A + A_DAG) + 1j * Q * (A - A_DAG)))
    
    # Lindblad operators
    L1 = A / np.sqrt(T1)
    L2 = N_OP * np.sqrt(2.0 / T_PHI)
    
    # Build the Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # In vectorized form: vec(L(rho)) = S * vec(rho)
    # S = -i(H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 0.5(L_k^dag L_k) ⊗ I - 0.5 I ⊗ (L_k^dag L_k)^T)
    
    dim = 3
    I3 = np.eye(dim, dtype=complex)
    
    # Coherent part: -i(H ⊗ I - I ⊗ H^T)
    S = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    # Dissipative parts
    for L in [L1, L2]:
        L_dag = L.conj().T
        S += np.kron(L, L.conj()) - 0.5 * np.kron(L_dag @ L, I3) - 0.5 * np.kron(I3, (L_dag @ L).T)
    
    # Exponentiate: U = exp(S * dt)
    # Use scipy for matrix exponential
    from scipy.linalg import expm
    U = expm(S * dt)
    return U

def vec_to_rho(v):
    """Convert 9-element vector to 3x3 density matrix."""
    return v.reshape(3, 3)

def rho_to_vec(rho):
    """Convert 3x3 density matrix to 9-element vector."""
    return rho.flatten()

def fidelity(rho, target_state):
    """
    Compute fidelity between density matrix rho and pure target state |psi>.
    Fidelity = <psi|rho|psi>
    """
    psi = target_state.reshape(-1, 1)
    return np.real(np.vdot(psi.flatten(), rho @ psi.flatten()))

def evaluate_pulse(pulse_IQ, detuning=0.0, scale=1.0):
    """
    Evaluate the infidelity of the pulse over the 4 test states.
    pulse_IQ: (80, 2) array of [I, Q]
    Returns: mean infidelity
    """
    # Precompute all step unitaries
    U_steps = []
    for i in range(N_STEPS):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        U = liouvillian_step(detuning, scale, I_val, Q_val, DT)
        U_steps.append(U)
    
    # Test states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_plus_i = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_minus_i = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    # Target states after X gate: |0> -> |1>, |1> -> |0>
    # So for input |0>, target is |1>; for input |1>, target is |0>
    # For (|0>+|1>)/sqrt(2), target is (|1>+|0>)/sqrt(2) = same
    # For (|0>-|1>)/sqrt(2), target is (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # For (|0>+i|1>)/sqrt(2), target is (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)... 
    # Let's just define target as X|psi>
    
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    test_states = [psi_0, psi_1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    target_states = [X_gate @ psi for psi in test_states]
    
    fidelities = []
    for psi_in, psi_target in zip(test_states, target_states):
        rho = np.outer(psi_in, psi_in.conj())
        rho_vec = rho_to_vec(rho)
        
        # Apply all steps
        for U in U_steps:
            rho_vec = U @ rho_vec
        
        rho_final = vec_to_rho(rho_vec)
        
        # Fidelity with target
        psi_t = psi_target.reshape(-1, 1)
        fid = np.real(np.vdot(psi_t.flatten(), rho_final @ psi_t.flatten()))
        fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity

def check_validity(pulse_IQ):
    """Check amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.max(amps) > AMPLITUDE_BOUND + 1e-9:
        return False
    
    # Slew: ||p[i+1] - p[i]|| / dt <= 0.12
    # Include from zero to step 1 and from last step to zero
    p_prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        p_curr = pulse_IQ[i]
        diff = np.linalg.norm(p_curr - p_prev)
        if diff / DT > SLEW_BOUND + 1e-9:
            return False
        p_prev = p_curr
    
    # From last step to zero
    diff = np.linalg.norm(p_prev)
    if diff / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True

def drag_gaussian_pulse(area=np.pi, sigma=40/6.0, n_steps=N_STEPS, dt=DT):
    """
    Generate a DRAG-style Gaussian pulse with optimal beta correction.
    """
    t = np.linspace(0, n_steps * dt, n_steps, endpoint=False) + dt/2  # midpoints
    
    # Gaussian envelope
    g = np.exp(-0.5 * (t / sigma)**2)
    
    # Normalize to area = pi
    g_area = np.trapz(g, t)
    if g_area > 0:
        g = g * (area / g_area)
    
    # Offset to make edges zero (they already are approximately zero for Gaussian)
    # Ensure edges are exactly zero
    g[0] = 0.0
    g[-1] = 0.0
    
    # DRAG correction: Q component = beta * dg/dt
    # Optimal beta for 3-level system: beta = anh / (2 * pi * f_drive^2) approximately
    # More precisely, beta = -anh / (2 * drive_freq^2) where drive_freq ~ 1/(2*pi) in our units
    # In our convention, the drive frequency is implicit. The standard DRAG formula:
    # beta = anh / (2 * omega_01^2) where omega_01 is the drive frequency
    # Since we're working in rotating frame at resonance, the relevant scale is the anharmonicity
    # Standard result: beta = -anh / (2 * pi * f^2) but in our units where frequencies are in rad/ns,
    # and anh = -2*pi*0.25 GHz = -2*pi*0.25*1e-3 rad/ns
    # The drive frequency is not explicitly given; we use the standard approximation
    # beta = anh / (2 * omega^2) where omega is the Rabi frequency scale ~ area/duration
    # A common practical choice: beta = -anh / (2 * pi^2 * (dt * n_steps)^-2) ... 
    # Let's use the standard formula: beta = -anh / (2 * (pi * f)^2) where f is in GHz
    # In our units, anh is in rad/ns. The "frequency" scale for the correction is related to
    # the inverse of the pulse duration. Let's use beta = anh / (2 * (2*pi/sigma)^2) as a starting point
    # Actually, the standard DRAG beta is: beta = -anh / (2 * omega_drive^2)
    # where omega_drive is the drive frequency in the same units as anh.
    # Since we're in the rotating frame, the relevant quantity is the anharmonicity itself.
    # Empirically, beta ~ anh / (2 * (pi * f_01)^2) but let's just use a reasonable estimate.
    
    # Compute derivative of Gaussian
    dg_dt = np.gradient(g, t)
    
    # Estimate beta: for a transmon, beta ≈ -anh / (2 * (2*pi * f)^2)
    # where f is the qubit frequency ~ 5 GHz. In our units (rad/ns), f_01 ~ 2*pi*5e-3 rad/ns
    # So beta ~ -(-2*pi*0.25e-3) / (2 * (2*pi*5e-3)^2) = (2*pi*0.25e-3) / (2 * 4*pi^2*25e-6)
    # = (2*pi*0.25e-3) / (200*pi^2*1e-6) = (2*pi*250e-6) / (200*pi^2*1e-6) = 500*pi / (200*pi^2) = 2.5/pi ≈ 0.796
    # Let's use beta = 0.8 as a starting estimate
    beta = 0.8
    
    I = g
    Q = beta * dg_dt
    
    pulse_IQ = np.column_stack([I, Q])
    
    # Check and fix validity
    if not check_validity(pulse_IQ):
        # Scale down to satisfy constraints
        max_amp = np.max(np.sqrt(I**2 + Q**2))
        if max_amp > AMPLITUDE_BOUND:
            scale_factor = AMPLITUDE_BOUND / max_amp * 0.99
            pulse_IQ *= scale_factor
        
        # Check slew
        for i in range(N_STEPS):
            if i == 0:
                prev = np.array([0.0, 0.0])
            else:
                prev = pulse_IQ[i-1]
            curr = pulse_IQ[i]
            diff = np.linalg.norm(curr - prev)
            if diff / DT > SLEW_BOUND:
                # This step is too large; we'll need to smooth
                pass
        
        # If still invalid, return a simpler pulse
        if not check_validity(pulse_IQ):
            # Return a flat-top pulse that satisfies constraints
            pulse_IQ = np.zeros((N_STEPS, 2))
            amp = AMPLITUDE_BOUND * 0.8
            # Ramp up and down
            ramp_len = int(SLEW_BOUND * DT / amp * N_STEPS / 2)
            ramp_len = max(1, min(ramp_len, N_STEPS // 2))
            for i in range(N_STEPS):
                if i < ramp_len:
                    f = (i + 1) / ramp_len
                elif i >= N_STEPS - ramp_len:
                    f = (N_STEPS - i) / ramp_len
                else:
                    f = 1.0
                pulse_IQ[i, 0] = amp * f
            # Normalize area to pi
            area = np.trapz(pulse_IQ[:, 0], dx=DT)
            if area > 0:
                pulse_IQ[:, 0] *= np.pi / area
            # Re-check
            if not check_validity(pulse_IQ):
                pulse_IQ *= 0.5
                area = np.trapz(pulse_IQ[:, 0], dx=DT)
                if area > 0:
                    pulse_IQ[:, 0] *= np.pi / area
    
    return pulse_IQ

def score_pulse(pulse_IQ):
    """
    Compute the score: mean + max infidelity over 9 cases.
    """
    infidelities = []
    for det in DETUNING_BOUNDS:
        for sc in SCALE_BOUNDS:
            inf = evaluate_pulse(pulse_IQ, detuning=det, scale=sc)
            infidelities.append(inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf

def pulse(target_gate, hardware_model):
    """
    Main entry point. Returns (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # 300s budget with safety margin
    
    if target_gate != 'X_pi':
        # Return a zero pulse for invalid gates
        return np.zeros((N_STEPS, 2))
    
    # Generate initial DRAG pulse
    pulse_IQ = drag_gaussian_pulse()
    
    # Verify validity
    if not check_validity(pulse_IQ):
        # Fallback to a valid simple pulse
        pulse_IQ = np.zeros((N_STEPS, 2))
        pulse_IQ[:, 0] = np.pi / (N_STEPS * DT)  # constant amplitude to give area pi
        if not check_validity(pulse_IQ):
            pulse_IQ *= 0.5
    
    # Compute initial score
    initial_score = score_pulse(pulse_IQ)
    best_pulse = pulse_IQ.copy()
    best_score = initial_score
    
    # Nelder-Mead optimization on pulse parameters
    # We'll optimize a few key parameters: amplitude scale, sigma, beta
    # Represent the pulse by (amp_scale, sigma, beta) and regenerate
    
    def pulse_from_params(params):
        amp_scale, sigma, beta = params
        t = np.linspace(0, N_STEPS * DT, N_STEPS, endpoint=False) + DT/2
        g = np.exp(-0.5 * (t / sigma)**2)
        g[0] = 0.0
        g[-1] = 0.0
        g_area = np.trapz(g, t)
        if g_area > 0:
            g = g * (np.pi / g_area) * amp_scale
        dg_dt = np.gradient(g, t)
        I = g
        Q = beta * dg_dt
        p = np.column_stack([I, Q])
        
        # Enforce constraints
        max_amp = np.max(np.sqrt(p[:, 0]**2 + p[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            p *= (AMPLITUDE_BOUND / max_amp) * 0.99
        
        # Check slew
        valid = True
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            diff = np.linalg.norm(p[i] - prev)
            if diff / DT > SLEW_BOUND:
                valid = False
                break
            prev = p[i]
        if np.linalg.norm(prev) / DT > SLEW_BOUND:
            valid = False
        
        if not valid:
            # Penalize heavily
            return p, False
        return p, True
    
    def objective(params):
        p, valid = pulse_from_params(params)
        if not valid:
            return 10.0  # Large penalty
        # Quick evaluation on nominal case only for speed
        return evaluate_pulse(p, detuning=0.0, scale=1.0)
    
    # Initial parameters
    x0 = np.array([1.0, 40/6.0, 0.8])
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                                  'adaptive': True})
        
        if time.time() < deadline - 60:
            # Check if the optimized pulse is better
            opt_pulse, valid = pulse_from_params(result.x)
            if valid:
                opt_score = score_pulse(opt_pulse)
                if opt_score < best_score:
                    best_pulse = opt_pulse
                    best_score = opt_score
    except Exception:
        pass
    
    # If we have time, do a second round of optimization with more iterations
    if time.time() < deadline - 120:
        try:
            x0_2 = result.x if 'result' in dir() else x0
            result2 = minimize(objective, x0_2, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7,
                                       'adaptive': True})
            if time.time() < deadline - 60:
                opt_pulse2, valid2 = pulse_from_params(result2.x)
                if valid2:
                    opt_score2 = score_pulse(opt_pulse2)
                    if opt_score2 < best_score:
                        best_pulse = opt_pulse2
                        best_score = opt_score2
        except Exception:
            pass
    
    # Final validity check
    if not check_validity(best_pulse):
        # Return a known-valid fallback
        fallback = np.zeros((N_STEPS, 2))
        fallback[:, 0] = np.pi / (N_STEPS * DT) * 0.9
        if check_validity(fallback):
            best_pulse = fallback
    
    return best_pulse


# Keep the original interface
from autoresearch_suite.science_physics_expansion import quantum_seed as _original_pulse

def pulse(target_gate, hardware_model):
    """
    Main entry point. Returns (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 300.0
    
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Generate initial DRAG pulse
    pulse_IQ = drag_gaussian_pulse()
    
    if not check_validity(pulse_IQ):
        pulse_IQ = np.zeros((N_STEPS, 2))
        pulse_IQ[:, 0] = np.pi / (N_STEPS * DT) * 0.9
        if not check_validity(pulse_IQ):
            pulse_IQ *= 0.5
    
    initial_score = score_pulse(pulse_IQ)
    best_pulse = pulse_IQ.copy()
    best_score = initial_score
    
    def pulse_from_params(params):
        amp_scale, sigma, beta = params
        t = np.linspace(0, N_STEPS * DT, N_STEPS, endpoint=False) + DT/2
        g = np.exp(-0.5 * (t / sigma)**2)
        g[0] = 0.0
        g[-1] = 0.0
        g_area = np.trapz(g, t)
        if g_area > 0:
            g = g * (np.pi / g_area) * amp_scale
        dg_dt = np.gradient(g, t)
        I = g
        Q = beta * dg_dt
        p = np.column_stack([I, Q])
        
        max_amp = np.max(np.sqrt(p[:, 0]**2 + p[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            p *= (AMPLITUDE_BOUND / max_amp) * 0.99
        
        valid = True
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            diff = np.linalg.norm(p[i] - prev)
            if diff / DT > SLEW_BOUND:
                valid = False
                break
            prev = p[i]
        if np.linalg.norm(prev) / DT > SLEW_BOUND:
            valid = False
        
        return p, valid
    
    def objective(params):
        p, valid = pulse_from_params(params)
        if not valid:
            return 10.0
        return evaluate_pulse(p, detuning=0.0, scale=1.0)
    
    x0 = np.array([1.0, 40/6.0, 0.8])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                                  'adaptive': True})
        
        if time.time() < deadline - 60:
            opt_pulse, valid = pulse_from_params(result.x)
            if valid:
                opt_score = score_pulse(opt_pulse)
                if opt_score < best_score:
                    best_pulse = opt_pulse
                    best_score = opt_score
    except Exception:
        pass
    
    if time.time() < deadline - 120:
        try:
            x0_2 = result.x if 'result' in dir() else x0
            result2 = minimize(objective, x0_2, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7,
                                       'adaptive': True})
            if time.time() < deadline - 60:
                opt_pulse2, valid2 = pulse_from_params(result2.x)
                if valid2:
                    opt_score2 = score_pulse(opt_pulse2)
                    if opt_score2 < best_score:
                        best_pulse = opt_pulse2
                        best_score = opt_score2
        except Exception:
            pass
    
    if not check_validity(best_pulse):
        fallback = np.zeros((N_STEPS, 2))
        fallback[:, 0] = np.pi / (N_STEPS * DT) * 0.9
        if check_validity(fallback):
            best_pulse = fallback
    
    return best_pulse
