"""Robust DRAG-style pulse for three-level transmon X-pi gate."""
import numpy as np
from scipy.integrate import solve_ivp
import time

# Constants
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns

# Precompute basis operators
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T
n_op = a_dag @ a_op

# Lindblad operators
L1 = a_op / np.sqrt(T1)  # amplitude damping
L2 = n_op * np.sqrt(2 / Tphi)  # pure dephasing

def build_hamiltonian(detuning, scale, I, Q):
    """Build the Hamiltonian for given parameters."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
    return H

def liouvillian_step(rho, H, dt):
    """Apply one Liouvillian step."""
    # Commutator term: -i[H, rho]
    comm = -1j * (H @ rho - rho @ H)
    # Dissipative terms
    diss = 0.0
    for L in [L1, L2]:
        Ldag = L.conj().T
        diss += 2 * (L @ rho @ Ldag - 0.5 * (Ldag @ L @ rho + rho @ Ldag @ L))
    # First-order update (small dt)
    rho_new = rho + dt * (comm + diss)
    # Renormalize trace
    tr = np.trace(rho_new)
    if abs(tr) > 1e-15:
        rho_new = rho_new / tr
    return rho_new

def simulate_pulse(pulse_I, pulse_Q, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # start in |0>
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        H = build_hamiltonian(detuning, scale, I_val, Q_val)
        rho = liouvillian_step(rho, H, DT)
    
    return rho

def compute_fidelity(rho, target_state):
    """Compute fidelity between density matrix and target state."""
    target_pure = np.outer(target_state, target_state.conj())
    fid = np.real(np.trace(rho @ target_pure))
    return max(0.0, min(1.0, fid))

def score_pulse(pulse_I, pulse_Q):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # X gate maps |0> -> |1>, |1> -> |0>
    # So target for input |0> is |1>, input |1> is |0>
    # For superpositions, X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2) = same
    # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
    # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
        np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+|1>)/sqrt(2)
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # X(|0>-|1>)/sqrt(2)
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # X(|0>+i|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # X(|0>-i|1>)/sqrt(2)
    ]
    
    fidelities = []
    for det in detunings:
        for sc in scales:
            rho = simulate_pulse(pulse_I, pulse_Q, det, sc)
            for ts in target_states:
                fid = compute_fidelity(rho, ts)
                fidelities.append(fid)
    
    fidelities = np.array(fidelities)
    infidelities = 1.0 - fidelities
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf

def check_validity(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
    amplitudes = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND + 1e-9:
        return False
    
    # Slew rate: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # Include from zero to step 1 and last step to zero
    for i in range(N_STEPS):
        if i == 0:
            dI = pulse_I[0]
            dQ = pulse_Q[0]
        else:
            dI = pulse_I[i] - pulse_I[i-1]
            dQ = pulse_Q[i] - pulse_Q[i-1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-9:
            return False
    
    # Last step to zero
    dI = -pulse_I[-1]
    dQ = -pulse_Q[-1]
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-9:
        return False
    
    return True

def generate_seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, DURATION, N_STEPS)
    sigma = 40 / 6  # ~6.67 ns
    
    # Gaussian envelope with area pi
    # Integral of A * exp(-(t-t0)^2/(2*sigma^2)) dt from -inf to inf = A * sigma * sqrt(2*pi)
    # We want area = pi, so A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    A = np.sqrt(np.pi / 2) / sigma
    
    I = A * np.exp(-(t - DURATION/2)**2 / (2 * sigma**2))
    
    # DRAG component: derivative of envelope
    dI_dt = -A * (t - DURATION/2) / sigma**2 * np.exp(-(t - DURATION/2)**2 / (2 * sigma**2))
    
    # DRAG amplitude to cancel leakage
    # For a transmon with anharmonicity alpha, the DRAG amplitude is approximately:
    # Q = -alpha * dI/dt / (2 * pi * f01) where f01 is the 0-1 transition frequency
    # But since we're working in a rotating frame, we need to be careful.
    # The standard DRAG formula: Q(t) = -alpha * sigma / (2*pi) * dI/dt(t)
    # Let's use a reasonable DRAG coefficient
    drag_coeff = ANHARMONICITY / (2 * np.pi)  # This is -0.25
    
    Q = drag_coeff * dI_dt
    
    # Scale to ensure constraints are met
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / np.max(amplitudes)
        I *= scale_factor
        Q *= scale_factor
    
    # Check slew rate
    for i in range(N_STEPS):
        if i == 0:
            dI = I[0]
            dQ = Q[0]
        else:
            dI = I[i] - I[i-1]
            dQ = Q[i] - Q[i-1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            # Scale down
            scale_factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            I[i:] *= scale_factor
            Q[i:] *= scale_factor
            break
    
    # Ensure area is approximately pi (for the I component)
    area = np.trapz(I, dx=DT)
    if abs(area) > 1e-6:
        I *= np.pi / area
        Q *= np.pi / area
    
    return I, Q

def optimize_pulse(I0, Q0):
    """Optimize the pulse to minimize infidelity."""
    def objective(params):
        I = params[:N_STEPS]
        Q = params[N_STEPS:]
        
        if not check_validity(I, Q):
            return 1e10
        
        return score_pulse(I, Q)
    
    # Use Nelder-Mead for optimization
    from scipy.optimize import minimize
    
    x0 = np.concatenate([I0, Q0])
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.success:
        I_opt = result.x[:N_STEPS]
        Q_opt = result.x[N_STEPS:]
        if check_validity(I_opt, Q_opt):
            return I_opt, Q_opt
    
    return I0, Q0

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds
    
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Generate seed pulse
    I_seed, Q_seed = generate_seed_pulse()
    
    # Check validity of seed
    if not check_validity(I_seed, Q_seed):
        # Try to fix by scaling
        amplitudes = np.sqrt(I_seed**2 + Q_seed**2)
        max_amp = np.max(amplitudes)
        if max_amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / max_amp
            I_seed *= scale
            Q_seed *= scale
    
    # Compute seed score
    seed_score = score_pulse(I_seed, Q_seed)
    
    # Try to optimize if we have time
    if time.time() < deadline - 60:  # Leave 60 seconds margin
        try:
            I_opt, Q_opt = optimize_pulse(I_seed, Q_seed)
            opt_score = score_pulse(I_opt, Q_opt)
            
            if opt_score < seed_score and check_validity(I_opt, Q_opt):
                I_seed, Q_seed = I_opt, Q_seed
                seed_score = opt_score
        except Exception:
            pass
    
    # Final validity check
    if not check_validity(I_seed, Q_seed):
        # Fallback: return zero pulse (invalid but won't crash)
        return np.zeros((80, 2))
    
    # Return as (80, 2) array
    return np.column_stack([I_seed, Q_seed])

# Keep the function name as expected
pulse = pulse
