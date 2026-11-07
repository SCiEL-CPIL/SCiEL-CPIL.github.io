"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T2 = 40000.0  # Tphi

N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Build ladder operators for 3-level system
def build_operators():
    """Build a, a_dagger, n for 3-level system."""
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
    drive_term = 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    H += drive_term
    return H

def liouvillian_step(rho, H, dt):
    """Apply one step of the Liouvillian evolution using matrix exponential."""
    # Coherent evolution
    U = expm(-1j * H * dt)
    rho = U @ rho @ U.conj().T
    
    # Incoherent evolution (T1 and Tphi)
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    
    # Apply Lindblad terms approximately for small dt
    for L in [L1, L2]:
        LdagL = L.conj().T @ L
        rho = rho - 0.5 * dt * (LdagL @ rho + rho @ LdagL)
        rho = rho + dt * L @ rho @ L.conj().T
    
    return rho

def fidelity_with_X(rho):
    """Compute mean fidelity to X gate over the 6 test states."""
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    fidelities = []
    for psi in states:
        psi_target = np.zeros(3, dtype=complex)
        psi_target[0] = psi[1]
        psi_target[1] = psi[0]
        psi_target[2] = psi[2]
        
        psi_final = rho @ psi
        overlap = np.vdot(psi_target, psi_final)
        fid = np.abs(overlap) ** 2
        fidelities.append(fid)
    
    return np.mean(fidelities)

def evolve_pulse(pulse, detuning, scale):
    """Evolve the system with the given pulse and return final density matrix."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        H = hamiltonian(detuning, scale, I[i], Q[i])
        rho = liouvillian_step(rho, H, DT)
    
    return rho

def compute_score(pulse):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            rho = evolve_pulse(pulse, det, sc)
            fid = fidelity_with_X(rho)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def check_validity(pulse):
    """Check if the pulse satisfies amplitude and slew constraints."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    slew0 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-10:
        return False
    
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
    
    slew_last = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slew_last > SLEW_BOUND + 1e-10:
        return False
    
    return True

def generate_pulse_from_params(params):
    """Generate a pulse from parameterized basis functions.
    
    Uses a combination of Gaussian-like envelopes with different widths
    and a DRAG-like correction term, all parameterized for optimization.
    """
    # params: [amp1, width1, amp2, width2, drag_amp, drag_width, phase]
    amp1, width1, amp2, width2, drag_amp, drag_width, phase = params
    
    t = np.arange(N_STEPS) * DT - TOTAL_TIME / 2.0
    
    # Two Gaussian components for the I channel
    g1 = amp1 * np.exp(-0.5 * (t / width1) ** 2)
    g2 = amp2 * np.exp(-0.5 * (t / width2) ** 2)
    I = g1 + g2
    
    # DRAG-like Q channel: derivative of a Gaussian
    # d/dt exp(-t^2/(2*drag_width^2)) = -(t/drag_width^2) * exp(...)
    q_gauss = np.exp(-0.5 * (t / drag_width) ** 2)
    Q = drag_amp * (t / (drag_width ** 2)) * q_gauss
    
    # Apply phase rotation to both I and Q
    c = np.cos(phase)
    s = np.sin(phase)
    I_rot = I * c - Q * s
    Q_rot = I * s + Q * c
    
    pulse = np.column_stack([I_rot, Q_rot])
    
    # Enforce constraints
    pulse = _enforce_constraints(pulse)
    
    return pulse

def _enforce_constraints(pulse):
    """Enforce amplitude and slew constraints by scaling."""
    # Amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        pulse *= AMPLITUDE_BOUND / max_amp
    
    # Slew from 0 to first step
    slew0 = np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2) / DT
    if slew0 > SLEW_BOUND:
        pulse *= SLEW_BOUND * DT / np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2)
    
    # Slew between steps
    d_pulse = np.diff(pulse, axis=0)
    slews = np.sqrt(d_pulse[:, 0]**2 + d_pulse[:, 1]**2) / DT
    max_slew = np.max(slews)
    if max_slew > SLEW_BOUND:
        pulse *= SLEW_BOUND / max_slew
    
    # Slew from last step to 0
    slew_last = np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2) / DT
    if slew_last > SLEW_BOUND:
        pulse *= SLEW_BOUND * DT / np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2)
    
    return pulse

def optimize_pulse():
    """Optimize the pulse using Nelder-Mead with a good initial guess."""
    def objective(params):
        pulse = generate_pulse_from_params(params)
        if not check_validity(pulse):
            return 1e6
        return compute_score(pulse)
    
    # Initial guess:
    # amp1 ~ 0.35, width1 ~ 6.0 (main Gaussian)
    # amp2 ~ 0.05, width2 ~ 3.0 (secondary for shaping)
    # drag_amp ~ 0.08, drag_width ~ 5.0 (DRAG correction)
    # phase ~ 0.0
    x0 = np.array([0.35, 6.0, 0.05, 3.0, 0.08, 5.0, 0.0])
    
    # Use Nelder-Mead for robustness
    from scipy.optimize import minimize
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                               'adaptive': True})
    
    # If Nelder-Mead didn't converge well, try a second pass with tighter bounds
    best_params = result.x
    best_score = result.fun
    
    # Try a few random restarts to find a better local minimum
    rng = np.random.default_rng(42)
    for _ in range(5):
        x_rand = x0 + rng.normal(0, 0.1, size=7)
        x_rand[1] = np.clip(x_rand[1], 2.0, 10.0)  # width1
        x_rand[3] = np.clip(x_rand[3], 1.0, 8.0)   # width2
        x_rand[5] = np.clip(x_rand[5], 2.0, 8.0)   # drag_width
        x_rand[0] = np.clip(x_rand[0], 0.1, 0.4)   # amp1
        x_rand[2] = np.clip(x_rand[2], -0.1, 0.1)  # amp2
        x_rand[4] = np.clip(x_rand[4], -0.2, 0.2)  # drag_amp
        x_rand[6] = np.clip(x_rand[6], -0.5, 0.5)  # phase
        
        res_rand = minimize(objective, x_rand, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
        if res_rand.fun < best_score:
            best_score = res_rand.fun
            best_params = res_rand.x
    
    return generate_pulse_from_params(best_params)

# Generate the optimized pulse at module load time
_optimized_pulse = optimize_pulse()

def pulse(target_gate, hardware_model):
    """Return the optimized pulse for the given target gate and hardware model."""
    global _optimized_pulse
    return _optimized_pulse.copy()
# EVOLVE-BLOCK-END
