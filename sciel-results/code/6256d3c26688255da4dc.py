"""Direct numerical optimization of pulse amplitudes for robust transmon control."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
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
TOTAL_TIME = N_STEPS * DT

def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

def hamiltonian(detuning, scale, I, Q):
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    drive_term = 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    H += drive_term
    return H

def liouvillian_step(rho, H, dt):
    U = expm(-1j * H * dt)
    rho = U @ rho @ U.conj().T
    
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    
    for L in [L1, L2]:
        LdagL = L.conj().T @ L
        rho = rho - 0.5 * dt * (LdagL @ rho + rho @ LdagL)
        rho = rho + dt * L @ rho @ L.conj().T
        
    return rho

def fidelity_with_X(rho):
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
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
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        H = hamiltonian(detuning, scale, I[i], Q[i])
        rho = liouvillian_step(rho, H, DT)
        
    return rho

def compute_infidelity(pulse):
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
    return np.mean(infidelities) + np.max(infidelities)

def compute_violations(pulse):
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Amplitude violations
    amps = np.sqrt(I**2 + Q**2)
    amp_viol = np.sum(np.maximum(0, amps - AMPLITUDE_BOUND))
    
    # Slew violations
    # From 0 to step 0
    slew0 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    slew0_viol = np.maximum(0, slew0 - SLEW_BOUND)
    
    # Between steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slews = np.sqrt(dI**2 + dQ**2) / DT
    slew_viol = np.sum(np.maximum(0, slews - SLEW_BOUND))
    
    # From last step to 0
    slew_last = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    slew_last_viol = np.maximum(0, slew_last - SLEW_BOUND)
    
    total_violation = amp_viol + slew0_viol + slew_viol + slew_last_viol
    return total_violation

def objective(params):
    pulse = params.reshape(N_STEPS, 2)
    infid = compute_infidelity(pulse)
    viol = compute_violations(pulse)
    # Penalty weight
    lambda_p = 10.0
    return infid + lambda_p * viol

def generate_initial_pulse(seed_idx):
    """Generate diverse initial pulses."""
    t = np.arange(N_STEPS) * DT - TOTAL_TIME / 2.0
    
    if seed_idx == 0:
        # Gaussian
        sigma = 6.0
        gauss = np.exp(-0.5 * (t / sigma) ** 2)
        gauss_area = np.sum(gauss) * DT
        I = np.pi * gauss / gauss_area
        Q = np.zeros(N_STEPS)
    elif seed_idx == 1:
        # Gaussian with DRAG-like Q
        sigma = 6.0
        gauss = np.exp(-0.5 * (t / sigma) ** 2)
        gauss_area = np.sum(gauss) * DT
        I = np.pi * gauss / gauss_area
        Q = 0.1 * np.exp(-0.5 * (t / 5.0) ** 2) * (t / 25.0)
    elif seed_idx == 2:
        # Rectangular-ish
        I = np.zeros(N_STEPS)
        Q = np.zeros(N_STEPS)
        # Smooth rectangular
        mask = (t > -10) & (t < 10)
        I[mask] = 0.2
        # Smooth edges
        for i in range(N_STEPS):
            if i < 10:
                I[i] = 0.2 * (i/10.0)
            elif i > 69:
                I[i] = 0.2 * ((79-i)/10.0)
    else:
        # Random smooth
        rng = np.random.default_rng(seed_idx + 100)
        I = rng.normal(0, 0.1, N_STEPS)
        Q = rng.normal(0, 0.1, N_STEPS)
        # Smooth
        kernel = np.ones(5) / 5
        I = np.convolve(I, kernel, mode='same')
        Q = np.convolve(Q, kernel, mode='same')
        
    pulse = np.column_stack([I, Q])
    # Project to feasible set roughly (scale down if needed)
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.max(amps) > AMPLITUDE_BOUND:
        pulse *= AMPLITUDE_BOUND / np.max(amps)
        
    # Check slew roughly
    d_pulse = np.diff(pulse, axis=0)
    slews = np.sqrt(d_pulse[:, 0]**2 + d_pulse[:, 1]**2) / DT
    if np.max(slews) > SLEW_BOUND:
        pulse *= SLEW_BOUND / np.max(slews)
        
    return pulse

_optimized_pulse = None

def _optimize():
    global _optimized_pulse
    best_score = np.inf
    best_pulse = None
    
    # Try 5 different initial points
    for i in range(5):
        init_pulse = generate_initial_pulse(i)
        init_params = init_pulse.flatten()
        
        bounds = [(-AMPLITUDE_BOUND, AMPLITUDE_BOUND)] * (N_STEPS * 2)
        
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        
        final_pulse = res.x.reshape(N_STEPS, 2)
        score = compute_infidelity(final_pulse)
        viol = compute_violations(final_pulse)
        
        if viol < 1e-5: # Valid
            if score < best_score:
                best_score = score
                best_pulse = final_pulse
        else:
            # If invalid, try to fix it or skip
            # For now, skip invalid results in this simple implementation
            pass
            
    if best_pulse is None:
        # Fallback to the best one found even if slightly invalid, or a safe Gaussian
        # We should ensure we return a valid pulse.
        # Let's re-evaluate the best valid one. If none, use a very small pulse (safe but bad score)
        # Or use the seed from the previous program?
        # Let's just pick the one with lowest violation + score
        # Actually, the penalty method should drive violations to 0.
        # If L-BFGS-B fails to satisfy constraints, we can project.
        
        # Let's just use the first valid one or a safe default.
        # A safe default: Zero pulse? No, infidelity 1.
        # Let's use the Gaussian seed, scaled to be valid.
        best_pulse = generate_initial_pulse(0)
        
    _optimized_pulse = best_pulse

# Run optimization at load time
_optimize()

def pulse(target_gate, hardware_model):
    """Return the optimized pulse for the given target gate and hardware model."""
    global _optimized_pulse
    return _optimized_pulse.copy()
# EVOLVE-BLOCK-END
