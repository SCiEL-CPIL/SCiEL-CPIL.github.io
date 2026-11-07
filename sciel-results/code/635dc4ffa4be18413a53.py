"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5

def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _liouvillian_superoperator(I, Q, detuning, scale, a, a_dag, n):
    """Build the Liouvillian superoperator for one step."""
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    I3 = np.eye(3, dtype=complex)
    
    L_ham = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    L_total = L_ham.copy()
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_total += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag @ L, I3) + np.kron(I3, (L_dag @ L).T))
    
    return L_total

def _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n):
    """Simulate the pulse and return mean fidelity over 6 input states."""
    # Precompute Liouvillian superoperators for each step
    superops = []
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        superops.append(U_super)
    
    # Input states
    input_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    fidelities = []
    for psi_in in input_states:
        # Target state after X gate
        psi_target = np.zeros(3, dtype=complex)
        psi_target[0] = psi_in[1]
        psi_target[1] = psi_in[0]
        
        # Evolve density matrix
        rho = np.outer(psi_in, psi_in.conj())
        rho_vec = rho.flatten(order='F')
        
        for i in range(N_STEPS):
            rho_vec = superops[i] @ rho_vec
        
        rho = rho_vec.reshape((3, 3), order='F')
        fidelity = np.real(np.conj(psi_target) @ rho @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False
    
    return True

def _compute_score(pulse, a, a_dag, n):
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _generate_pulse(sigma_factor=1.0, beta=0.0, amplitude_scale=1.0):
    """Generate a DRAG-style pulse with tunable parameters."""
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    sigma = (40.0 / 6.0) * sigma_factor
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    # DRAG: add a derivative term
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss[1:] - gauss[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I = gauss * amplitude_scale
    Q = beta * dgauss * amplitude_scale
    
    pulse = np.column_stack([I, Q])
    
    # Ensure amplitude constraint
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    for i in range(N_STEPS):
        if amps[i] > AMPLITUDE_BOUND:
            scale_factor = AMPLITUDE_BOUND / amps[i]
            pulse[i] *= scale_factor
    
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 290.0  # 290 second limit (10s safety margin)
    
    a, a_dag, n = _build_matrices()
    
    # Start with DRAG seed
    best_pulse = _generate_pulse(sigma_factor=1.0, beta=0.0, amplitude_scale=1.0)
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Try different parameter combinations
    sigma_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
    amplitude_scales = [0.9, 1.0, 1.1]
    
    for sf in sigma_factors:
        for b in betas:
            for asc in amplitude_scales:
                if time.time() > deadline - 60.0:
                    break
                
                p = _generate_pulse(sigma_factor=sf, beta=b, amplitude_scale=asc)
                if not _check_validity(p):
                    continue
                
                score = _compute_score(p, a, a_dag, n)
                if score < best_score:
                    best_score = score
                    best_pulse = p.copy()
    
    # Local optimization: try small perturbations
    rng = np.random.RandomState(42)
    
    for iteration in range(500):
        if time.time() > deadline - 60.0:
            break
        
        # Generate a perturbation
        perturbation = rng.normal(0, 0.01, size=(N_STEPS, 2))
        
        # Apply perturbation
        new_pulse = best_pulse + perturbation
        
        # Check validity
        if not _check_validity(new_pulse):
            continue
        
        score = _compute_score(new_pulse, a, a_dag, n)
        if score < best_score:
            best_score = score
            best_pulse = new_pulse.copy()
    
    return best_pulse
# EVOLVE-BLOCK-END
