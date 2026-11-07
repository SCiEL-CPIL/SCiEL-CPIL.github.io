import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

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
    """Build the 9x9 Liouvillian superoperator."""
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
    """
    Simulate the pulse for 6 input states simultaneously.
    Returns mean fidelity.
    """
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    # Initial density matrices vectorized
    initial_states = []
    for psi in [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    # Stack into a 9x6 matrix
    states = np.column_stack(initial_states)
    
    # Precompute Liouvillian for each step? No, I,Q change.
    # But we can compute the propagator for each step.
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        
        states = U_super @ states
        
    # Compute fidelities
    fidelities = []
    
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in
        
        # Extract rho_out
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')
        
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
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

def _generate_pulse(params):
    """Generate pulse from parameters [A, sigma, beta, alpha]."""
    A, sigma, beta, alpha = params
    
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    # Gaussian
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to 0 at edges
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    # Normalize area to pi (approx)
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = gauss
    
    # Derivative
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss_norm[1:] - gauss_norm[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    # Second derivative for alpha term
    d2gauss = np.zeros(N_STEPS)
    d2gauss[1:-1] = (dgauss[2:] - dgauss[:-2]) / (2 * DT)
    d2gauss[0] = d2gauss[1]
    d2gauss[-1] = d2gauss[-2]
    
    I = A * gauss_norm
    Q = beta * A * dgauss + alpha * A * d2gauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
        
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 290.0
    
    a, a_dag, n = _build_matrices()
    
    # Initial guess: DRAG pulse
    x0 = np.array([0.3, 5.0, 0.2, 0.0])
    
    best_pulse = _generate_pulse(x0)
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Optimize
    def objective(params):
        p = _generate_pulse(params)
        if not _check_validity(p):
            return 1.0
        return _compute_score(p, a, a_dag, n)
        
    # Try a few different starting points
    starts = [
        np.array([0.3, 5.0, 0.2, 0.0]),
        np.array([0.35, 4.0, 0.1, 0.0]),
        np.array([0.25, 6.0, 0.3, 0.0]),
        np.array([0.4, 5.0, 0.0, 0.0]),
        np.array([0.3, 5.0, -0.1, 0.0]),
        np.array([0.3, 5.0, 0.2, 0.05]),
        np.array([0.3, 5.0, 0.2, -0.05]),
    ]
    
    for x_start in starts:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = _generate_pulse(res.x)
        except Exception:
            pass
            
    return best_pulse
