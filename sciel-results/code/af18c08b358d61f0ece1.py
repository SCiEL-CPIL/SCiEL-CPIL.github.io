import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    initial_states = []
    for psi in [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    states = np.column_stack(initial_states)
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        states = U_super @ states
        
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in
        
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')
        
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
        
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-10:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-10:
            return False
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-10:
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

def _generate_pulse_legendre(coeffs):
    """
    Generate pulse from Legendre polynomial coefficients.
    coeffs: array of 16 values [c0_I, c1_I, ..., c7_I, c0_Q, c1_Q, ..., c7_Q]
    The pulse is constructed as:
      I(t) = sum_{k=0}^{7} c_k_I * P_k(t') * envelope(t)
      Q(t) = sum_{k=0}^{7} c_k_Q * P_k(t') * envelope(t)
    where t' = 2t/T - 1 maps [0, T] to [-1, 1], and envelope ensures zero at edges.
    """
    T = N_STEPS * DT
    t = np.arange(N_STEPS) * DT
    t_norm = 2 * t / T - 1  # [-1, 1]
    
    # Envelope: sin(pi * t / T) ensures zero at both edges
    envelope = np.sin(np.pi * t / T)
    
    # Compute Legendre polynomials P_0 through P_7
    P = np.zeros((8, N_STEPS))
    P[0] = 1.0
    P[1] = t_norm
    for k in range(2, 8):
        P[k] = ((2*k - 1) * t_norm * P[k-1] - (k - 1) * P[k-2]) / k
    
    # Build I and Q
    c_I = coeffs[:8]
    c_Q = coeffs[8:16]
    
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    for k in range(8):
        I += c_I[k] * P[k]
        Q += c_Q[k] * P[k]
    
    I *= envelope
    Q *= envelope
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    # Enforce slew bound by clipping differences
    for i in range(N_STEPS - 1):
        diff = pulse[i+1] - pulse[i]
        norm_diff = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if norm_diff > max_diff:
            pulse[i+1] = pulse[i] + diff * (max_diff / norm_diff)
    
    # Re-check amplitude after slew enforcement
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    # Final slew check
    for i in range(N_STEPS - 1):
        diff = pulse[i+1] - pulse[i]
        norm_diff = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if norm_diff > max_diff:
            pulse[i+1] = pulse[i] + diff * (max_diff / norm_diff)
    
    return pulse

def _generate_pulse_gaussian(params):
    """Generate pulse from parameters [A, sigma, beta]."""
    A, sigma, beta = params
    
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = gauss
    
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss_norm[1:] - gauss_norm[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I = A * gauss_norm
    Q = beta * A * dgauss
    
    pulse = np.column_stack([I, Q])
    
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    return pulse

def _seed_pulse():
    """Generate the seed DRAG-style Gaussian pulse."""
    A, sigma, beta = 0.3, 5.0, 0.2
    return _generate_pulse_gaussian([A, sigma, beta])

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    a, a_dag, n = _build_matrices()
    
    # Start with seed pulse
    best_pulse = _seed_pulse()
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Objective function for optimization
    def objective(params):
        p = _generate_pulse_legendre(params)
        if not _check_validity(p):
            return 10.0  # Penalty
        return _compute_score(p, a, a_dag, n)
    
    # Initial parameters: try to match the seed pulse approximately
    # The seed has area pi, so the DC component should be around pi/T * envelope_integral
    # For now, start with reasonable values
    x0 = np.zeros(16)
    x0[0] = 0.35  # I DC component
    x0[8] = 0.0   # Q DC component
    
    # Multi-phase optimization
    # Phase 1: L-BFGS-B with moderate iterations
    if time.time() < deadline - 60:
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_score and _check_validity(_generate_pulse_legendre(res.x)):
                best_score = res.fun
                best_pulse = _generate_pulse_legendre(res.x)
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 30:
        x_nm = np.array(_generate_pulse_legendre(best_pulse).flatten())
        # Actually, we need to work in parameter space
        # Let's just do Nelder-Mead on the parameters
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_score and _check_validity(_generate_pulse_legendre(res.x)):
                best_score = res.fun
                best_pulse = _generate_pulse_legendre(res.x)
        except Exception:
            pass
    
    # Phase 3: Final polish with L-BFGS-B from best point
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score and _check_validity(_generate_pulse_legendre(res.x)):
                best_score = res.fun
                best_pulse = _generate_pulse_legendre(res.x)
        except Exception:
            pass
    
    # Direct pulse optimization: optimize the 160-dimensional pulse directly with constraints
    if time.time() < deadline - 5:
        # Use the best pulse as starting point
        x_direct = best_pulse.flatten().copy()
        
        def objective_direct(params):
            p = params.reshape(N_STEPS, 2)
            if not _check_validity(p):
                return 10.0
            return _compute_score(p, a, a_dag, n)
        
        try:
            res = minimize(objective_direct, x_direct, method='L-BFGS-B',
                           options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score and _check_validity(res.x.reshape(N_STEPS, 2)):
                best_score = res.fun
                best_pulse = res.x.reshape(N_STEPS, 2)
        except Exception:
            pass
    
    return best_pulse
