import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
from numpy.polynomial.legendre import legvander

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
N_COEFFS = 8

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
    I3 = np.eye(3, dtype=complex)
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - I3)) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
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

def _legendre_basis():
    """Create Legendre polynomial evaluation matrix for N_STEPS points in [0,1]."""
    t = np.linspace(0, 1, N_STEPS)
    V = legvander(t, N_COEFFS - 1)
    return V

def _pulse_from_legendre(coeffs):
    """Convert 16 coefficients (8 for I, 8 for Q) to pulse array."""
    V = _legendre_basis()
    I_coeffs = coeffs[:N_COEFFS]
    Q_coeffs = coeffs[N_COEFFS:]
    I = V @ I_coeffs
    Q = V @ Q_coeffs
    pulse = np.column_stack([I, Q])
    return pulse

def _objective_legendre(coeffs, a, a_dag, n):
    """Objective function for optimization."""
    pulse = _pulse_from_legendre(coeffs)
    
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-6):
        return 1e6 + np.max(amps - AMPLITUDE_BOUND)
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-6:
        return 1e6 + (slew0 - SLEW_BOUND)
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-6:
            return 1e6 + (diff - SLEW_BOUND)
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-6:
        return 1e6 + (slew_last - SLEW_BOUND)
    
    score = _compute_score(pulse, a, a_dag, n)
    return score

def _generate_initial_guess_legendre(a, a_dag, n):
    """Generate an initial guess in Legendre coefficient space."""
    t = np.arange(N_STEPS) * DT
    sigma = 5.0
    t_center = 20.0
    A = 0.25
    beta = 0.15
    
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
    
    V = _legendre_basis()
    I_coeffs, _, _, _ = np.linalg.lstsq(V, I, rcond=None)
    Q_coeffs, _, _, _ = np.linalg.lstsq(V, Q, rcond=None)
    
    coeffs = np.concatenate([I_coeffs, Q_coeffs])
    
    pulse = _pulse_from_legendre(coeffs)
    if _check_validity(pulse):
        return coeffs
    
    coeffs = np.zeros(2 * N_COEFFS)
    coeffs[0] = 0.2
    coeffs[N_COEFFS] = 0.05
    return coeffs

def _enforce_constraints_legendre(coeffs):
    """Enforce constraints by scaling if necessary."""
    pulse = _pulse_from_legendre(coeffs)
    
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        coeffs *= scale_factor
        pulse = _pulse_from_legendre(coeffs)
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    max_slew = slew0
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        max_slew = max(max_slew, diff)
    slew_last = np.linalg.norm(pulse[-1]) / DT
    max_slew = max(max_slew, slew_last)
    
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        coeffs *= scale_factor
        pulse = _pulse_from_legendre(coeffs)
    
    return coeffs

def _direct_pulse_optimization(a, a_dag, n, start_time, deadline):
    """
    Direct optimization of the 80-step pulse with explicit constraint handling.
    Uses a smooth penalty for constraints and L-BFGS-B.
    """
    # Start from a good DRAG-like seed
    t = np.arange(N_STEPS) * DT
    sigma = 5.0
    t_center = 20.0
    A = 0.25
    beta = 0.15
    
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
    
    I_seed = A * gauss_norm
    Q_seed = beta * A * dgauss
    
    # Ensure seed is valid
    if not _check_validity(np.column_stack([I_seed, Q_seed])):
        # Scale down if needed
        amps = np.sqrt(I_seed**2 + Q_seed**2)
        if np.max(amps) > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / np.max(amps)
            I_seed *= scale
            Q_seed *= scale
    
    pulse0 = np.column_stack([I_seed, Q_seed])
    
    def objective(pulse_flat):
        pulse = pulse_flat.reshape(N_STEPS, 2)
        
        # Amplitude constraint
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        amp_violation = np.max(np.maximum(0, amps - AMPLITUDE_BOUND))
        
        # Slew constraints
        slew0 = np.linalg.norm(pulse[0]) / DT
        slew_violation = max(0, slew0 - SLEW_BOUND)
        
        for i in range(N_STEPS - 1):
            diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
            slew_violation = max(slew_violation, diff - SLEW_BOUND)
        
        slew_last = np.linalg.norm(pulse[-1]) / DT
        slew_violation = max(slew_violation, slew_last - SLEW_BOUND)
        
        if amp_violation > 0 or slew_violation > 0:
            return 1e6 + amp_violation + slew_violation
        
        score = _compute_score(pulse, a, a_dag, n)
        return score
    
    # Optimize
    result = minimize(
        objective,
        pulse0.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    if result.fun < 1e5:
        pulse_opt = result.x.reshape(N_STEPS, 2)
        if _check_validity(pulse_opt):
            return pulse_opt
    
    return pulse0

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 310.0
    
    a, a_dag, n = _build_matrices()
    rng = np.random.default_rng(42)
    
    # Strategy 1: Legendre basis optimization
    best_coeffs = _generate_initial_guess_legendre(a, a_dag, n)
    best_pulse = _pulse_from_legendre(best_coeffs)
    
    if not _check_validity(best_pulse):
        best_coeffs = _enforce_constraints_legendre(best_coeffs)
        best_pulse = _pulse_from_legendre(best_coeffs)
    
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Multiple restarts with Legendre
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 120.0:
            break
        
        if restart == 0:
            coeffs0 = best_coeffs.copy()
        else:
            coeffs0 = best_coeffs + rng.normal(0, 0.03, size=2 * N_COEFFS)
            coeffs0 = _enforce_constraints_legendre(coeffs0)
        
        try:
            result = minimize(
                _objective_legendre,
                coeffs0,
                args=(a, a_dag, n),
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-7, 'gtol': 1e-5}
            )
            
            if result.fun < best_score and result.fun < 1e5:
                new_coeffs = _enforce_constraints_legendre(result.x)
                new_pulse = _pulse_from_legendre(new_coeffs)
                
                if _check_validity(new_pulse):
                    new_score = _compute_score(new_pulse, a, a_dag, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_pulse = new_pulse
                        best_coeffs = new_coeffs
        except Exception:
            pass
    
    # Strategy 2: Direct pulse optimization (more flexible shape space)
    if time.time() < deadline - 60.0:
        try:
            pulse_direct = _direct_pulse_optimization(a, a_dag, n, start_time, deadline)
            if _check_validity(pulse_direct):
                score_direct = _compute_score(pulse_direct, a, a_dag, n)
                if score_direct < best_score:
                    best_score = score_direct
                    best_pulse = pulse_direct
        except Exception:
            pass
    
    # Final refinement if time permits
    if time.time() < deadline - 30.0:
        try:
            # Refine in Legendre space
            V = _legendre_basis()
            I_coeffs, _, _, _ = np.linalg.lstsq(V, best_pulse[:, 0], rcond=None)
            Q_coeffs, _, _, _ = np.linalg.lstsq(V, best_pulse[:, 1], rcond=None)
            coeffs = np.concatenate([I_coeffs, Q_coeffs])
            
            result = minimize(
                _objective_legendre,
                coeffs,
                args=(a, a_dag, n),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            
            if result.fun < best_score and result.fun < 1e5:
                new_coeffs = _enforce_constraints_legendre(result.x)
                new_pulse = _pulse_from_legendre(new_coeffs)
                
                if _check_validity(new_pulse):
                    new_score = _compute_score(new_pulse, a, a_dag, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_pulse = new_pulse
        except Exception:
            pass
    
    # Ensure final pulse is valid
    if not _check_validity(best_pulse):
        best_coeffs = _enforce_constraints_legendre(best_coeffs)
        best_pulse = _pulse_from_legendre(best_coeffs)
    
    return best_pulse
