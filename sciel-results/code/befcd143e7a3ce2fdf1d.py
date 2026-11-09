"""Improved program for robust X-pi gate control pulse with DRAG and optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

# Precompute constant operators
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_n_minus_1 = _n - np.eye(3)
_X_gate = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]], dtype=complex)

def _build_hamiltonian(detuning, scale, I, Q, anharmonicity):
    """Build the 3x3 Hamiltonian for the transmon system."""
    H = detuning * _n + 0.5 * anharmonicity * _n @ _n_minus_1
    H += 0.5 * scale * (I * (_a + _adag) + 1j * Q * (_a - _adag))
    return H

def _simulate_unitary(pulse_array, detuning, scale, anharmonicity, dt=0.5):
    """Simulate unitary evolution and return the full 3x3 unitary matrix."""
    n_steps = pulse_array.shape[0]
    U_total = np.eye(3, dtype=complex)
    
    for i in range(n_steps):
        I_val = pulse_array[i, 0]
        Q_val = pulse_array[i, 1]
        H = _build_hamiltonian(detuning, scale, I_val, Q_val, anharmonicity)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
    
    return U_total

def _compute_infidelity_from_unitary(U, target_U):
    """Compute infidelity from unitary vs target unitary."""
    # Fidelity = (1/d) * Tr(U_target^dag * U)
    # For pure states, we can use the overlap of each basis state
    # Better: use the process fidelity approach
    # Actually, let's compute mean fidelity over the 6 test states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    fidelities = []
    for state in states:
        evolved = U @ state
        target = target_U @ state
        fid = np.abs(np.vdot(target, evolved))**2
        fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    return 1.0 - mean_fid

def _score_all_cases(pulse_array, anharmonicity):
    """Compute score over all 9 detuning/scale cases using unitary simulation."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            U = _simulate_unitary(pulse_array, det, sc, anharmonicity)
            inf = _compute_infidelity_from_unitary(U, _X_gate)
            infidelities.append(inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf

def _check_constraints(pulse_array, dt=0.5, amp_bound=0.4, slew_bound=0.12):
    """Check if pulse satisfies amplitude and slew rate constraints."""
    amplitudes = np.sqrt(pulse_array[:, 0]**2 + pulse_array[:, 1]**2)
    if np.any(amplitudes > amp_bound + 1e-10):
        return False
    
    # Slew rate: from zero to first, between steps, last to zero
    p0 = np.array([0.0, 0.0])
    if np.linalg.norm(pulse_array[0] - p0) / dt > slew_bound + 1e-10:
        return False
    
    for i in range(len(pulse_array) - 1):
        diff = pulse_array[i+1] - pulse_array[i]
        if np.linalg.norm(diff) / dt > slew_bound + 1e-10:
            return False
    
    if np.linalg.norm(p0 - pulse_array[-1]) / dt > slew_bound + 1e-10:
        return False
    
    return True

def _enforce_constraints(pulse_array, dt=0.5, amp_bound=0.4, slew_bound=0.12):
    """Enforce constraints by iterative projection."""
    pulse = pulse_array.copy()
    
    for _ in range(50):
        # Enforce amplitude
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        if np.any(amps > amp_bound + 1e-12):
            scale = np.where(amps > amp_bound, amp_bound / (amps + 1e-15), 1.0)
            pulse[:, 0] *= scale
            pulse[:, 1] *= scale
        
        # Enforce slew rate
        # From zero to first
        p0 = np.array([0.0, 0.0])
        diff = pulse[0] - p0
        norm = np.linalg.norm(diff)
        if norm > slew_bound * dt:
            pulse[0] = p0 + diff * (slew_bound * dt / norm)
        
        # Between consecutive
        for i in range(len(pulse) - 1):
            diff = pulse[i+1] - pulse[i]
            norm = np.linalg.norm(diff)
            if norm > slew_bound * dt:
                pulse[i+1] = pulse[i] + diff * (slew_bound * dt / norm)
        
        # Last to zero
        plast = pulse[-1]
        diff = p0 - plast
        norm = np.linalg.norm(diff)
        if norm > slew_bound * dt:
            pulse[-1] = p0 - diff * (slew_bound * dt / norm)
    
    return pulse

def _generate_drag_pulse(params, n_steps=80, dt=0.5, anharmonicity=-2*np.pi*0.25):
    """Generate a DRAG pulse from parameters.
    params: [sigma_ratio, drag_beta, phase]
    sigma_ratio = sigma / (n_steps * dt / 2)  -- normalized width
    """
    sigma_ratio, drag_beta, phase = params
    t = np.arange(n_steps) * dt + dt/2
    t_center = (n_steps * dt) / 2.0
    
    # Gaussian with sigma = sigma_ratio * gate_duration
    gate_duration = n_steps * dt
    sigma = sigma_ratio * gate_duration
    
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Derivative for DRAG
    dgauss = -((t - t_center) / sigma**2) * gauss
    
    # DRAG formula: the Q component compensates for leakage
    # Standard DRAG: I = A * cos(phase) * gauss, Q = A * sin(phase) * gauss + beta * A * cos(phase) * dgauss
    # But we need to normalize area to pi
    A = 1.0  # Will be normalized
    
    I = A * gauss * np.cos(phase)
    Q = A * gauss * np.sin(phase) + A * drag_beta * dgauss * np.cos(phase)
    
    pulse = np.column_stack([I, Q])
    
    # Normalize area to pi
    area = np.sum(np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)) * dt
    if area > 1e-10:
        pulse *= np.pi / area
    
    # Enforce constraints
    pulse = _enforce_constraints(pulse, dt)
    
    return pulse

def _objective(params, anharmonicity):
    """Objective function for optimization."""
    pulse_array = _generate_drag_pulse(params, anharmonicity=anharmonicity)
    
    if not _check_constraints(pulse_array):
        return 1e10
    
    score = _score_all_cases(pulse_array, anharmonicity)
    return score

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    dt = 0.5
    n_steps = 80
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Initial DRAG pulse parameters: [sigma_ratio, drag_beta, phase]
    # sigma_ratio ~ 0.25 means sigma = 0.25 * 40ns = 10ns, close to 40/6 ≈ 6.67ns
    initial_params = [0.25, 0.0, 0.0]
    
    best_score = 1e10
    best_pulse = None
    
    # Generate initial pulse
    pulse_array = _generate_drag_pulse(initial_params, anharmonicity=anharmonicity)
    if _check_constraints(pulse_array):
        best_pulse = pulse_array.copy()
        best_score = _score_all_cases(pulse_array, anharmonicity)
    
    # Nelder-Mead optimization on 3 parameters
    bounds = [(0.1, 0.5), (-0.5, 0.5), (-np.pi, np.pi)]
    
    if time.time() < deadline - 30:
        try:
            result = minimize(_objective, initial_params, args=(anharmonicity,),
                             method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            
            if result.fun < best_score:
                pulse_array = _generate_drag_pulse(result.x, anharmonicity=anharmonicity)
                if _check_constraints(pulse_array):
                    best_pulse = pulse_array.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    # Try multiple random initializations
    if time.time() < deadline - 20:
        np.random.seed(42)
        for _ in range(20):
            if time.time() >= deadline - 15:
                break
            try:
                rand_params = [
                    np.random.uniform(0.15, 0.4),
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.5, 0.5)
                ]
                pulse_array = _generate_drag_pulse(rand_params, anharmonicity=anharmonicity)
                if not _check_constraints(pulse_array):
                    continue
                score = _score_all_cases(pulse_array, anharmonicity)
                if score < best_score:
                    best_pulse = pulse_array.copy()
                    best_score = score
                    # Polish from this point
                    if time.time() < deadline - 15:
                        try:
                            result = minimize(_objective, rand_params, args=(anharmonicity,),
                                             method='Nelder-Mead',
                                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                                      'adaptive': True})
                            if result.fun < best_score:
                                pulse_array = _generate_drag_pulse(result.x, anharmonicity=anharmonicity)
                                if _check_constraints(pulse_array):
                                    best_pulse = pulse_array.copy()
                                    best_score = result.fun
                        except Exception:
                            pass
            except Exception:
                continue
    
    # Direct pulse optimization as final polish
    if best_pulse is not None and time.time() < deadline - 10:
        try:
            def _direct_objective(pulse_flat):
                p = pulse_flat.reshape(n_steps, 2)
                if not _check_constraints(p):
                    return 1e10
                return _score_all_cases(p, anharmonicity)
            
            result = minimize(_direct_objective, best_pulse.flatten(),
                             method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < best_score:
                p = result.x.reshape(n_steps, 2)
                p = _enforce_constraints(p, dt)
                if _check_constraints(p):
                    best_pulse = p.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    # Final fallback
    if best_pulse is None:
        best_pulse = _generate_drag_pulse(initial_params, anharmonicity=anharmonicity)
    
    # Final constraint enforcement
    if not _check_constraints(best_pulse):
        best_pulse = _enforce_constraints(best_pulse, dt)
    
    return best_pulse
# EVOLVE-BLOCK-END
