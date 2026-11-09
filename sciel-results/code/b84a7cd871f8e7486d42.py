"""
Robust DRAG pulse for 3-level transmon X-pi gate.
Uses a time-domain feedback control approach with direct pulse optimization
via L-BFGS-B on the full 160-dimensional pulse vector.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants from task contract
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2.0 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2.0 * np.pi * 0.002
DT = 0.5  # ns
N_STEPS = 80
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Precompute ladder operator
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_a_dag = _a.conj().T
_n = _a_dag @ _a
_n2 = _n @ _n
_nn1 = _n2 - _n

# Precompute Hamiltonian components
_I3 = np.eye(3, dtype=complex)
_a_plus_adag = _a + _a_dag
_a_minus_adag = _a - _a_dag

# Detuning and scale values for the 9 cases
detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
scales = [0.97, 1.0, 1.03]

# Initial states for fidelity calculation
_states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [1/np.sqrt(2), 1j/np.sqrt(2), 0],
    [1/np.sqrt(2), -1j/np.sqrt(2), 0]
], dtype=complex)

# Target X gate matrix (in 3-level space, acts on |0>,|1> subspace)
_X_gate = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]], dtype=complex)


def _simulate_unitary(pulse_IQ, detuning, scale):
    """Simulate the quantum dynamics and return the final unitary matrix."""
    U_total = _I3.copy()
    H_static = detuning * _n + 0.5 * ANHARMONICITY * _nn1
    half_scale = 0.5 * scale
    for i in range(N_STEPS):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + half_scale * (I_val * _a_plus_adag + 1j * Q_val * _a_minus_adag)
        U_step = expm(-1j * H * DT)
        U_total = U_step @ U_total
    return U_total


def _compute_score_from_unitary(U):
    """Compute the score (mean + max infidelity) from the final unitary."""
    infidelities = []
    for state in _states:
        target = _X_gate @ state
        psi_final = U @ state
        F = np.abs(np.vdot(target, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid


def _check_validity(pulse_IQ):
    """Check if pulse satisfies amplitude and slew rate constraints."""
    amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amplitudes > AMPLITUDE_BOUND + 1e-10):
        return False
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        diff = np.linalg.norm(pulse_IQ[i] - prev)
        if diff / DT > SLEW_BOUND + 1e-10:
            return False
        prev = pulse_IQ[i]
    diff = np.linalg.norm(-prev)
    if diff / DT > SLEW_BOUND + 1e-10:
        return False
    return True


def _project_to_valid(pulse_IQ):
    """Project pulse to satisfy constraints by scaling."""
    p = pulse_IQ.copy()
    # Amplitude constraint
    amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        p *= AMPLITUDE_BOUND / max_amp
    # Slew constraint
    prev = np.array([0.0, 0.0])
    max_slew = 0
    for i in range(N_STEPS):
        diff = np.linalg.norm(p[i] - prev)
        slew = diff / DT
        max_slew = max(max_slew, slew)
        prev = p[i]
    diff = np.linalg.norm(-prev)
    max_slew = max(max_slew, diff / DT)
    if max_slew > SLEW_BOUND:
        p *= SLEW_BOUND / max_slew
    return p


def _make_drag_pulse(sigma, beta, amp_scale=1.0):
    """Create a DRAG-style Gaussian pulse."""
    t_samples = (np.arange(N_STEPS) + 0.5) * DT
    t_center = TOTAL_TIME / 2.0
    
    # Gaussian envelope with area = pi * amp_scale
    A = np.pi * amp_scale / (sigma * np.sqrt(2.0 * np.pi))
    I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
    
    # DRAG correction proportional to derivative
    dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
    Q_drag = -beta * dI_dt
    
    p = np.column_stack([I_base, Q_drag])
    return _project_to_valid(p)


def _evaluate_pulse(pulse_IQ):
    """Evaluate the full 9-case score for a pulse."""
    if not _check_validity(pulse_IQ):
        return float('inf')
    scores = []
    for det in detunings:
        for sc in scales:
            U = _simulate_unitary(pulse_IQ, det, sc)
            score = _compute_score_from_unitary(U)
            scores.append(score)
    return np.mean(scores) + np.max(scores)


def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I,Q amplitudes."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # Generate seed pulse
    sigma_seed = TOTAL_TIME / 6.0
    beta_seed = 0.15
    best_pulse = _make_drag_pulse(sigma_seed, beta_seed)
    best_score = _evaluate_pulse(best_pulse)
    
    # Coarse grid search over sigma and beta
    sigma_vals = np.linspace(TOTAL_TIME / 10, TOTAL_TIME / 4.5, 8)
    beta_vals = np.linspace(-0.25, 0.35, 12)
    
    for sigma in sigma_vals:
        if time.time() > deadline - 150:
            break
        for beta in beta_vals:
            if time.time() > deadline - 150:
                break
            p = _make_drag_pulse(sigma, beta)
            s = _evaluate_pulse(p)
            if s < best_score:
                best_score = s
                best_pulse = p.copy()
    
    # Direct time-domain optimization using L-BFGS-B
    if time.time() < deadline - 80:
        def objective_flat(flat_params):
            if time.time() > deadline - 20:
                return float('inf')
            p = flat_params.reshape(N_STEPS, 2)
            p = _project_to_valid(p)
            return _evaluate_pulse(p)
        
        init_flat = best_pulse.flatten()
        try:
            result = minimize(objective_flat, init_flat, method='L-BFGS-B',
                             options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8,
                                      'maxls': 40})
            if result.fun < best_score:
                p_opt = _project_to_valid(result.x.reshape(N_STEPS, 2))
                s_opt = _evaluate_pulse(p_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_pulse = p_opt.copy()
        except Exception:
            pass
    
    # Second round of refinement
    if time.time() < deadline - 40:
        def objective_flat2(flat_params):
            if time.time() > deadline - 15:
                return float('inf')
            p = flat_params.reshape(N_STEPS, 2)
            p = _project_to_valid(p)
            return _evaluate_pulse(p)
        
        init_flat2 = best_pulse.flatten()
        try:
            result2 = minimize(objective_flat2, init_flat2, method='L-BFGS-B',
                              options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10,
                                       'maxls': 30})
            if result2.fun < best_score:
                p_opt2 = _project_to_valid(result2.x.reshape(N_STEPS, 2))
                s_opt2 = _evaluate_pulse(p_opt2)
                if s_opt2 < best_score:
                    best_score = s_opt2
                    best_pulse = p_opt2.copy()
        except Exception:
            pass
    
    # Final validity check and projection
    best_pulse = _project_to_valid(best_pulse)
    
    return best_pulse
# EVOLVE-BLOCK-END
