"""
Robust X-pi gate control pulse for a three-level transmon.

Strategy:
1. Fast unitary simulation (no density matrix) for the 9 detuning/scale cases.
2. Start from a DRAG-style Gaussian seed with area pi.
3. Refine with:
   - Nelder-Mead on a 5-parameter DRAG family (fast, global-ish).
   - Direct L-BFGS-B on the 160 pulse values with soft constraint penalties
     (local, fine-grained).
4. Enforce amplitude and slew-rate constraints by projection/scaling.
5. Return the best valid pulse found within the time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _precompute_operators():
    """Precompute the 3x3 ladder and number operators."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    return a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag


def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate the unitary evolution for a given pulse and hardware params."""
    U_total = I3.copy()
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
    return U_total


def _compute_score_from_unitary(U, X):
    """Compute mean + max infidelity over the 6 test states."""
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    infidelities = []
    for psi in initial_states:
        phi = X @ psi
        psi_final = U @ psi
        F = np.abs(np.vdot(phi, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid


def _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Check amplitude and slew-rate constraints including boundaries."""
    amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amplitudes > amp_bound + 1e-10):
        return False
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse_IQ[i] - prev)
        if diff / dt > slew_bound + 1e-10:
            return False
        prev = pulse_IQ[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    if diff / dt > slew_bound + 1e-10:
        return False
    return True


def _enforce_constraints(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Project pulse onto feasible set by scaling if needed."""
    # Amplitude constraint
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        pulse_IQ = pulse_IQ * (amp_bound / max_amp)
    
    # Slew rate constraint
    max_slew = 0.0
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse_IQ[i] - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        prev = pulse_IQ[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    slew = diff / dt
    if slew > max_slew:
        max_slew = slew
    
    if max_slew > slew_bound:
        pulse_IQ = pulse_IQ * (slew_bound / max_slew)
    
    return pulse_IQ


def _generate_drag_pulse(params, n_steps=80, dt=0.5):
    """Generate a DRAG pulse from parameters [amp, sigma, drag_beta, phase, offset]."""
    amp, sigma, drag_beta, phase, offset = params
    t = np.arange(n_steps) * dt + dt/2
    t_center = (n_steps - 1) * dt / 2 + dt/2
    t_rel = t - t_center + offset
    
    gauss = np.exp(-0.5 * (t_rel / sigma)**2)
    # Offset to make edges approximately zero
    gauss = gauss - gauss[0]
    gauss = gauss - gauss[-1]
    if np.max(gauss) > 0:
        gauss = gauss / np.max(gauss)
    
    # Derivative of Gaussian
    dgauss = -(t_rel / sigma**2) * gauss
    
    I = amp * gauss * np.cos(phase) - amp * drag_beta * dgauss * np.sin(phase)
    Q = amp * gauss * np.sin(phase) + amp * drag_beta * dgauss * np.cos(phase)
    
    pulse = np.column_stack([I, Q])
    
    # Normalize area to pi
    area = np.sum(np.sqrt(I**2 + Q**2)) * dt
    if area > 1e-10:
        pulse *= np.pi / area
    
    return pulse


def _score_all_cases(pulse_IQ, T1, Tphi, anharmonicity, detunings, scales,
                     a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Compute score over all 9 detuning/scale cases using unitary simulation."""
    dt = 0.5
    n_steps = pulse_IQ.shape[0]
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    scores = []
    for det in detunings:
        for sc in scales:
            U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                  a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
            score = _compute_score_from_unitary(U, X)
            scores.append(score)
    
    mean_score = np.mean(scores)
    max_score = np.max(scores)
    return mean_score + max_score


def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    anharmonicity = -2 * np.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    n_steps = 80
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag = _precompute_operators()
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Initial DRAG pulse parameters
    initial_params = [0.3, 40.0/6.0, 0.08, 0.0, 0.0]
    
    # Generate initial pulse
    pulse_array = _generate_drag_pulse(initial_params)
    pulse_array = _enforce_constraints(pulse_array, amp_bound, slew_bound, dt, n_steps)
    
    best_score = _score_all_cases(pulse_array, T1, Tphi, anharmonicity,
                                   detunings, scales, a, a_dag, n, I3,
                                   n_nm1, a_plus_adag, a_minus_adag)
    best_pulse = pulse_array.copy()
    
    # --- Nelder-Mead on 5-parameter DRAG family ---
    def objective_drag(params):
        p = _generate_drag_pulse(params)
        p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
        if not _check_validity(p, amp_bound, slew_bound, dt, n_steps):
            return 1e10
        return _score_all_cases(p, T1, Tphi, anharmonicity,
                                detunings, scales, a, a_dag, n, I3,
                                n_nm1, a_plus_adag, a_minus_adag)
    
    if time.time() < deadline - 120:
        try:
            bounds = [(0.1, 0.35), (3.0, 8.0), (0.0, 0.3), (0.0, 2*np.pi), (-0.1, 0.1)]
            result = minimize(objective_drag, initial_params,
                             method='Nelder-Mead', bounds=bounds,
                             options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                p = _generate_drag_pulse(result.x)
                p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
                if _check_validity(p, amp_bound, slew_bound, dt, n_steps):
                    best_pulse = p.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    # --- L-BFGS-B on 160 pulse values ---
    def objective_direct(p_flat):
        p = p_flat.reshape(80, 2)
        
        # Amplitude penalty
        amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        amp_pen = np.sum(np.maximum(0, amps - amp_bound)**2)
        
        # Slew penalty
        slew_pen = 0.0
        prev = np.array([0.0, 0.0])
        for i in range(80):
            diff = np.linalg.norm(p[i] - prev)
            if diff > slew_bound * dt:
                slew_pen += (diff - slew_bound * dt)**2
            prev = p[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        if diff > slew_bound * dt:
            slew_pen += (diff - slew_bound * dt)**2
        
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(p, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex))
                scores.append(score)
        
        raw_score = np.mean(scores) + np.max(scores)
        return raw_score + 1000.0 * (amp_pen + slew_pen)
    
    if time.time() < deadline - 30:
        try:
            p0 = best_pulse.flatten()
            bounds_direct = [(-0.4, 0.4)] * 160
            result = minimize(objective_direct, p0, method='L-BFGS-B',
                             bounds=bounds_direct,
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.fun < best_score + 1e-6:
                p = result.x.reshape(80, 2)
                p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
                if _check_validity(p, amp_bound, slew_bound, dt, n_steps):
                    sc = _score_all_cases(p, T1, Tphi, anharmonicity,
                                          detunings, scales, a, a_dag, n, I3,
                                          n_nm1, a_plus_adag, a_minus_adag)
                    if sc < best_score:
                        best_pulse = p.copy()
                        best_score = sc
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        best_pulse = _enforce_constraints(best_pulse, amp_bound, slew_bound, dt, n_steps)
    
    return best_pulse
# EVOLVE-BLOCK-END
