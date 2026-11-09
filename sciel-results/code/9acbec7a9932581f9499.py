import numpy as np
from scipy.linalg import expm
import time
import sys

_PRECOMPUTED = None

def _get_precomputed():
    global _PRECOMPUTED
    if _PRECOMPUTED is None:
        a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
        a_dag = a.conj().T
        n = a_dag @ a
        n_nm1 = n @ (n - np.eye(3))
        a_plus_adag = a + a_dag
        a_minus_adag = a - a_dag
        _PRECOMPUTED = {
            'n': n,
            'n_nm1': n_nm1,
            'a_plus_adag': a_plus_adag,
            'a_minus_adag': a_minus_adag
        }
    return _PRECOMPUTED

def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps):
    """Simulate the quantum dynamics and return the final unitary matrix."""
    ops = _get_precomputed()
    n = ops['n']
    n_nm1 = ops['n_nm1']
    a_plus_adag = ops['a_plus_adag']
    a_minus_adag = ops['a_minus_adag']
    
    I3 = np.eye(3, dtype=complex)
    U_total = I3.copy()
    
    H_static_base = detuning * n + 0.5 * anharmonicity * n_nm1
    
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        
        # H = H_static + 0.5 * scale * (I * (a+a†) + iQ * (a-a†))
        H = H_static_base + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        
        # Evolve
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
        
    return U_total

def _compute_score_from_unitary(U):
    """Compute the score (mean + max infidelity) from the final unitary."""
    # X gate
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    # Initial states
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
        # Final state vector
        psi_final = U @ psi
        # Fidelity
        F = np.abs(np.vdot(phi, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
        
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    dt = 0.5
    n_steps = 80
    T_total = 40.0
    
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0
    
    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]
    
    def check_validity(pulse_IQ):
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amp_bound + 1e-10):
            return False
        
        p = pulse_IQ
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(p[i] - prev)
            if diff / dt > slew_bound + 1e-10:
                return False
            prev = p[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        if diff / dt > slew_bound + 1e-10:
            return False
            
        return True

    def evaluate_pulse(pulse_IQ):
        if not check_validity(pulse_IQ):
            return float('inf')
            
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps)
                score = _compute_score_from_unitary(U)
                scores.append(score)
                
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        return mean_score + max_score

    def make_pulse(sigma, beta, amplitude_scale=1.0):
        """Create a DRAG-style Gaussian pulse."""
        A = np.sqrt(np.pi / 2) / sigma * amplitude_scale
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        pulse_IQ = np.column_stack([I_base, Q_drag])
        return pulse_IQ

    # Seed pulse (matches the task description)
    sigma_seed = T_total / 6.0
    pulse_seed = make_pulse(sigma_seed, 0.0)
    score_seed = evaluate_pulse(pulse_seed)
    best_pulse = pulse_seed.copy()
    best_score = score_seed
    best_sigma = sigma_seed
    best_beta = 0.0
    
    # Coarse grid search
    sigma_range = np.linspace(T_total / 7.0, T_total / 5.0, 12)
    beta_range = np.linspace(-0.05, 0.15, 15)
    
    for sigma in sigma_range:
        if time.time() > deadline - 90.0:
            break
        for beta in beta_range:
            if time.time() > deadline - 90.0:
                break
            
            pulse_IQ = make_pulse(sigma, beta)
            score = evaluate_pulse(pulse_IQ)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ.copy()
                best_sigma = sigma
                best_beta = beta
    
    # Fine grid search around best found
    if time.time() < deadline - 30.0:
        sigma_fine_range = np.linspace(max(T_total/8.0, best_sigma - 0.5), min(T_total/4.0, best_sigma + 0.5), 12)
        beta_fine_range = np.linspace(max(-0.1, best_beta - 0.03), min(0.2, best_beta + 0.03), 12)
        
        for sigma in sigma_fine_range:
            if time.time() > deadline - 30.0:
                break
            for beta in beta_fine_range:
                if time.time() > deadline - 30.0:
                    break
                
                pulse_IQ = make_pulse(sigma, beta)
                score = evaluate_pulse(pulse_IQ)
                if score < best_score:
                    best_score = score
                    best_pulse = pulse_IQ.copy()
                    best_sigma = sigma
                    best_beta = beta
    
    # Try amplitude scaling to fine-tune the area
    if time.time() < deadline - 15.0:
        amp_scales = np.linspace(0.98, 1.02, 9)
        for amp in amp_scales:
            if time.time() > deadline - 15.0:
                break
            pulse_IQ = make_pulse(best_sigma, best_beta, amp)
            score = evaluate_pulse(pulse_IQ)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ.copy()
    
    # Try adding a small constant Q offset to compensate for asymmetric errors
    if time.time() < deadline - 5.0:
        q_offsets = np.linspace(-0.02, 0.02, 5)
        for q_off in q_offsets:
            if time.time() > deadline - 5.0:
                break
            pulse_IQ = best_pulse.copy()
            pulse_IQ[:, 1] += q_off
            if check_validity(pulse_IQ):
                score = evaluate_pulse(pulse_IQ)
                if score < best_score:
                    best_score = score
                    best_pulse = pulse_IQ.copy()
    
    return best_pulse
