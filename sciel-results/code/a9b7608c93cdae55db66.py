import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
import time

def _precompute_operators():
    """Precompute all constant operators."""
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    return a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag

def _simulate_unitary_fast(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate the quantum dynamics and return the final unitary matrix (optimized)."""
    U_total = I3.copy()
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    half_scale = 0.5 * scale
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + half_scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
    return U_total

def _compute_score_from_unitary(U, X):
    """Compute the score (mean + max infidelity) from the final unitary."""
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
    """Check if pulse satisfies amplitude and slew rate constraints."""
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

def _project_to_valid(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Project pulse to satisfy constraints by scaling."""
    p = pulse_IQ.copy()
    # Amplitude constraint
    amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        p *= amp_bound / max_amp
    # Slew constraint
    prev = np.array([0.0, 0.0])
    max_slew = 0
    for i in range(n_steps):
        diff = np.linalg.norm(p[i] - prev)
        slew = diff / dt
        max_slew = max(max_slew, slew)
        prev = p[i]
    diff = np.linalg.norm(-prev)
    max_slew = max(max_slew, diff / dt)
    if max_slew > slew_bound:
        p *= slew_bound / max_slew
    return p

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin

    dt = 0.5
    n_steps = 80
    T_total = 40.0
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002

    a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag = _precompute_operators()
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    def evaluate_pulse(pulse_IQ):
        """Evaluate the full 9-case score for a pulse."""
        if not _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
            return float('inf')
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary_fast(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    def make_parametric_pulse(params):
        """
        params: [sigma, beta, t_center, amp_scale, q_offset]
        """
        sigma = params[0]
        beta = params[1]
        t_c = params[2]
        amp_scale = params[3]
        q_offset = params[4]
        
        # Base Gaussian
        A_base = np.sqrt(np.pi / 2) / sigma
        I_base = A_base * np.exp(-(t_samples - t_c)**2 / (2 * sigma**2))
        
        # DRAG term
        dI_dt = -A_base * (t_samples - t_c) / sigma**2 * np.exp(-(t_samples - t_c)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        
        # Apply scaling and offset
        I_pulse = amp_scale * I_base
        Q_pulse = amp_scale * Q_drag + q_offset
        
        p = np.column_stack([I_pulse, Q_pulse])
        return _project_to_valid(p, amp_bound, slew_bound, dt, n_steps)

    # 1. Differential Evolution on 5 parameters
    # Bounds:
    # sigma: [4, 10] (T_total/10 to T_total/4)
    # beta: [-0.5, 0.5]
    # t_center: [10, 30]
    # amp_scale: [0.8, 1.2]
    # q_offset: [-0.05, 0.05]
    bounds = [(4.0, 10.0), (-0.5, 0.5), (10.0, 30.0), (0.8, 1.2), (-0.05, 0.05)]
    
    def de_objective(params):
        if time.time() > deadline - 100:
            return float('inf')
        p = make_parametric_pulse(params)
        return evaluate_pulse(p)

    try:
        de_result = differential_evolution(de_objective, bounds, maxiter=50, popsize=10, tol=1e-4, atol=1e-8, seed=42, polish=False)
        if de_result.fun < float('inf'):
            best_params = de_result.x
            best_pulse_de = make_parametric_pulse(best_params)
            best_score = de_result.fun
        else:
            # Fallback to seed
            sigma_seed = T_total / 6.0
            best_params = [sigma_seed, 0.15, t_center, 1.0, 0.0]
            best_pulse_de = make_parametric_pulse(best_params)
            best_score = evaluate_pulse(best_pulse_de)
    except Exception:
        sigma_seed = T_total / 6.0
        best_params = [sigma_seed, 0.15, t_center, 1.0, 0.0]
        best_pulse_de = make_parametric_pulse(best_params)
        best_score = evaluate_pulse(best_pulse_de)

    best_pulse = best_pulse_de.copy()

    # 2. Local Refinement with L-BFGS-B on full 160-dim pulse
    if time.time() < deadline - 50:
        def objective_flat(flat_params):
            if time.time() > deadline - 10:
                return float('inf')
            p = flat_params.reshape(n_steps, 2)
            p = _project_to_valid(p, amp_bound, slew_bound, dt, n_steps)
            return evaluate_pulse(p)
        
        init_flat = best_pulse.flatten()
        try:
            result = minimize(objective_flat, init_flat, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8,
                                      'maxls': 50})
            if result.fun < best_score:
                p_opt = _project_to_valid(result.x.reshape(n_steps, 2), 
                                          amp_bound, slew_bound, dt, n_steps)
                s_opt = evaluate_pulse(p_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_pulse = p_opt.copy()
        except Exception:
            pass

    return best_pulse
