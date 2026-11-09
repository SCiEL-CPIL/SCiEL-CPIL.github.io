import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution

def _precompute_operators():
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

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 315.0  # 15s safety margin

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
    t = t_samples - t_center

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    # Generate initial DRAG pulse
    sigma_seed = T_total / 6.0
    A_seed = 0.35
    beta_seed = 0.08
    I_base = A_seed * np.exp(-t**2 / (2 * sigma_seed**2))
    dI_dt = -A_seed * t / sigma_seed**2 * np.exp(-t**2 / (2 * sigma_seed**2))
    Q_drag = -beta_seed * dI_dt
    pulse_seed = np.column_stack([I_base, Q_drag])
    
    # Enforce constraints on seed
    amps = np.sqrt(pulse_seed[:, 0]**2 + pulse_seed[:, 1]**2)
    if np.max(amps) > amp_bound:
        pulse_seed *= (amp_bound / np.max(amps))
    
    max_slew = 0
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse_seed[i] - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        prev = pulse_seed[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    slew = diff / dt
    if slew > max_slew:
        max_slew = slew
    if max_slew > slew_bound:
        pulse_seed *= (slew_bound / max_slew)

    # Low-dimensional parameterization using Hermite-Gaussian basis
    # Parameters: [A0, sigma, center, beta_drag, A1, phase1, A2, phase2]
    # I(t) = A0 * H0(t) * exp(-(t-center)^2/(2*sigma^2)) + A1 * H1(t) * exp(-(t-center)^2/(2*sigma^2)) * cos(phase1)
    # Q(t) = beta_drag * dI/dt + A2 * H2(t) * exp(-(t-center)^2/(2*sigma^2)) * cos(phase2)
    
    def _hermite_h0(t, center, sigma):
        return np.exp(-(t - center)**2 / (2 * sigma**2))
    
    def _hermite_h1(t, center, sigma):
        x = (t - center) / sigma
        return x * np.exp(-x**2 / 2)
    
    def _hermite_h2(t, center, sigma):
        x = (t - center) / sigma
        return (x**2 - 1) * np.exp(-x**2 / 2)
    
    def _params_to_pulse(params):
        A0, sigma, center, beta_drag, A1, phase1, A2, phase2 = params
        sigma = np.clip(sigma, 2.0, 10.0)
        center = np.clip(center, -10.0, 10.0)
        
        h0 = _hermite_h0(t, center, sigma)
        h1 = _hermite_h1(t, center, sigma)
        h2 = _hermite_h2(t, center, sigma)
        
        I_comp = A0 * h0 + A1 * h1 * np.cos(phase1)
        Q_comp = A2 * h2 * np.cos(phase2)
        
        # Add DRAG term from I derivative
        dI_dt = A0 * (- (t - center) / sigma**2) * h0 + A1 * np.cos(phase1) * ((1 - (t - center)**2 / sigma**2) / sigma) * h0
        
        pulse = np.column_stack([I_comp, beta_drag * dI_dt + Q_comp])
        
        # Enforce constraints
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        if np.max(amps) > amp_bound:
            pulse *= (amp_bound / np.max(amps))
        
        max_slew = 0
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(pulse[i] - prev)
            slew = diff / dt
            if slew > max_slew:
                max_slew = slew
            prev = pulse[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        if max_slew > slew_bound:
            pulse *= (slew_bound / max_slew)
        
        return pulse

    def _objective_direct(x_flat):
        p = x_flat.reshape(n_steps, 2)
        if not _check_validity(p, amp_bound, slew_bound, dt, n_steps):
            return 1e6
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary_fast(p, det, sc, anharmonicity, dt, n_steps,
                                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    def _objective_params(params):
        p = _params_to_pulse(params)
        if not _check_validity(p, amp_bound, slew_bound, dt, n_steps):
            return 1e6
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary_fast(p, det, sc, anharmonicity, dt, n_steps,
                                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    # Differential Evolution for global optimization in low-dimensional space
    # Bounds for 8 parameters: [A0, sigma, center, beta_drag, A1, phase1, A2, phase2]
    bounds_de = [(-0.4, 0.4), (2.0, 10.0), (-10.0, 10.0), (-0.5, 0.5), 
                 (-0.3, 0.3), (0, 2*np.pi), (-0.3, 0.3), (0, 2*np.pi)]
    
    best_pulse = pulse_seed.copy()
    best_score = _objective_direct(pulse_seed.flatten())
    
    # Stage 1: Differential Evolution (global search)
    if time.time() < deadline - 60.0:
        try:
            res_de = differential_evolution(_objective_params, bounds_de, 
                                            maxiter=100, popsize=15, tol=1e-6,
                                            seed=42, polish=False)
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_pulse = _params_to_pulse(res_de.x)
        except Exception as e:
            pass
    
    # Stage 2: Local refinement with L-BFGS-B on full 160-dim space
    if time.time() < deadline - 30.0:
        try:
            x0 = best_pulse.flatten()
            # Use tighter bounds to help convergence, but keep them wide enough
            bounds_full = [(-0.39, 0.39)] * 160
            res_local = minimize(_objective_direct, x0, method='L-BFGS-B', bounds=bounds_full,
                                 options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6,
                                          'maxfun': 2000})
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_pulse = res_local.x.reshape(80, 2)
        except Exception as e:
            pass
    
    # Stage 3: Fine-tune with Nelder-Mead if time remains
    if time.time() < deadline - 10.0:
        try:
            x0 = best_pulse.flatten()
            bounds_full = [(-0.39, 0.39)] * 160
            res_nm = minimize(_objective_direct, x0, method='Nelder-Mead', bounds=bounds_full,
                              options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_pulse = res_nm.x.reshape(80, 2)
        except Exception as e:
            pass

    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Try to project by scaling
        amps = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
        if np.max(amps) > amp_bound:
            best_pulse *= (amp_bound / np.max(amps))
        
        # Check slew again
        max_slew = 0
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(best_pulse[i] - prev)
            slew = diff / dt
            if slew > max_slew:
                max_slew = slew
            prev = best_pulse[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        if max_slew > slew_bound:
            best_pulse *= (slew_bound / max_slew)
        
        # If still invalid, use seed
        if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
            best_pulse = pulse_seed

    return best_pulse
