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

def _parametric_pulse(params, dt, n_steps, T_total):
    """Generate a DRAG-style pulse from low-dimensional parameters.
    params: [amp, sigma, beta, c1, c2, c3]
    c1-c3 are Hermite-Gaussian correction coefficients
    """
    amp, sigma, beta, c1, c2, c3 = params
    t = (np.arange(n_steps) + 0.5) * dt - T_total / 2.0
    
    # Base Gaussian
    g = np.exp(-t**2 / (2.0 * sigma**2))
    
    # Normalize area to pi
    area = np.sum(g) * dt
    g = g * (np.pi / area) if area > 0 else g
    
    I_base = amp * g
    
    # DRAG term: proportional to dI/dt
    dI_dt = -amp * t / sigma**2 * g
    Q_drag = -beta * dI_dt
    
    # Correction terms: higher-order Hermite-Gaussians
    H0 = g
    H1 = t / sigma * g
    H2 = (t**2 / sigma**2 - 1.0) * g
    
    I_corr = c1 * H0 + c2 * H1 + c3 * H2
    Q_corr = c1 * H1 + c2 * H2  # orthogonal corrections
    
    pulse = np.column_stack([I_base + I_corr, Q_drag + Q_corr])
    return pulse

def _compute_score_from_unitary(U, X):
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = np.array([psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]).T  # 3x6
    target_states = X @ initial_states  # 3x6
    final_states = U @ initial_states  # 3x6
    # Fidelity for each state: |vdot(target_i, final_i)|^2
    overlaps = np.abs(np.sum(np.conj(target_states) * final_states, axis=0))**2
    infidelities = 1.0 - overlaps
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

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

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

    def _evaluate_pulse(pulse_IQ):
        """Compute the score for a given pulse."""
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary_fast(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    def _parametric_objective(params):
        """Objective for parametric optimization."""
        pulse_IQ = _parametric_pulse(params, dt, n_steps, T_total)
        
        # Check constraints and add penalties
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        amp_pen = np.sum(np.maximum(0, amps - amp_bound)**2)
        
        slew_pen = 0.0
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(pulse_IQ[i] - prev)
            if diff > slew_bound * dt:
                slew_pen += (diff - slew_bound * dt)**2
            prev = pulse_IQ[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        if diff > slew_bound * dt:
            slew_pen += (diff - slew_bound * dt)**2
        
        if amp_pen > 1e-12 or slew_pen > 1e-12:
            return 1000.0 + 100.0 * (amp_pen + slew_pen)
        
        return _evaluate_pulse(pulse_IQ)

    # Phase 1: Global optimization in parametric space
    # params: [amp, sigma, beta, c1, c2, c3]
    bounds_param = [
        (0.1, 0.39),      # amp
        (3.0, 10.0),      # sigma (ns)
        (-0.5, 0.5),      # beta
        (-0.1, 0.1),      # c1
        (-0.1, 0.1),      # c2
        (-0.1, 0.1),      # c3
    ]
    
    best_params = None
    best_score = float('inf')
    
    # Use differential evolution for global search
    try:
        de_result = differential_evolution(
            _parametric_objective,
            bounds=bounds_param,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True,
            callback=None
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_params = de_result.x
    except Exception:
        pass
    
    # If DE didn't find anything good, try a few manual starting points
    if best_params is None or best_score > 10:
        test_params_list = [
            [0.35, 6.67, 0.08, 0, 0, 0],
            [0.35, 6.67, -0.08, 0, 0, 0],
            [0.30, 6.0, 0.10, 0, 0, 0],
            [0.35, 7.0, 0.05, 0, 0, 0],
            [0.35, 6.67, 0.08, 0.02, 0, 0],
            [0.35, 6.67, 0.08, 0, 0.02, 0],
        ]
        for tp in test_params_list:
            if time.time() > deadline - 60:
                break
            sc = _parametric_objective(tp)
            if sc < best_score:
                best_score = sc
                best_params = tp
    
    # Phase 2: Local refinement in parametric space
    if best_params is not None:
        try:
            res = minimize(_parametric_objective, best_params, method='L-BFGS-B',
                          bounds=bounds_param,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Generate pulse from best parametric solution
    if best_params is not None:
        best_pulse = _parametric_pulse(best_params, dt, n_steps, T_total)
    else:
        # Fallback: seed pulse
        t = (np.arange(n_steps) + 0.5) * dt - T_total / 2.0
        sigma_seed = T_total / 6.0
        A_seed = 0.35
        beta_seed = 0.08
        I_base = A_seed * np.exp(-t**2 / (2 * sigma_seed**2))
        dI_dt = -A_seed * t / sigma_seed**2 * np.exp(-t**2 / (2 * sigma_seed**2))
        Q_drag = -beta_seed * dI_dt
        best_pulse = np.column_stack([I_base, Q_drag])
    
    # Enforce constraints
    amps = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
    if np.max(amps) > amp_bound:
        best_pulse *= (amp_bound / np.max(amps))
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
    
    # Phase 3: Full-space local refinement with time budget
    if time.time() < deadline - 30:
        def objective_direct(p_flat):
            p = p_flat.reshape(80, 2)
            amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
            amp_pen = np.sum(np.maximum(0, amps - amp_bound)**2)
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
            if amp_pen > 1e-12 or slew_pen > 1e-12:
                return 1000.0 + 100.0 * (amp_pen + slew_pen)
            return _evaluate_pulse(p)
        
        x0 = best_pulse.flatten()
        bounds_full = [(-amp_bound, amp_bound)] * 160
        
        try:
            res = minimize(objective_direct, x0, method='L-BFGS-B', bounds=bounds_full,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9,
                                   'maxfun': 2000})
            if hasattr(res, 'fun') and res.fun < best_score:
                best_score = res.fun
                best_pulse = res.x.reshape(80, 2)
        except Exception:
            pass
        
        # Final validity check and projection
        if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
            amps = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
            if np.max(amps) > amp_bound:
                best_pulse *= (amp_bound / np.max(amps))
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
            
            if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
                # Fallback to seed
                t = (np.arange(n_steps) + 0.5) * dt - T_total / 2.0
                sigma_seed = T_total / 6.0
                A_seed = 0.35
                beta_seed = 0.08
                I_base = A_seed * np.exp(-t**2 / (2 * sigma_seed**2))
                dI_dt = -A_seed * t / sigma_seed**2 * np.exp(-t**2 / (2 * sigma_seed**2))
                Q_drag = -beta_seed * dI_dt
                best_pulse = np.column_stack([I_base, Q_drag])
                amps = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
                if np.max(amps) > amp_bound:
                    best_pulse *= (amp_bound / np.max(amps))
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

    return best_pulse
