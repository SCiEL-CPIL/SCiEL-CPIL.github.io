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

def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate the quantum dynamics and return the final unitary matrix."""
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

def _enforce_constraints(pulse, amp_bound, slew_bound, dt, n_steps):
    """Scale pulse to satisfy amplitude and slew constraints."""
    p = pulse.copy()
    amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        p = p * (amp_bound / max_amp)
    
    prev = np.array([0.0, 0.0])
    max_slew = 0.0
    for i in range(n_steps):
        diff = np.linalg.norm(p[i] - prev)
        slew = diff / dt
        max_slew = max(max_slew, slew)
        prev = p[i]
    diff = np.linalg.norm(-prev)
    max_slew = max(max_slew, diff / dt)
    
    if max_slew > slew_bound:
        p = p * (slew_bound / max_slew)
    
    return p

def _make_drag_pulse(t_samples, t_center, sigma, beta, amp_scale=1.0):
    """Create a DRAG-style Gaussian pulse."""
    A = np.pi * amp_scale / (sigma * np.sqrt(2.0 * np.pi))
    I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
    dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
    Q_drag = -beta * dI_dt
    p = np.column_stack([I_base, Q_drag])
    return p

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0

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
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    # Stage 1: Coarse grid search over DRAG parameters
    best_pulse = None
    best_score = float('inf')
    
    sigma_vals = np.linspace(T_total / 10, T_total / 4.5, 8)
    beta_vals = np.linspace(-0.25, 0.35, 12)
    
    for sigma in sigma_vals:
        if time.time() > deadline - 150:
            break
        for beta in beta_vals:
            if time.time() > deadline - 150:
                break
            p_raw = _make_drag_pulse(t_samples, t_center, sigma, beta)
            p = _enforce_constraints(p_raw, amp_bound, slew_bound, dt, n_steps)
            s = evaluate_pulse(p)
            if s < best_score:
                best_score = s
                best_pulse = p.copy()
    
    # If no valid pulse found from grid, use seed
    if best_pulse is None:
        sigma_seed = T_total / 6.0
        beta_seed = 0.15
        p_raw = _make_drag_pulse(t_samples, t_center, sigma_seed, beta_seed)
        best_pulse = _enforce_constraints(p_raw, amp_bound, slew_bound, dt, n_steps)
        best_score = evaluate_pulse(best_pulse)

    # Stage 2: Low-dimensional Hermite-Gaussian basis optimization
    t_norm = (t_samples - t_center) / (T_total / 6.0)
    
    def hermite_gauss(k, x):
        if k == 0:
            return np.exp(-0.5 * x**2)
        elif k == 1:
            return x * np.exp(-0.5 * x**2)
        elif k == 2:
            return (x**2 - 1) * np.exp(-0.5 * x**2)
        else:
            return np.zeros_like(x)
    
    hg0 = hermite_gauss(0, t_norm)
    hg1 = hermite_gauss(1, t_norm)
    hg2 = hermite_gauss(2, t_norm)
    
    hg0_norm = np.linalg.norm(hg0) * np.sqrt(dt)
    hg1_norm = np.linalg.norm(hg1) * np.sqrt(dt)
    hg2_norm = np.linalg.norm(hg2) * np.sqrt(dt)
    
    hg0 = hg0 / hg0_norm
    hg1 = hg1 / hg1_norm
    hg2 = hg2 / hg2_norm
    
    def make_pulse_from_coeffs(coeffs):
        I_p = coeffs[0] * hg0 + coeffs[1] * hg1 + coeffs[2] * hg2
        Q_p = coeffs[3] * hg0 + coeffs[4] * hg1 + coeffs[5] * hg2
        p = np.column_stack([I_p, Q_p])
        return _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
    
    B_I = np.column_stack([hg0, hg1, hg2])
    B_Q = np.column_stack([hg0, hg1, hg2])
    
    cI_seed = np.linalg.lstsq(B_I, best_pulse[:, 0], rcond=None)[0]
    cQ_seed = np.linalg.lstsq(B_Q, best_pulse[:, 1], rcond=None)[0]
    init_coeffs = np.concatenate([cI_seed, cQ_seed])
    
    best_pulse_basis = make_pulse_from_coeffs(init_coeffs)
    best_score_basis = evaluate_pulse(best_pulse_basis)
    if best_score_basis < best_score:
        best_score = best_score_basis
        best_pulse = best_pulse_basis.copy()
    
    # Differential evolution on 6-parameter space
    if time.time() < deadline - 100:
        def objective(coeffs):
            if time.time() > deadline - 20:
                return float('inf')
            p = make_pulse_from_coeffs(coeffs)
            return evaluate_pulse(p)
        
        bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                  (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
        
        try:
            de_result = differential_evolution(objective, bounds, 
                                               seed=42, maxiter=30, 
                                               tol=1e-8, popsize=8,
                                               init='latinhypercube')
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_pulse = make_pulse_from_coeffs(de_result.x)
                init_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Stage 3: Direct time-domain refinement using L-BFGS-B on full 160-dim vector
    if time.time() < deadline - 60:
        def objective_flat(flat_params):
            if time.time() > deadline - 15:
                return float('inf')
            p = flat_params.reshape(n_steps, 2)
            p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
            return evaluate_pulse(p)
        
        init_flat = best_pulse.flatten()
        try:
            result = minimize(objective_flat, init_flat, method='L-BFGS-B',
                             options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8,
                                      'maxls': 40})
            if result.fun < best_score:
                p_opt = _enforce_constraints(result.x.reshape(n_steps, 2), amp_bound, slew_bound, dt, n_steps)
                s_opt = evaluate_pulse(p_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_pulse = p_opt.copy()
        except Exception:
            pass
    
    # Stage 4: Second refinement pass
    if time.time() < deadline - 30:
        def objective_flat2(flat_params):
            if time.time() > deadline - 10:
                return float('inf')
            p = flat_params.reshape(n_steps, 2)
            p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
            return evaluate_pulse(p)
        
        init_flat2 = best_pulse.flatten()
        try:
            result2 = minimize(objective_flat2, init_flat2, method='L-BFGS-B',
                              options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10,
                                       'maxls': 30})
            if result2.fun < best_score:
                p_opt2 = _enforce_constraints(result2.x.reshape(n_steps, 2), amp_bound, slew_bound, dt, n_steps)
                s_opt2 = evaluate_pulse(p_opt2)
                if s_opt2 < best_score:
                    best_score = s_opt2
                    best_pulse = p_opt2.copy()
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        best_pulse = _enforce_constraints(best_pulse, amp_bound, slew_bound, dt, n_steps)
    
    return best_pulse
