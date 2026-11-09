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
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
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

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill

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

    def make_drag_pulse(sigma, beta):
        """Create a DRAG-style Gaussian pulse."""
        A = np.sqrt(np.pi / 2) / sigma
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        # Ensure edges are near zero
        for i in range(n_steps):
            if I_base[i] < 0.001:
                I_base[i] = 0.0
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        pulse = np.column_stack([I_base, Q_drag])
        
        # Clamp to validity
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        if np.max(amps) > amp_bound:
            pulse *= amp_bound / np.max(amps)
        
        # Check slew
        prev = np.array([0.0, 0.0])
        max_slew = 0
        for i in range(n_steps):
            diff = np.linalg.norm(pulse[i] - prev)
            slew = diff / dt
            max_slew = max(max_slew, slew)
            prev = pulse[i]
        diff = np.linalg.norm(-prev)
        max_slew = max(max_slew, diff / dt)
        
        if max_slew > slew_bound:
            pulse *= slew_bound / max_slew
        
        return pulse

    # Initial seed: DRAG with physics-informed beta
    sigma_seed = T_total / 6.0
    beta_seed = 0.12  # Physics: beta ~ |anharmonicity| / (2*pi*freq) ~ 0.25/2 ~ 0.12
    pulse_seed = make_drag_pulse(sigma_seed, beta_seed)
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)

    # Smoother objective using soft-max for the max term
    def evaluate_pulse_smooth(pulse_IQ, beta_sm=50.0):
        """Evaluate with smooth surrogate for max term."""
        if not _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
            return float('inf')
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        scores = np.array(scores)
        mean_s = np.mean(scores)
        # Soft-max: approximates max but is smooth
        m = np.max(scores)
        soft_max = m + np.log(np.sum(np.exp(scores - m))) / len(scores)
        return mean_s + soft_max

    # Coarse search over sigma and beta with better ranges
    sigma_vals = np.linspace(T_total / 8, T_total / 4.5, 10)
    beta_vals = np.linspace(-0.15, 0.25, 11)
    
    for sigma in sigma_vals:
        if time.time() > deadline - 150:
            break
        for beta in beta_vals:
            if time.time() > deadline - 150:
                break
            p = make_drag_pulse(sigma, beta)
            s = evaluate_pulse_smooth(p, beta_sm=50.0)
            if s < best_score:
                # Verify with exact score
                s_exact = evaluate_pulse(p)
                if s_exact < best_score:
                    best_score = s_exact
                    best_pulse = p.copy()

    # Basis expansion with 5 terms for richer pulse shapes
    # Use Gaussian envelope with sigma tuned for the pulse duration
    g_sigma = T_total / 5.0
    g = np.exp(-(t_samples - t_center)**2 / (2 * g_sigma**2))
    g /= np.linalg.norm(g)
    
    t_norm = (t_samples - t_center) / g_sigma
    
    # Build 5 orthogonal basis functions
    basis = []
    basis.append(g.copy())
    for k in range(1, 5):
        raw = (t_norm**k) * g
        for b in basis:
            raw = raw - np.dot(raw, b) * b
        norm = np.linalg.norm(raw)
        if norm > 1e-10:
            raw /= norm
            basis.append(raw)
    
    n_basis = len(basis)  # Should be 5
    n_params = 2 * n_basis  # I and Q coefficients
    
    # Project best DRAG pulse onto basis
    I_best = best_pulse[:, 0]
    Q_best = best_pulse[:, 1]
    
    init_coeffs = np.zeros(n_params)
    for k in range(n_basis):
        init_coeffs[k] = np.dot(I_best, basis[k])
        init_coeffs[n_basis + k] = np.dot(Q_best, basis[k])
    
    def make_pulse_from_basis(coeffs):
        """Reconstruct pulse from basis coefficients."""
        I_p = np.zeros(n_steps)
        Q_p = np.zeros(n_steps)
        for k in range(n_basis):
            I_p += coeffs[k] * basis[k]
            Q_p += coeffs[n_basis + k] * basis[k]
        p = np.column_stack([I_p, Q_p])
        
        # Clamp to validity
        amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            p *= amp_bound / max_amp
        
        prev = np.array([0.0, 0.0])
        max_slew = 0
        for i in range(n_steps):
            diff = np.linalg.norm(p[i] - prev)
            max_slew = max(max_slew, diff / dt)
            prev = p[i]
        max_slew = max(max_slew, np.linalg.norm(-prev) / dt)
        
        if max_slew > slew_bound:
            p *= slew_bound / max_slew
        
        return p
    
    # Differential Evolution for global search on basis coefficients
    if time.time() < deadline - 80:
        bounds = [(-0.5, 0.5)] * n_params
        
        def de_objective(coeffs):
            if time.time() > deadline - 10:
                return float('inf')
            p = make_pulse_from_basis(coeffs)
            return evaluate_pulse_smooth(p, beta_sm=50.0)
        
        try:
            de_result = differential_evolution(de_objective, bounds, maxiter=30, 
                                               tol=1e-6, seed=42, polish=False,
                                               workers=1)
            if de_result.fun < best_score:
                p_de = make_pulse_from_basis(de_result.x)
                s_exact = evaluate_pulse(p_de)
                if s_exact < best_score:
                    best_score = s_exact
                    best_pulse = p_de.copy()
                    init_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead local polish with smooth objective
    if time.time() < deadline - 40:
        def objective_smooth(coeffs):
            if time.time() > deadline - 5:
                return float('inf')
            p = make_pulse_from_basis(coeffs)
            return evaluate_pulse_smooth(p, beta_sm=50.0)
        
        try:
            nm_result = minimize(objective_smooth, init_coeffs, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
            if nm_result.fun < best_score:
                p_nm = make_pulse_from_basis(nm_result.x)
                s_exact = evaluate_pulse(p_nm)
                if s_exact < best_score:
                    best_score = s_exact
                    best_pulse = p_nm.copy()
                    init_coeffs = nm_result.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with exact objective
    if time.time() < deadline - 15:
        def objective_exact(coeffs):
            if time.time() > deadline - 3:
                return float('inf')
            p = make_pulse_from_basis(coeffs)
            return evaluate_pulse(p)
        
        try:
            final_result = minimize(objective_exact, init_coeffs, method='Nelder-Mead',
                                   options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12,
                                            'adaptive': True})
            if final_result.fun < best_score:
                p_final = make_pulse_from_basis(final_result.x)
                s_final = evaluate_pulse(p_final)
                if s_final < best_score:
                    best_score = s_final
                    best_pulse = p_final.copy()
        except Exception:
            pass
    
    return best_pulse
