import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize

def _precompute_operators():
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

def _enforce_constraints(pulse, amp_bound, slew_bound, dt, n_steps):
    """Enforce amplitude and slew-rate constraints by scaling."""
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.max(amps) > amp_bound:
        pulse = pulse * (amp_bound / np.max(amps))
    
    # Enforce slew rate bound
    max_slew = 0.0
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
        pulse = pulse * (slew_bound / max_slew)
    
    return pulse

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0  # safety margin before 330s limit

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

    def evaluate_pulse(pulse_IQ):
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

    # Basis expansion approach: represent pulse as combination of basis functions
    # Basis functions: Gaussian envelope and its derivatives (Hermite-Gaussian-like)
    # I(t) = c1 * G(t) + c2 * G'(t) + c3 * G''(t)
    # Q(t) = c4 * G'(t) + c5 * G''(t) + c6 * G'''(t)
    # where G(t) = exp(-t^2/(2*sigma^2))
    
    def make_basis_pulse(params, sigma):
        """Generate pulse from basis coefficients."""
        c1, c2, c3, c4, c5, c6 = params
        
        # Gaussian and derivatives
        G = np.exp(-t**2 / (2 * sigma**2))
        G1 = -t / sigma**2 * G  # G'
        G2 = (t**2 / sigma**4 - 1 / sigma**2) * G  # G''
        G3 = (-t**3 / sigma**6 + 3*t / sigma**4) * G  # G'''
        
        I = c1 * G + c2 * G1 + c3 * G2
        Q = c4 * G1 + c5 * G2 + c6 * G3
        
        pulse = np.column_stack([I, Q])
        pulse = _enforce_constraints(pulse, amp_bound, slew_bound, dt, n_steps)
        return pulse

    # Seed: standard DRAG pulse converted to basis representation
    sigma_seed = T_total / 6.0
    A_seed = np.sqrt(np.pi / 2) / sigma_seed
    I_seed = A_seed * np.exp(-t**2 / (2 * sigma_seed**2))
    dI_dt = -A_seed * t / sigma_seed**2 * np.exp(-t**2 / (2 * sigma_seed**2))
    pulse_seed = np.column_stack([I_seed, np.zeros(n_steps)])
    pulse_seed = _enforce_constraints(pulse_seed, amp_bound, slew_bound, dt, n_steps)
    
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)

    # Phase 1: Differential evolution on basis coefficients + sigma
    # 7 parameters: [c1, c2, c3, c4, c5, c6, sigma]
    def objective_basis(params):
        if time.time() > deadline - 10.0:
            return float('inf')
        c1, c2, c3, c4, c5, c6, sigma = params
        if sigma < 2.0 or sigma > 20.0:
            return float('inf')
        p = make_basis_pulse(params[:6], sigma)
        return evaluate_pulse(p)

    if time.time() < deadline - 80.0:
        try:
            result = differential_evolution(objective_basis,
                                            [(-0.5, 0.5), (-0.5, 0.5), (-0.1, 0.1),
                                             (-0.5, 0.5), (-0.1, 0.1), (-0.1, 0.1),
                                             (3.0, 15.0)],
                                            maxiter=60, popsize=20, tol=1e-7,
                                            mutation=(0.3, 0.9), recombination=0.8,
                                            seed=42, polish=True)
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_basis_pulse(result.x[:6], result.x[6])
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement on basis parameters
    if time.time() < deadline - 30.0:
        def objective_basis_nm(params):
            if time.time() > deadline - 10.0:
                return float('inf')
            c1, c2, c3, c4, c5, c6, sigma = params
            if sigma < 1.0 or sigma > 25.0:
                return float('inf')
            p = make_basis_pulse(params[:6], sigma)
            return evaluate_pulse(p)
        
        # Multiple starting points
        starts = [
            [A_seed, 0.0, 0.0, 0.0, 0.0, 0.0, sigma_seed],
            [A_seed * 0.9, 0.0, 0.0, -0.05, 0.0, 0.0, sigma_seed],
            [A_seed * 1.1, 0.0, 0.0, 0.05, 0.0, 0.0, sigma_seed * 0.9],
            [A_seed, 0.05, 0.0, -0.1, 0.0, 0.0, sigma_seed * 1.1],
            [A_seed * 0.95, -0.03, 0.01, 0.08, -0.02, 0.0, sigma_seed * 0.95],
        ]
        
        for s in starts:
            if time.time() > deadline - 15.0:
                break
            try:
                res = minimize(objective_basis_nm, s, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_score:
                    best_score = res.fun
                    best_pulse = make_basis_pulse(res.x[:6], res.x[6])
            except Exception:
                pass

    # Phase 3: Direct fine-tuning of individual samples near the center
    # Use the best basis pulse as a starting point and optimize a subset of samples
    if time.time() < deadline - 20.0:
        # Identify the center region (steps 30-50 roughly)
        center_idx = np.arange(30, 51)
        n_center = len(center_idx)
        
        # Create a pulse that matches best_pulse outside center, parameterized inside
        def make_fine_pulse(params):
            p = best_pulse.copy()
            # params: 2*n_center values for I and Q in center region
            p[center_idx, 0] = params[:n_center]
            p[center_idx, 1] = params[n_center:2*n_center]
            p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
            return p
        
        def objective_fine(params):
            if time.time() > deadline - 10.0:
                return float('inf')
            p = make_fine_pulse(params)
            return evaluate_pulse(p)
        
        # Start from current best
        init_params = np.concatenate([best_pulse[center_idx, 0], best_pulse[center_idx, 1]])
        
        try:
            res = minimize(objective_fine, init_params, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = make_fine_pulse(res.x)
        except Exception:
            pass

    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Fallback to seed
        best_pulse = pulse_seed.copy()

    return best_pulse
