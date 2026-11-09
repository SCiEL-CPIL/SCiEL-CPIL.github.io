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
    deadline = start_time + 300.0

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
        A = np.sqrt(np.pi / 2) / sigma
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        return np.column_stack([I_base, Q_drag])

    # Basis expansion: use a small set of basis functions to represent the pulse
    # We use a Gaussian window times a few polynomial/Fourier terms
    # I(t) = A * g(t) * (1 + c1*h1(t) + c2*h2(t))
    # Q(t) = A * g(t) * (d1*h1(t) + d2*h2(t))
    # where g(t) is the Gaussian envelope, h1, h2 are orthonormal basis functions within the window

    # Define basis functions
    g = np.exp(-(t_samples - t_center)**2 / (2 * (T_total/6.0)**2))
    g_norm = np.sqrt(np.sum(g**2))
    g = g / g_norm  # normalize

    # h1: first Hermite-like function (linear within window)
    t_norm = (t_samples - t_center) / (T_total/6.0)
    h1 = t_norm * g
    h1_norm = np.sqrt(np.sum(h1**2))
    h1 = h1 / h1_norm

    # h2: second Hermite-like function (quadratic within window)
    h2 = (t_norm**2 - 0.5) * g
    h2_norm = np.sqrt(np.sum(h2**2))
    h2 = h2 / h2_norm

    # h3: third Hermite-like function (cubic within window)
    h3 = (t_norm**3 - 1.5*t_norm) * g
    h3_norm = np.sqrt(np.sum(h3**2))
    h3 = h3 / h3_norm

    def make_pulse_from_basis(params):
        """Convert basis coefficients to a pulse."""
        A = params[0]
        c1, c2, c3 = params[1], params[2], params[3]
        d1, d2, d3 = params[4], params[5], params[6]
        
        I_pulse = A * (g + c1*h1 + c2*h2 + c3*h3)
        Q_pulse = A * (d1*h1 + d2*h2 + d3*h3)
        
        pulse_IQ = np.column_stack([I_pulse, Q_pulse])
        
        # Clamp to validity constraints
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        max_amp = np.max(amplitudes)
        if max_amp > amp_bound:
            scale_factor = amp_bound / max_amp
            pulse_IQ = pulse_IQ * scale_factor
        
        # Check slew rate and clamp if needed
        prev = np.array([0.0, 0.0])
        max_slew = 0
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
            scale_factor = slew_bound / max_slew
            pulse_IQ = pulse_IQ * scale_factor
        
        return pulse_IQ

    # Seed pulse
    sigma_seed = T_total / 6.0
    pulse_seed = make_drag_pulse(sigma_seed, 0.0)
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)

    # First: coarse grid search on DRAG parameters to get a good starting point
    sigma_range = np.linspace(T_total / 9.0, T_total / 4.5, 8)
    beta_range = np.linspace(-0.15, 0.15, 8)
    best_sigma, best_beta = sigma_seed, 0.0
    
    for sigma in sigma_range:
        if time.time() > deadline - 150.0:
            break
        for beta in beta_range:
            if time.time() > deadline - 150.0:
                break
            pulse_IQ = make_drag_pulse(sigma, beta)
            score = evaluate_pulse(pulse_IQ)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ.copy()
                best_sigma, best_beta = sigma, beta

    # Initialize basis parameters from the best DRAG pulse
    # Project the best DRAG pulse onto the basis
    I_best = best_pulse[:, 0]
    Q_best = best_pulse[:, 1]
    
    A_init = np.sqrt(np.sum(I_best**2))
    if A_init < 1e-10:
        A_init = 0.3
    c1_init = np.dot(I_best, h1) / A_init
    c2_init = np.dot(I_best, h2) / A_init
    c3_init = np.dot(I_best, h3) / A_init
    d1_init = np.dot(Q_best, h1) / A_init
    d2_init = np.dot(Q_best, h2) / A_init
    d3_init = np.dot(Q_best, h3) / A_init
    
    init_params = [A_init, c1_init, c2_init, c3_init, d1_init, d2_init, d3_init]

    # Differential evolution on the basis parameters
    if time.time() < deadline - 60.0:
        bounds = [(0.1, 0.4), (-0.5, 0.5), (-0.3, 0.3), (-0.2, 0.2),
                  (-0.5, 0.5), (-0.3, 0.3), (-0.2, 0.2)]
        
        def objective(params):
            if time.time() > deadline - 10.0:
                return float('inf')
            pulse_IQ = make_pulse_from_basis(params)
            return evaluate_pulse(pulse_IQ)
        
        try:
            result = differential_evolution(objective, bounds, maxiter=50, 
                                            popsize=10, tol=1e-6, 
                                            mutation=(0.3, 0.8), recombination=0.7,
                                            seed=42, polish=False)
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_basis(result.x)
        except Exception:
            pass

    # Nelder-Mead polish on the basis parameters
    if time.time() < deadline - 20.0:
        def objective(params):
            if time.time() > deadline - 5.0:
                return float('inf')
            pulse_IQ = make_pulse_from_basis(params)
            return evaluate_pulse(pulse_IQ)
        
        try:
            result = minimize(objective, init_params, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-9})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_basis(result.x)
        except Exception:
            pass

    # Final Nelder-Mead on DRAG parameters as a fallback check
    if time.time() < deadline - 10.0:
        def objective_drag(params):
            if time.time() > deadline - 2.0:
                return float('inf')
            p = make_drag_pulse(params[0], params[1])
            return evaluate_pulse(p)
        
        try:
            result = minimize(objective_drag, [best_sigma, best_beta], method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_drag_pulse(result.x[0], result.x[1])
        except Exception:
            pass

    return best_pulse
