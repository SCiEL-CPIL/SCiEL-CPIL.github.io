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

    # DRAG pulse with additional shape parameters
    def make_drag_pulse(A, sigma, beta, drag_sigma, phase):
        t = t_samples - t_center
        I_base = A * np.exp(-t**2 / (2 * sigma**2))
        dI_dt = -A * t / sigma**2 * np.exp(-t**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        # Add a small phase component
        Q_phase = A * phase * np.sin(np.pi * t / T_total) * np.exp(-t**2 / (2 * drag_sigma**2))
        pulse = np.column_stack([I_base, Q_drag + Q_phase])
        
        # Enforce amplitude bound
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        if np.max(amps) > amp_bound:
            pulse *= amp_bound / np.max(amps)
        
        # Enforce slew rate bound
        max_slew = 0
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(pulse[i] - prev)
            slew = diff / dt
            if slew > max_slew:
                max_slew = slew
            prev = pulse[i]
        diff = np.linalg.norm(pulse[-1])
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        
        if max_slew > slew_bound:
            pulse *= slew_bound / max_slew
        
        return pulse

    # Seed: standard DRAG
    sigma_seed = T_total / 6.0
    pulse_seed = make_drag_pulse(0.35, sigma_seed, 0.0, sigma_seed, 0.0)
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)

    # Phase 1: Differential Evolution on 5 parameters (A, sigma, beta, drag_sigma, phase)
    def objective(params):
        if time.time() > deadline - 5.0:
            return float('inf')
        A, sigma, beta, drag_sigma, phase = params
        if A < 0 or A > 1 or sigma < 2 or sigma > 20 or drag_sigma < 2 or drag_sigma > 20:
            return float('inf')
        p = make_drag_pulse(A, sigma, beta, drag_sigma, phase)
        return evaluate_pulse(p)

    if time.time() < deadline - 60.0:
        try:
            result = differential_evolution(objective,
                                            [(0.2, 0.5), (3, 15), (-0.5, 0.5), (3, 15), (-0.1, 0.1)],
                                            maxiter=100, popsize=20, tol=1e-8,
                                            mutation=(0.3, 0.9), recombination=0.8,
                                            seed=42, polish=True)
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_drag_pulse(*result.x)
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 30.0:
        def objective_nm(params):
            if time.time() > deadline - 5.0:
                return float('inf')
            A, sigma, beta, drag_sigma, phase = params
            if A < 0 or A > 1 or sigma < 1 or sigma > 25 or drag_sigma < 1 or drag_sigma > 25:
                return float('inf')
            p = make_drag_pulse(A, sigma, beta, drag_sigma, phase)
            return evaluate_pulse(p)
        
        # Try from current best and a few other starting points
        starts = [
            [0.35, sigma_seed, 0.0, sigma_seed, 0.0],
            [0.30, 5.0, 0.1, 5.0, 0.0],
            [0.40, 6.0, -0.1, 6.0, 0.05],
            [0.35, 7.0, 0.05, 7.0, -0.05],
        ]
        
        for s in starts:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(objective_nm, s, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_score:
                    best_score = res.fun
                    best_pulse = make_drag_pulse(*res.x)
            except Exception:
                pass

    # Phase 3: Direct optimization of a subset of amplitudes for fine-tuning
    # Represent pulse as Gaussian + linear correction on each sample
    if time.time() < deadline - 20.0:
        def make_fine_pulse(params):
            # params: [A, sigma, beta, c1, c2, c3]
            # c1, c2, c3 are small corrections to I, Q at center
            A, sigma, beta, c1, c2, c3 = params
            t = t_samples - t_center
            I_base = A * np.exp(-t**2 / (2 * sigma**2))
            dI_dt = -A * t / sigma**2 * np.exp(-t**2 / (2 * sigma**2))
            Q_base = -beta * dI_dt
            
            # Add small Gaussian bumps for fine-tuning
            bump_I = c1 * np.exp(-t**2 / (2 * 3.0**2))
            bump_Q = c2 * np.exp(-t**2 / (2 * 3.0**2))
            bump_I2 = c3 * np.exp(-t**2 / (2 * 5.0**2))
            
            pulse = np.column_stack([I_base + bump_I + bump_I2, Q_base + bump_Q])
            
            # Enforce amplitude bound
            amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
            if np.max(amps) > amp_bound:
                pulse *= amp_bound / np.max(amps)
            
            # Enforce slew rate bound
            max_slew = 0
            prev = np.array([0.0, 0.0])
            for i in range(n_steps):
                diff = np.linalg.norm(pulse[i] - prev)
                slew = diff / dt
                if slew > max_slew:
                    max_slew = slew
                prev = pulse[i]
            diff = np.linalg.norm(pulse[-1])
            slew = diff / dt
            if slew > max_slew:
                max_slew = slew
            
            if max_slew > slew_bound:
                pulse *= slew_bound / max_slew
            
            return pulse

        def objective_fine(params):
            if time.time() > deadline - 5.0:
                return float('inf')
            A, sigma, beta, c1, c2, c3 = params
            if A < 0 or A > 1 or sigma < 2 or sigma > 20:
                return float('inf')
            p = make_fine_pulse(params)
            return evaluate_pulse(p)
        
        # Start from current best
        if best_score < float('inf'):
            try:
                res = minimize(objective_fine, [0.35, sigma_seed, 0.0, 0.0, 0.0, 0.0],
                              method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-11})
                if res.fun < best_score:
                    best_score = res.fun
                    best_pulse = make_fine_pulse(res.x)
            except Exception:
                pass

    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        best_pulse = make_drag_pulse(0.35, sigma_seed, 0.0, sigma_seed, 0.0)

    return best_pulse
