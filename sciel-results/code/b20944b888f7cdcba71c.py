import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 285.0  # 285s safety margin (300s limit)
    
    # Parameters
    dt = 0.5
    n_steps = 80
    T_total = 40.0
    
    anharmonicity = -2 * np.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    
    # Time samples
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0
    
    # Precompute operators
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    
    # Anharmonicity term
    anham_term = 0.5 * anharmonicity * n_op @ (n_op - I3)
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    # Lindblad superoperator
    lindblad_super = (np.kron(L1.conj(), L1) - 0.5 * (np.kron(I3, L1_dag_L1) + np.kron(L1_dag_L1.T, I3))
                     + np.kron(L2.conj(), L2) - 0.5 * (np.kron(I3, L2_dag_L2) + np.kron(L2_dag_L2.T, I3)))
    
    # Hamiltonian basis terms
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    
    def ham_super(H_comp):
        return -1j * (np.kron(I3, H_comp) - np.kron(H_comp.T, I3))
    
    n_super = ham_super(n_op)
    anham_super = ham_super(anham_term)
    I_drive_super = ham_super(0.5 * a_plus_adag)
    Q_drive_super = ham_super(0.5 * 1j * a_minus_adag)
    
    # Detuning and scale cases
    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]
    
    # Precompute Liouvillian bases
    liouvillian_bases = {}
    for det in detunings:
        for sc in scales:
            base = det * n_super + anham_super + lindblad_super
            liouvillian_bases[(det, sc)] = base
    
    # Initial states and targets
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    target_states = [X @ psi for psi in initial_states]
    
    def check_validity(pulse_IQ):
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
    
    def simulate_one_case(pulse_IQ, det, sc, rho_init):
        base = liouvillian_bases[(det, sc)]
        rho = rho_init.copy()
        for i in range(n_steps):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            L_super = base + sc * (I_val * I_drive_super + Q_val * Q_drive_super)
            rho_flat = rho.flatten()
            rho_flat = expm(L_super * dt) @ rho_flat
            rho = rho_flat.reshape((3, 3))
        return rho
    
    def evaluate_pulse(pulse_IQ):
        if not check_validity(pulse_IQ):
            return float('inf')
        scores = []
        for det in detunings:
            for sc in scales:
                fidelities = []
                for idx, psi in enumerate(initial_states):
                    rho_init = np.outer(psi, psi.conj())
                    rho_final = simulate_one_case(pulse_IQ, det, sc, rho_init)
                    phi = target_states[idx]
                    F = np.vdot(phi, rho_final @ phi).real
                    F = max(0.0, min(1.0, F))
                    fidelities.append(F)
                mean_fid = np.mean(fidelities)
                infid = 1.0 - mean_fid
                scores.append(infid)
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        return mean_score + max_score
    
    # Basis expansion setup
    sigma_seed = T_total / 6.0
    g = np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    g_norm = np.sqrt(np.sum(g**2))
    g = g / g_norm
    
    t_norm = (t_samples - t_center) / sigma_seed
    h1 = t_norm * g
    h1_norm = np.sqrt(np.sum(h1**2))
    h1 = h1 / h1_norm
    
    h2 = (t_norm**2 - 0.5) * g
    h2_norm = np.sqrt(np.sum(h2**2))
    h2 = h2 / h2_norm
    
    h3 = (t_norm**3 - 1.5 * t_norm) * g
    h3_norm = np.sqrt(np.sum(h3**2))
    h3 = h3 / h3_norm
    
    def make_pulse_from_basis(params):
        A = params[0]
        c1, c2, c3 = params[1], params[2], params[3]
        d1, d2, d3 = params[4], params[5], params[6]
        
        I_pulse = A * (g + c1 * h1 + c2 * h2 + c3 * h3)
        Q_pulse = A * (d1 * h1 + d2 * h2 + d3 * h3)
        
        pulse_IQ = np.column_stack([I_pulse, Q_pulse])
        
        # Clamp to validity
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        max_amp = np.max(amplitudes)
        if max_amp > amp_bound:
            pulse_IQ = pulse_IQ * (amp_bound / max_amp)
        
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
            pulse_IQ = pulse_IQ * (slew_bound / max_slew)
        
        return pulse_IQ
    
    # Seed pulse
    sigma_seed_val = T_total / 6.0
    A_seed = np.sqrt(np.pi / 2) / sigma_seed_val
    I_seed = A_seed * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed_val**2))
    Q_seed = np.zeros(n_steps)
    pulse_seed = np.column_stack([I_seed, Q_seed])
    
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)
    
    # Project seed onto basis
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
    
    # Objective function for optimization
    def objective(params):
        pulse_IQ = make_pulse_from_basis(params)
        return evaluate_pulse(pulse_IQ)
    
    # Phase 1: Differential Evolution on basis parameters
    if time.time() < deadline - 120.0:
        bounds = [(0.1, 0.35), (-0.5, 0.5), (-0.3, 0.3), (-0.2, 0.2),
                  (-0.3, 0.3), (-0.2, 0.2), (-0.1, 0.1)]
        
        try:
            de_result = differential_evolution(
                objective, bounds,
                x0=init_params,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                updating='immediate'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_pulse = make_pulse_from_basis(de_result.x)
        except Exception:
            pass
    
    # Phase 2: Local refinement with BFGS
    if time.time() < deadline - 30.0:
        try:
            # Start from best found so far
            current_pulse = best_pulse
            I_cur = current_pulse[:, 0]
            Q_cur = current_pulse[:, 1]
            A_cur = np.sqrt(np.sum(I_cur**2))
            if A_cur < 1e-10:
                A_cur = 0.3
            c1_cur = np.dot(I_cur, h1) / A_cur
            c2_cur = np.dot(I_cur, h2) / A_cur
            c3_cur = np.dot(I_cur, h3) / A_cur
            d1_cur = np.dot(Q_cur, h1) / A_cur
            d2_cur = np.dot(Q_cur, h2) / A_cur
            d3_cur = np.dot(Q_cur, h3) / A_cur
            start_params = [A_cur, c1_cur, c2_cur, c3_cur, d1_cur, d2_cur, d3_cur]
            
            res = minimize(
                objective, start_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = make_pulse_from_basis(res.x)
        except Exception:
            pass
    
    # Phase 3: Try DRAG parameter sweep for robustness
    if time.time() < deadline - 20.0:
        def make_drag_pulse(sigma, beta):
            A = np.sqrt(np.pi / 2) / sigma
            I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
            dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
            Q_drag = -beta * dI_dt
            return np.column_stack([I_base, Q_drag])
        
        sigma_vals = [T_total / 6.0, T_total / 5.5, T_total / 6.5, T_total / 5.0, T_total / 7.0]
        beta_vals = [0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]
        
        for sigma in sigma_vals:
            if time.time() > deadline - 10.0:
                break
            for beta in beta_vals:
                if time.time() > deadline - 10.0:
                    break
                pulse_candidate = make_drag_pulse(sigma, beta)
                if not check_validity(pulse_candidate):
                    continue
                score = evaluate_pulse(pulse_candidate)
                if score < best_score:
                    best_score = score
                    best_pulse = pulse_candidate.copy()
    
    return best_pulse
