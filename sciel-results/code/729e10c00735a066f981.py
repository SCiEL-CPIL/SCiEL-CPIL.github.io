import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0
    
    # Detuning and scale cases
    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # The 6 states as specified
    sqrt2 = np.sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([1, -1, 0], dtype=complex) / sqrt2,
        np.array([1, 1j, 0], dtype=complex) / sqrt2,
        np.array([1, -1j, 0], dtype=complex) / sqrt2
    ]
    
    # Target states for X gate
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([-1, 1, 0], dtype=complex) / sqrt2,
        np.array([1j, 1, 0], dtype=complex) / sqrt2,
        np.array([-1j, 1, 0], dtype=complex) / sqrt2
    ]
    
    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    
    # Precompute constant parts
    H_anharm = 0.5 * alpha * (n_op @ (n_op - np.eye(3)))
    
    # Dissipator superoperator (constant)
    dim = 3
    d2 = dim * dim
    L_diss_super = np.zeros((d2, d2), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L1_term = np.kron(L_op, L_op.conj())
        L2_term = 0.5 * (np.kron(L_dag @ L_op, np.eye(dim)) + np.kron(np.eye(dim), (L_dag @ L_op).T))
        L_diss_super += L1_term - L2_term
    
    # Precompute identity for kron
    I3 = np.eye(3)
    
    # Precompute Liouvillian parts for all 9 cases
    liouvillian_parts = {}
    for delta in deltas:
        for scale in scales:
            H0 = delta * n_op + H_anharm
            HI = 0.5 * scale * (a_op + a_dag)
            HQ = 0.5j * scale * (a_op - a_dag)
            
            H_term_0 = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
            H_term_I = -1j * (np.kron(HI, I3) - np.kron(I3, HI.T))
            H_term_Q = -1j * (np.kron(HQ, I3) - np.kron(I3, HQ.T))
            
            L_base_0 = H_term_0 + L_diss_super
            liouvillian_parts[(delta, scale)] = (L_base_0, H_term_I, H_term_Q)
    
    def simulate_pulse(I_Q, delta, scale):
        """
        Simulates the pulse and returns mean infidelity over 6 states.
        """
        L_base_0, H_term_I, H_term_Q = liouvillian_parts[(delta, scale)]
        
        # Precompute all evolution operators
        L_steps = []
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            L_total = L_base_0 + I * H_term_I + Q * H_term_Q
            L_steps.append(expm(L_total * dt))
        
        # Simulate all 6 states
        total_infidelity = 0.0
        for idx in range(6):
            psi0 = states[idx]
            rho0 = np.outer(psi0, psi0.conj())
            rho_vec = rho0.flatten()
            
            for i in range(N):
                rho_vec = L_steps[i] @ rho_vec
            
            rho = rho_vec.reshape(3, 3)
            target = targets[idx]
            proj = np.outer(target, target.conj())
            fid = np.real(np.trace(rho @ proj))
            fid = np.clip(fid, 0, 1)
            total_infidelity += (1.0 - fid)
        
        return total_infidelity / 6.0
    
    def check_constraints(I_Q):
        # Amplitude
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-12:
            return False
        # Slew
        if np.sqrt(I_Q[0,0]**2 + I_Q[0,1]**2) / dt > slew_bound + 1e-12:
            return False
        if np.sqrt(I_Q[-1,0]**2 + I_Q[-1,1]**2) / dt > slew_bound + 1e-12:
            return False
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-12:
            return False
        return True
    
    def calculate_score(I_Q):
        """
        Calculates the score: mean + max of the 9 cases.
        """
        case_scores = []
        for delta in deltas:
            for scale in scales:
                mean_inf = simulate_pulse(I_Q, delta, scale)
                case_scores.append(mean_inf)
        
        return np.mean(case_scores) + np.max(case_scores)
    
    def generate_pulse(sigma, eta, center=20.0):
        t = np.linspace(0, 40, N)
        A = np.sqrt(np.pi / 2) / sigma
        I = A * np.exp(-(t - center)**2 / (2 * sigma**2))
        dI = -(t - center) / (sigma**2) * I
        Q = eta * dI
        I_Q = np.column_stack((I, Q))
        return I_Q
    
    # Seed pulse
    sigma_seed = 40.0 / 6.0
    I_Q_seed = generate_pulse(sigma_seed, 0.0)
    
    # Check seed
    if not check_constraints(I_Q_seed):
        # Adjust seed if necessary
        pass
    
    best_I_Q = I_Q_seed.copy()
    best_score = calculate_score(best_I_Q)
    
    # Phase 1: Coarse Grid Search
    # Time budget: 40s
    grid_deadline = start_time + 40.0
    
    sigmas = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    etas = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
    
    for s in sigmas:
        for e in etas:
            if time.time() > grid_deadline:
                break
            I_Q = generate_pulse(s, e)
            if not check_constraints(I_Q):
                continue
            score = calculate_score(I_Q)
            if score < best_score:
                best_score = score
                best_I_Q = I_Q
        if time.time() > grid_deadline:
            break
    
    # Phase 2: Local optimization using gradient-free method
    # Time budget: remaining time minus 10s
    local_deadline = deadline - 10.0
    
    # Define objective for Nelder-Mead
    def objective(params):
        if time.time() > local_deadline:
            return 1e10
        sigma, eta = params
        if sigma <= 0.5 or sigma > 50:
            return 1e10
        I_Q = generate_pulse(sigma, eta)
        if not check_constraints(I_Q):
            return 1e10
        return calculate_score(I_Q)
    
    # Extract best parameters from grid
    # We need to track which (sigma, eta) gave best_score
    # Let's re-search around best to find parameters
    best_sigma, best_eta = sigma_seed, 0.0
    min_score = best_score
    for s in sigmas:
        for e in etas:
            if time.time() > local_deadline:
                break
            I_Q = generate_pulse(s, e)
            if not check_constraints(I_Q):
                continue
            score = calculate_score(I_Q)
            if score < min_score:
                min_score = score
                best_sigma, best_eta = s, e
        if time.time() > local_deadline:
            break
    
    # Run Nelder-Mead from multiple starting points
    from scipy.optimize import minimize
    
    nm_starts = [
        [best_sigma, best_eta],
        [6.0, -1.0],
        [8.0, -2.0],
        [10.0, -3.0],
        [12.0, -1.5],
        [sigma_seed, 0.0],
        [5.0, -2.5],
        [15.0, -2.0]
    ]
    
    # Remove duplicates
    unique_starts = []
    for sp in nm_starts:
        if sp not in unique_starts:
            unique_starts.append(sp)
    
    for sp in unique_starts:
        if time.time() > local_deadline:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_I_Q = generate_pulse(res.x[0], res.x[1])
        except Exception as e:
            pass
    
    # Final Constraint Check
    if not check_constraints(best_I_Q):
        best_I_Q = I_Q_seed.copy()
    
    return best_I_Q
