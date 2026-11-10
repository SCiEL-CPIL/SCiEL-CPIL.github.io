import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import math
import time

def pulse(target_gate, hardware_model):
    # Constants
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters
    try:
        chi = hardware_model.get('anharmonicity', -2 * math.pi * 0.25)
        amp_bound = hardware_model.get('amplitude_bound', 0.4)
        slew_bound = hardware_model.get('slew_bound', 0.12)
        detuning_bounds = hardware_model.get('detuning_bounds', [-2 * math.pi * 0.002, 2 * math.pi * 0.002])
        scale_bounds = hardware_model.get('scale_bounds', [0.97, 1.0, 1.03])
        T1 = hardware_model.get('T1', 25000.0)
        Tphi = hardware_model.get('Tphi', 40000.0)
    except:
        chi = -2 * math.pi * 0.25
        amp_bound = 0.4
        slew_bound = 0.12
        detuning_bounds = [-2 * math.pi * 0.002, 2 * math.pi * 0.002]
        scale_bounds = [0.97, 1.0, 1.03]
        T1 = 25000.0
        Tphi = 40000.0

    # Operators
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    
    # Precompute Lindblad superoperators
    L1 = a / math.sqrt(T1)
    L2 = n * math.sqrt(2.0 / Tphi)
    
    def make_lindblad_superop(L):
        LL_dag = L.conj().T @ L
        term1 = np.kron(L.conj().T.T, L)
        term2 = 0.5 * (np.kron(I3, LL_dag) + np.kron(LL_dag.T, I3))
        return term1 - term2
        
    M_L1 = make_lindblad_superop(L1)
    M_L2 = make_lindblad_superop(L2)
    M_Lindblad = M_L1 + M_L2
    
    # Precompute Hamiltonian components
    # H = delta * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I (a+a_dag) + i Q (a-a_dag))
    n_mat = n
    nn_minus_1 = n @ (n - I3)
    A_plus = a + a_dag
    A_minus = a - a_dag
    i_A_minus = 1j * A_minus
    
    # Initial states (6 states)
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (psi_0 + psi_1) / math.sqrt(2)
    psi_m = (psi_0 - psi_1) / math.sqrt(2)
    psi_r = (psi_0 + 1j*psi_1) / math.sqrt(2)
    psi_l = (psi_0 - 1j*psi_1) / math.sqrt(2)
    
    initial_states = [psi_0, psi_1, psi_p, psi_m, psi_r, psi_l]
    
    # Target states (X gate)
    X_mat = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
    target_states = [X_mat @ psi for psi in initial_states]
    
    # Vectorize initial density matrices: shape (9, 6)
    rho_vecs = np.zeros((9, 6), dtype=complex)
    for i, psi in enumerate(initial_states):
        rho = np.outer(psi, psi.conj())
        rho_vecs[:, i] = rho.flatten()

    # Precompute basis for pulse representation
    # Use sine series to ensure zero at boundaries
    K = 6 # Number of harmonics
    t_samples = np.arange(n_steps) * dt
    # t_samples are at 0.5, 1.0, ..., 39.5? Or 0, 0.5...?
    # Task says "80 constant steps of dt = 0.5 ns".
    # Usually step k corresponds to time k*dt.
    # If we have 80 steps, times are 0, 0.5, ..., 39.5.
    # Let's assume the pulse value is constant over [k*dt, (k+1)*dt).
    # The "slew" constraint is on the change between steps.
    # p[0] is the amplitude for step 0.
    # Constraint: ||p[0] - 0|| / dt <= slew.
    
    # Basis functions: sin(k * pi * t / T)
    # We need to evaluate this at the 80 time points.
    # Which time points? The center of the step? Or the start?
    # "80 constant steps". Usually means the value is applied for that step.
    # Let's use the midpoint of each step for the basis evaluation to be smooth.
    # t_mid = (k + 0.5) * dt
    t_mid = (np.arange(n_steps) + 0.5) * dt
    
    B = np.zeros((n_steps, K), dtype=float)
    for k in range(K):
        # Frequency: (k+1) * pi / T
        freq = (k + 1) * math.pi / T
        B[:, k] = np.sin(freq * t_mid)
    
    def get_pulse_from_coeffs(coeffs):
        # coeffs: [a1..aK, b1..bK]
        a_coeffs = coeffs[:K]
        b_coeffs = coeffs[K:]
        I = B @ a_coeffs
        Q = B @ b_coeffs
        return np.column_stack((I, Q))
    
    def calculate_score(I, Q):
        # Check constraints
        mags = np.sqrt(I**2 + Q**2)
        if np.max(mags) > amp_bound + 1e-9:
            return 10.0 # High penalty
        
        p = np.column_stack((I, Q))
        # Slew check
        # 0 -> p[0]
        if np.linalg.norm(p[0]) / dt > slew_bound + 1e-9:
            return 10.0
        # p[-1] -> 0
        if np.linalg.norm(p[-1]) / dt > slew_bound + 1e-9:
            return 10.0
        # p[i+1] - p[i]
        diffs = np.diff(p, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        if np.max(norms) / dt > slew_bound + 1e-9:
            return 10.0
            
        # Simulation
        # 9 cases
        detunings = detuning_bounds
        scales = scale_bounds
        
        infidelities = []
        
        for delta in detunings:
            for scale in scales:
                H0 = delta * n_mat + 0.5 * chi * nn_minus_1
                
                R = rho_vecs.copy() # 9x6
                
                for k in range(n_steps):
                    Ik = I[k]
                    Qk = Q[k]
                    HD = 0.5 * scale * (Ik * A_plus + Qk * i_A_minus)
                    H = H0 + HD
                    
                    M_coh = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
                    M_liouv = M_coh + M_Lindblad
                    
                    E = sla.expm(dt * M_liouv)
                    R = E @ R
                
                # Calculate infidelity for each state
                case_infid = 0.0
                for i in range(6):
                    rho_final_vec = R[:, i]
                    rho_final = rho_final_vec.reshape(3, 3)
                    target = target_states[i]
                    # Fidelity F = <target| rho |target>
                    F = np.real(np.vdot(target, rho_final @ target))
                    # Clamp fidelity to [0,1] to avoid numerical noise causing infid > 1
                    if F < 0: F = 0
                    if F > 1: F = 1
                    infid = 1.0 - F
                    case_infid += infid
                
                case_infid /= 6.0
                infidelities.append(case_infid)
        
        score = np.mean(infidelities) + np.max(infidelities)
        return score

    def objective(coeffs):
        pulse_arr = get_pulse_from_coeffs(coeffs)
        I = pulse_arr[:, 0]
        Q = pulse_arr[:, 1]
        return calculate_score(I, Q)

    # Initial guess: Project Gaussian seed onto basis
    # Seed: Gaussian sigma = 40/6, area pi
    sigma = T / 6.0
    t_seed = t_mid
    I_seed = (math.sqrt(math.pi) / sigma) * np.exp(-t_seed**2 / sigma**2)
    # DRAG Q component: -alpha * dI/dt
    alpha_seed = 0.2
    dIdt_seed = -2 * (math.sqrt(math.pi) / sigma) * t_seed / sigma**2 * np.exp(-t_seed**2 / sigma**2)
    Q_seed = -alpha_seed * dIdt_seed
    
    # Least squares fit to sine basis
    # I = B @ a  => a = pinv(B) @ I
    B_pinv = np.linalg.pinv(B)
    a_init = B_pinv @ I_seed
    b_init = B_pinv @ Q_seed
    x0 = np.concatenate((a_init, b_init))
    
    # Optimization
    start_time = time.time()
    deadline = 300.0 # Safety margin
    
    best_score = float('inf')
    best_coeffs = x0
    
    # Try a few different optimizers or restarts if time permits
    # L-BFGS-B is fast but doesn't handle constraints natively.
    # We use penalties in objective.
    
    try:
        res = opt.minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
            
        elapsed = time.time() - start_time
        if elapsed < deadline - 10: # Keep 10s margin
            # Try SLSQP with constraints if time allows? 
            # Or just a local Nelder-Mead refinement
            res2 = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x
                
    except Exception as e:
        pass # Keep best found so far

    # Generate final pulse
    pulse_arr = get_pulse_from_coeffs(best_coeffs)
    I = pulse_arr[:, 0]
    Q = pulse_arr[:, 1]
    
    # Final strict constraint check and correction if needed (scaling)
    mags = np.sqrt(I**2 + Q**2)
    if np.max(mags) > amp_bound:
        scale_factor = amp_bound / np.max(mags)
        I *= scale_factor
        Q *= scale_factor
        pulse_arr = np.column_stack((I, Q))
        
    # Check slew
    p = pulse_arr
    max_slew = 0
    if np.linalg.norm(p[0]) / dt > max_slew: max_slew = np.linalg.norm(p[0]) / dt
    if np.linalg.norm(p[-1]) / dt > max_slew: max_slew = np.linalg.norm(p[-1]) / dt
    if len(p) > 1:
        diffs = np.diff(p, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        if np.max(norms) / dt > max_slew: max_slew = np.max(norms) / dt
        
    if max_slew > slew_bound:
        # Scale down to meet slew
        scale_factor = slew_bound / max_slew
        I *= scale_factor
        Q *= scale_factor
        pulse_arr = np.column_stack((I, Q))

    return pulse_arr

# Do not call pulse() here, the evaluator will call it.
# pulse("X_pi", {})
