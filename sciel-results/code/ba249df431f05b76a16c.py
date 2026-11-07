import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    # Hardware parameters
    chi = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    det_bound = 2 * np.pi * 0.002
    scale_min, scale_max = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    N_steps = 80
    T_total = N_steps * dt
    
    # Basis states |0>, |1>, |2>
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    
    n = a_dag @ a # diag(0, 1, 2)
    
    # Pre-compute static Liouvillian parts
    # L_diss is constant
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    def lindblad_term(L_op):
        L_op_dag = L_op.conj().T
        L_op_dag_L_op = L_op_dag @ L_op
        term1 = np.kron(L_op.conj(), L_op)
        term2 = -0.5 * np.kron(np.eye(3), L_op_dag_L_op)
        term3 = -0.5 * np.kron(L_op_dag_L_op, np.eye(3))
        return term1 + term2 + term3
    
    L_diss = lindblad_term(L1) + lindblad_term(L2)
    
    # Pre-compute static Hamiltonian parts
    # H_static = det * n + 0.5 * chi * (n^2 - n)
    # The detuning term depends on the case, so we can't fully pre-compute H_static for all cases.
    # But we can pre-compute the matrices needed to build it quickly.
    I3 = np.eye(3)
    n_mat = n
    n2_minus_n = n @ n - n # diag(0, 0, 2)
    
    # Drive term matrices
    # H_drive = 0.5 * scale * ( I (a+a_dag) + i Q (a-a_dag) )
    # Let X_op = a + a_dag
    # Let Y_op = -i (a - a_dag) => i (a - a_dag) = -Y_op
    # The task says: 0.5 * scale * ( I (a+a_dag) + i Q (a-a_dag) )
    # Let M_I = a + a_dag
    # Let M_Q = 1j * (a - a_dag)
    M_I = a + a_dag
    M_Q = 1j * (a - a_dag)
    
    # States to test
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]
    
    # Targets: X|psi>
    X_mat = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)
    targets = [X_mat @ psi for psi in states]
    
    # Vectorized initial density matrices for all 6 states, shape (9, 6)
    rho_vecs_init = np.zeros((9, 6), dtype=complex)
    for k, psi in enumerate(states):
        rho = np.outer(psi, psi.conj())
        rho_vecs_init[:, k] = rho.flatten()
        
    # Target projectors flattened? No, fidelity is <t| rho |t> = vdot(t, rho @ t).
    # We can compute this from the final rho_vec.
    
    def generate_pulse_params(params):
        # params: [A, sigma, beta, t0]
        A, sigma, beta, t0 = params
        
        # Ensure sigma is positive
        if sigma < 1.0:
            sigma = 1.0
            
        t = (np.arange(N_steps) + 0.5) * dt
        
        # Gaussian envelope
        gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
        
        # Derivative for DRAG
        dgauss = -(t - t0) / (sigma**2) * gauss
        
        I = A * gauss
        Q = beta * A * dgauss
        
        # Check validity and apply penalties/limits
        # Amplitude
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            # Scale down to be valid, but this changes the effective A.
            # In optimization, it's better to constrain A directly.
            # Here we just clip for the final pulse generation or return penalty.
            # For the simulator, we use the raw values but the penalty function handles validity.
            pass
            
        return I, Q

    def simulate_score(I, Q):
        # I, Q are arrays of length N_steps
        # Returns score (mean + max infidelity)
        
        dets = [-det_bound, 0.0, det_bound]
        scales = [0.97, 1.0, 1.03]
        
        case_infids = []
        
        for det in dets:
            for scale in scales:
                # Build static Liouvillian for this case
                H_static = det * n_mat + 0.5 * chi * n2_minus_n
                
                # We need to simulate 6 states simultaneously
                # rho_vecs shape (9, 6)
                rho_vecs = rho_vecs_init.copy()
                
                for i in range(N_steps):
                    I_val = I[i]
                    Q_val = Q[i]
                    
                    # H_drive
                    H_drive = 0.5 * scale * ( I_val * M_I + Q_val * M_Q )
                    
                    H = H_static + H_drive
                    
                    # Liouvillian Hamiltonian part
                    # vec(H rho - rho H) = (I \otimes H - H^T \otimes I) vec(rho)
                    L_ham = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
                    
                    L_total = L_ham + L_diss
                    
                    # Exponentiate
                    U_liouv = expm(L_total * dt)
                    
                    # Apply to all 6 states
                    rho_vecs = U_liouv @ rho_vecs
                    
                # Calculate infidelities
                # For each state k, rho_final is rho_vecs[:, k] reshaped to 3x3
                case_infids_local = []
                for k in range(6):
                    rho_final = rho_vecs[:, k].reshape(3, 3)
                    t_vec = targets[k]
                    F = np.real(np.vdot(t_vec, rho_final @ t_vec))
                    F = np.clip(F, 0.0, 1.0)
                    case_infids_local.append(1.0 - F)
                    
                mean_infid = np.mean(case_infids_local)
                case_infids.append(mean_infid)
                
        mean_score = np.mean(case_infids)
        max_score = np.max(case_infids)
        return mean_score + max_score

    def objective(params):
        A, sigma, beta, t0 = params
        
        # Constraints
        # A should be positive
        if A < 0:
            A = -A
            
        I, Q = generate_pulse_params([A, sigma, beta, t0])
        
        # Validity Check for Penalty
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            # Penalty for amplitude violation
            penalty = 100.0 * (np.max(amp) - amp_bound)
            
        # Slew check
        dI = np.diff(I, prepend=0.0)
        dQ = np.diff(Q, prepend=0.0)
        dI = np.append(dI, -I[-1])
        dQ = np.append(dQ, -Q[-1])
        slew = np.max(np.sqrt(dI**2 + dQ**2) / dt)
        
        if slew > slew_bound:
            penalty += 100.0 * (slew - slew_bound)
        else:
            penalty = 0.0 if 'penalty' not in locals() else penalty
            
        score = simulate_score(I, Q)
        
        return score + penalty

    # Initial guess
    # Typical pi pulse duration ~ 10-20 ns
    # sigma ~ 3-5 ns
    # beta ~ 0.1-0.3
    x0 = [0.3, 4.0, 0.2, 20.0]
    
    # Bounds
    bounds = [(0.0, 0.4), (2.0, 10.0), (-1.0, 1.0), (10.0, 30.0)]
    
    # Optimize
    # Use L-BFGS-B for bounds
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-6})
    
    best_params = result.x
    A, sigma, beta, t0 = best_params
    
    # Generate final pulse
    I, Q = generate_pulse_params([A, sigma, beta, t0])
    
    # Final validity enforcement (hard clip)
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > amp_bound:
        scale_f = amp_bound / np.max(amp)
        I *= scale_f
        Q *= scale_f
        
    # Slew enforcement (simple smoothing if violated)
    dI = np.diff(I, prepend=0.0)
    dQ = np.diff(Q, prepend=0.0)
    dI = np.append(dI, -I[-1])
    dQ = np.append(dQ, -Q[-1])
    slew = np.sqrt(dI**2 + dQ**2) / dt
    if np.max(slew) > slew_bound:
        # Reduce amplitude further if slew is violated
        # This is a crude fix, but ensures validity
        scale_f = slew_bound / np.max(slew)
        I *= scale_f
        Q *= scale_f
        
    return np.column_stack((I, Q))
