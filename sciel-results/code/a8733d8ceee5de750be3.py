import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

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
    
    # Basis states and operators
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    eye3 = np.eye(3)
    
    # Lindbladian
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    def lindblad_term(L_op):
        L_op_dag = L_op.conj().T
        L_op_dag_L_op = L_op_dag @ L_op
        term1 = np.kron(L_op.conj(), L_op)
        term2 = -0.5 * np.kron(eye3, L_op_dag_L_op)
        term3 = -0.5 * np.kron(L_op_dag_L_op, eye3)
        return term1 + term2 + term3
    
    L_diss = lindblad_term(L1) + lindblad_term(L2)
    
    # Pre-compute Hamiltonian Kronecker terms for fast Liouvillian construction
    # H = det * n + 0.5 * chi * (n^2 - n) + 0.5 * scale * ( I (a+a_dag) + i Q (a-a_dag) )
    H_n = n
    H_anh = 0.5 * chi * (n @ n - n)
    H_I = 0.5 * (a + a_dag)
    H_Q = 0.5 * (1j * (a - a_dag))
    
    # L_ham = -i ( I \otimes H - H^T \otimes I )
    Kron_H_n = np.kron(eye3, H_n) - np.kron(H_n.T, eye3)
    Kron_H_anh = np.kron(eye3, H_anh) - np.kron(H_anh.T, eye3)
    Kron_H_I = np.kron(eye3, H_I) - np.kron(H_I.T, eye3)
    Kron_H_Q = np.kron(eye3, H_Q) - np.kron(H_Q.T, eye3)
    
    # States
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]
    
    # Initial density matrices (vectorized)
    rho_vecs = np.array([np.outer(psi, psi.conj()).flatten() for psi in states])
    # rho_vecs shape: (6, 9)
    
    targets = []
    X_mat = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)
    for psi in states:
        targets.append(X_mat @ psi)
    
    dets = [-det_bound, 0.0, det_bound]
    scales = [0.97, 1.0, 1.03]
    
    t_grid = (np.arange(N_steps) + 0.5) * dt
    
    def evaluate_pulse(I, Q):
        # Check validity
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound + 1e-9:
            return 10.0 # Penalty
        
        dI = np.diff(I, prepend=0.0)
        dQ = np.diff(Q, prepend=0.0)
        dI = np.append(dI, -I[-1])
        dQ = np.append(dQ, -Q[-1])
        slew = np.sqrt(dI**2 + dQ**2) / dt
        if np.max(slew) > slew_bound + 1e-9:
            return 10.0 # Penalty
            
        infidelities = []
        
        for det in dets:
            for scale in scales:
                # Initialize states
                rho_cur = rho_vecs.copy()
                
                for i in range(N_steps):
                    I_val = I[i]
                    Q_val = Q[i]
                    
                    # Construct Liouvillian
                    # L_ham = -1j * ( scale * (I_val * Kron_H_I + Q_val * Kron_H_Q) + det * Kron_H_n + Kron_H_anh )
                    L_ham = -1j * ( scale * (I_val * Kron_H_I + Q_val * Kron_H_Q) + det * Kron_H_n + Kron_H_anh )
                    L_total = L_ham + L_diss
                    
                    U = expm(L_total * dt)
                    rho_cur = (rho_cur @ U.T).T # rho_vec is column vector? 
                    # Wait, vec(rho) is a column vector. U @ vec(rho).
                    # rho_cur is (6, 9). We want U @ rho_cur[j] for each j.
                    # So rho_cur = rho_cur @ U.T if rho_cur is row-major?
                    # Let's check: rho_cur[j] is a row vector of length 9.
                    # We want new_rho[j] = U @ old_rho[j].
                    # If we store rho_cur as (6, 9), then rho_cur @ U.T gives (6, 9) where each row is U @ old_row?
                    # (U @ v) = (v^T U^T)^T. So if v is row, v^T is col. U @ v^T is col.
                    # So yes, rho_cur = rho_cur @ U.T works if rho_cur rows are vec(rho)^T.
                    # But vec(rho) is a column. So we should store it as (9, 6) or just loop?
                    # Let's stick to (6, 9) and use @ U.T.
                    
                # Compute infidelities
                case_infids = []
                for j in range(6):
                    rho_final = rho_cur[j].reshape((3, 3))
                    t_vec = targets[j]
                    F = np.real(np.vdot(t_vec, rho_final @ t_vec))
                    F = np.clip(F, 0.0, 1.0)
                    case_infids.append(1.0 - F)
                
                mean_infid = np.mean(case_infids)
                infidelities.append(mean_infid)
        
        mean_score = np.mean(infidelities)
        max_score = np.max(infidelities)
        return mean_score + max_score

    def objective(params):
        # params: [I0, I1, I2, I3, I4, Q0, Q1, Q2, Q3, Q4]
        I_ctrl = params[:5]
        Q_ctrl = params[5:10]
        
        # Interpolate
        t_ctrl = np.array([0, 10, 20, 30, 40])
        try:
            spline_I = CubicSpline(t_ctrl, I_ctrl)
            spline_Q = CubicSpline(t_ctrl, Q_ctrl)
            I = spline_I(t_grid)
            Q = spline_Q(t_grid)
        except Exception:
            return 10.0
            
        return evaluate_pulse(I, Q)

    # Initial guess: Gaussian-like
    t0 = 20.0
    sigma = 6.0
    I_guess = 0.3 * np.exp(-(t_ctrl - t0)**2 / (2*sigma**2))
    Q_guess = np.zeros(5)
    # Scale to ensure area is roughly pi? 
    # Area approx sum(I) * dt. 
    # We will let the optimizer find the amplitude.
    # Start with a valid pulse.
    initial_params = np.concatenate([I_guess, Q_guess])
    
    # Run Nelder-Mead
    result = minimize(objective, initial_params, method='Nelder-Mead', 
                      options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
    
    best_params = result.x
    I_ctrl = best_params[:5]
    Q_ctrl = best_params[5:10]
    
    spline_I = CubicSpline(t_ctrl, I_ctrl)
    spline_Q = CubicSpline(t_ctrl, Q_ctrl)
    I = spline_I(t_grid)
    Q = spline_Q(t_grid)
    
    # Final validity check and correction
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > amp_bound:
        scale_f = amp_bound / np.max(amp)
        I *= scale_f
        Q *= scale_f
        
    dI = np.diff(I, prepend=0.0)
    dQ = np.diff(Q, prepend=0.0)
    dI = np.append(dI, -I[-1])
    dQ = np.append(dQ, -Q[-1])
    slew = np.sqrt(dI**2 + dQ**2) / dt
    if np.max(slew) > slew_bound:
        # If slew is violated, reduce amplitude further or smooth
        # Simple reduction
        scale_f = slew_bound * dt / np.max(np.sqrt(dI**2 + dQ**2))
        I *= scale_f
        Q *= scale_f
        
    return np.column_stack((I, Q))

pulse = pulse
