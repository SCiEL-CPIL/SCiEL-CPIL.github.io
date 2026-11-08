"""Rewritten program: DRAG pulse optimization for three-level transmon X-pi gate."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns

# Time grid
t_grid = np.arange(N_STEPS) * DT  # 0 to 39.5 ns

# Precompute basis operators
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
# a_dag = a.T
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.T  # conjugate transpose for 3x3 matrix
n_op = a_dag @ a_op

# Lindblad operators
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / Tphi)

def build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for a given step."""
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    n2 = n_op @ n_op
    H = (detuning * n_op + 0.5 * ANHARMONICITY * (n2 - n_op) + 
         0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag)))
    
    # Liouvillian: L(rho) = -i*[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # In vectorized form: L = -i*(H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 0.5*(L_k^dag L_k ⊗ I + I ⊗ (L_k^dag L_k)^T))
    dim = 3
    I_mat = np.eye(dim, dtype=complex)
    
    # Coherent part
    H_vec = H.reshape(dim*dim, 1)
    L_coh = -1j * (np.kron(H, I_mat) - np.kron(I_mat, H.T))
    
    # Dissipative part
    L_diss = np.zeros((dim*dim, dim*dim), dtype=complex)
    for Lk in [L1, L2]:
        Lk_dag = Lk.T
        Lk_dag_Lk = Lk_dag @ Lk
        L_diss += np.kron(Lk, Lk.T) - 0.5 * np.kron(Lk_dag_Lk, I_mat) - 0.5 * np.kron(I_mat, Lk_dag_Lk.T)
    
    return L_coh + L_diss

def simulate_pulse(pulse_I, pulse_Q, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    dim = 3
    rho = np.zeros((dim*dim, dim*dim), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0><0|
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        L = build_liouvillian(detuning, scale, I_val, Q_val)
        
        # Exponentiate the Liouvillian over dt
        # Use matrix exponential
        from scipy.linalg import expm
        L_dt = L * DT
        expL = expm(L_dt)
        rho = expL @ rho
    
    return rho

def compute_infidelity(pulse_I, pulse_Q):
    """Compute the mean infidelity over the 9 cases and 5 states."""
    detunings = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
    scales = np.array([0.97, 1.0, 1.03])
    
    # Initial states
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    # Target: X gate on |0> and |1>
    # X|0> = |1>, X|1> = |0>
    # For superpositions, X acts as: X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
    # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
    # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2)
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
        np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # X(|0>+|1>)/sqrt(2)
        -(np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # X(|0>-|1>)/sqrt(2)
        (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2),  # X(|0>+i|1>)/sqrt(2)
        (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),  # X(|0>-i|1>)/sqrt(2)
    ]
    
    total_fidelity = 0.0
    count = 0
    
    for det in detunings:
        for scale in scales:
            for state, target in zip(states, target_states):
                rho_init = np.outer(state, state.T)
                rho_vec = rho_init.reshape(9, 1)
                
                # Simulate
                rho_final = rho_vec
                for i in range(N_STEPS):
                    I_val = pulse_I[i]
                    Q_val = pulse_Q[i]
                    L = build_liouvillian(det, scale, I_val, Q_val)
                    from scipy.linalg import expm
                    expL = expm(L * DT)
                    rho_final = expL @ rho_final
                
                rho_final_mat = rho_final.reshape(3, 3)
                target_mat = np.outer(target, target.T)
                
                # Fidelity = Tr(rho_final * target_mat)
                fidelity = np.trace(rho_final_mat @ target_mat).real
                fidelity = max(0.0, min(1.0, fidelity))
                total_fidelity += fidelity
                count += 1
    
    mean_fidelity = total_fidelity / count
    infidelity = 1.0 - mean_fidelity
    return infidelity

def score_pulse(pulse_I, pulse_Q):
    """Compute the score: mean + max of infidelities over 9 cases."""
    detunings = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
    scales = np.array([0.97, 1.0, 1.03])
    
    # Initial states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        -(np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    case_infidelities = []
    
    for det in detunings:
        for scale in scales:
            total_fidelity = 0.0
            for state, target in zip(states, target_states):
                rho_init = np.outer(state, state.T)
                rho_vec = rho_init.reshape(9, 1)
                
                rho_final = rho_vec
                for i in range(N_STEPS):
                    I_val = pulse_I[i]
                    Q_val = pulse_Q[i]
                    L = build_liouvillian(det, scale, I_val, Q_val)
                    from scipy.linalg import expm
                    expL = expm(L * DT)
                    rho_final = expL @ rho_final
                
                rho_final_mat = rho_final.reshape(3, 3)
                target_mat = np.outer(target, target.T)
                fidelity = np.trace(rho_final_mat @ target_mat).real
                fidelity = max(0.0, min(1.0, fidelity))
                total_fidelity += fidelity
            
            mean_fidelity = total_fidelity / len(states)
            infidelity = 1.0 - mean_fidelity
            case_infidelities.append(infidelity)
    
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    return mean_inf + max_inf

def check_validity(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMPLITUDE_BOUND + 1e-9:
            return False
    
    # Slew: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero into step 1
    p0 = np.array([0.0, 0.0])
    p1 = np.array([pulse_I[0], pulse_Q[0]])
    if np.linalg.norm(p1 - p0) / DT > SLEW_BOUND + 1e-9:
        return False
    
    for i in range(N_STEPS - 1):
        pi = np.array([pulse_I[i], pulse_Q[i]])
        pi1 = np.array([pulse_I[i+1], pulse_Q[i+1]])
        if np.linalg.norm(pi1 - pi) / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    plast = np.array([pulse_I[-1], pulse_Q[-1]])
    if np.linalg.norm(plast) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True

def make_drag_pulse(alpha=0.0, beta=0.0):
    """Create a DRAG-style Gaussian pulse with area pi."""
    t = np.arange(N_STEPS) * DT
    center = DURATION / 2.0
    sigma = DURATION / 6.0
    
    # Gaussian envelope
    gauss = np.exp(-(t - center)**2 / (2 * sigma**2))
    
    # Offset to make it zero at edges
    edge_val = np.exp(-((0 - center)**2) / (2 * sigma**2))
    gauss = gauss - edge_val
    gauss = np.maximum(gauss, 0)
    
    # Scale to get area pi
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    I = gauss
    Q = beta * np.gradient(gauss)  # DRAG term
    
    return I, Q

def pulse(target_gate, hardware_model):
    """Main function: returns an (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    # Start with DRAG seed
    I, Q = make_drag_pulse(alpha=0.0, beta=0.0)
    
    # Parameterize for optimization
    # Use a simple parameterization: optimize I and Q directly with constraints
    # But for speed, use a coarser approach first
    
    start_time = time.time()
    deadline = 300.0  # seconds
    
    best_I = I.copy()
    best_Q = Q.copy()
    best_score = score_pulse(best_I, best_Q)
    
    # Simple local refinement using Nelder-Mead on a subset of parameters
    # To keep it fast, optimize only a few key parameters
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return best_score  # Return current best if close to deadline
        
        # params: [beta, alpha, width_shift]
        beta_val = params[0]
        alpha_val = params[1]
        width_shift = params[2]
        
        t = np.arange(N_STEPS) * DT
        center = DURATION / 2.0 + width_shift
        sigma = DURATION / 6.0
        
        gauss = np.exp(-(t - center)**2 / (2 * sigma**2))
        edge_val = np.exp(-((0 - center)**2) / (2 * sigma**2))
        gauss = gauss - edge_val
        gauss = np.maximum(gauss, 0)
        
        area = np.sum(gauss) * DT
        if area > 0:
            gauss = gauss * (np.pi / area)
        
        I_new = gauss
        Q_new = beta_val * np.gradient(gauss) + alpha_val * gauss
        
        # Check validity
        if not check_validity(I_new, Q_new):
            return 1e6
        
        sc = score_pulse(I_new, Q_new)
        return sc
    
    # Optimize
    x0 = np.array([0.0, 0.0, 0.0])
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if result.fun < best_score:
            beta_val = result.x[0]
            alpha_val = result.x[1]
            width_shift = result.x[2]
            
            t = np.arange(N_STEPS) * DT
            center = DURATION / 2.0 + width_shift
            sigma = DURATION / 6.0
            
            gauss = np.exp(-(t - center)**2 / (2 * sigma**2))
            edge_val = np.exp(-((0 - center)**2) / (2 * sigma**2))
            gauss = gauss - edge_val
            gauss = np.maximum(gauss, 0)
            
            area = np.sum(gauss) * DT
            if area > 0:
                gauss = gauss * (np.pi / area)
            
            best_I = gauss
            best_Q = beta_val * np.gradient(gauss) + alpha_val * gauss
            best_score = result.fun
    except Exception:
        pass
    
    # Final validity check
    if not check_validity(best_I, best_Q):
        # Fall back to seed
        best_I, best_Q = make_drag_pulse(alpha=0.0, beta=0.0)
    
    return np.column_stack([best_I, best_Q])
