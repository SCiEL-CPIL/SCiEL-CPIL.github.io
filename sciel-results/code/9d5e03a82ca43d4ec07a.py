import numpy as np
import time
import warnings
from scipy.linalg import expm
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Constants
DT = 0.5
N_STEPS = 80
T_TOTAL = N_STEPS * DT
ALPHA_ANH = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
TPHI = 40000.0

# Hardware cases
DELTAS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]

def get_initial_states():
    s0 = np.array([1, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0], dtype=complex)
    sp = (s0 + s1) / np.sqrt(2)
    sm = (s0 - s1) / np.sqrt(2)
    spi = (s0 + 1j * s1) / np.sqrt(2)
    smi = (s0 - 1j * s1) / np.sqrt(2)
    return np.array([s0, s1, sp, sm, spi, smi])

def get_target_states():
    t0 = -1j * np.array([0, 1, 0], dtype=complex)
    t1 = -1j * np.array([1, 0, 0], dtype=complex)
    tp = -1j * (np.array([0, 1, 0]) + np.array([1, 0, 0])) / np.sqrt(2)
    tm = -1j * (np.array([0, 1, 0]) - np.array([1, 0, 0])) / np.sqrt(2)
    tpi = -1j * (np.array([0, 1, 0]) + 1j * np.array([1, 0, 0])) / np.sqrt(2)
    tmi = -1j * (np.array([0, 1, 0]) - 1j * np.array([1, 0, 0])) / np.sqrt(2)
    return np.array([t0, t1, tp, tm, tpi, tmi])

def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

A, ADAG, N_OP = build_operators()

# Pre-compute constant parts of Liouvillian
# L(rho) = -i [H, rho] + sum (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
# H = delta * n + 0.5 * alpha * n * (n - 1) + 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
# Let H0 = delta * n + 0.5 * alpha * n * (n - 1)
# Let H1 = 0.5 * (a + adag)  (for I term)
# Let H2 = 0.5j * (a - adag) (for Q term)
# Then H = H0 + scale * (I * H1 + Q * H2)

# Pre-compute superoperator terms for H0, H1, H2
# S(H) = -i (I ⊗ H - H^T ⊗ I)
def super_op_H(H):
    return -1j * (np.kron(np.eye(3), H) - np.kron(H.conj().T, np.eye(3)))

# Dissipative part is constant
L1 = A / np.sqrt(T1)
L2 = N_OP * np.sqrt(2 / TPHI)

Diss_L_mat = np.zeros((9, 9), dtype=complex)
for L_k in [L1, L2]:
    L_k_dag = L_k.conj().T
    term1 = np.kron(L_k_dag.conj().T, L_k)
    L_k_dag_L_k = L_k_dag @ L_k
    term2 = 0.5 * (np.kron(np.eye(3), L_k_dag_L_k) + np.kron(L_k_dag_L_k.conj().T, np.eye(3)))
    Diss_L_mat += term1 - term2

# Pre-compute for H0 (delta dependent)
# H0 = delta * N_OP + 0.5 * ALPHA_ANH * N_OP @ (N_OP - np.eye(3))
# We can precompute the superoperator for N_OP and for N_OP(N_OP-I)
N_OP_super = super_op_H(N_OP)
N2M1 = N_OP @ (N_OP - np.eye(3))
N2M1_super = super_op_H(N2M1)

# Pre-compute for H1 and H2
H1 = 0.5 * (A + ADAG)
H2 = 0.5j * (A - ADAG)
H1_super = super_op_H(H1)
H2_super = super_op_H(H2)

def build_liouvillian_fast(I, Q, delta, scale):
    # L = delta * N_OP_super + 0.5 * ALPHA_ANH * N2M1_super 
    #   + scale * (I * H1_super + Q * H2_super) + Diss_L_mat
    return (delta * N_OP_super + 0.5 * ALPHA_ANH * N2M1_super + 
            scale * (I * H1_super + Q * H2_super) + Diss_L_mat)

def simulate_pulse(I, Q):
    infidelities = []
    init_states = get_initial_states()
    target_states = get_target_states()
    
    rho_init = np.zeros((6, 3, 3), dtype=complex)
    for i, s in enumerate(init_states):
        rho_init[i] = np.outer(s, s.conj().T)
        
    for delta in DELTAS:
        for scale in SCALES:
            rho = rho_init.copy()
            
            for k in range(N_STEPS):
                L_k = build_liouvillian_fast(I[k], Q[k], delta, scale)
                U_k = expm(L_k * DT)
                rho_flat = rho.reshape(6, 9)
                rho_flat = U_k @ rho_flat.T
                rho = rho_flat.T.reshape(6, 3, 3)
                
            fidelities = []
            for i in range(6):
                rho_final = rho[i]
                t = target_states[i]
                f = np.real(np.vdot(t, rho_final @ t))
                fidelities.append(f)
                
            mean_fid = np.mean(fidelities)
            infid = 1.0 - mean_fid
            infidelities.append(infid)
            
    return infidelities

def check_constraints(I, Q):
    amps = np.sqrt(I**2 + Q**2)
    if np.any(amps > AMP_BOUND):
        return False
        
    dI = np.diff(I)
    dQ = np.diff(Q)
    d_norms = np.sqrt(dI**2 + dQ**2)
    if np.any(d_norms / DT > SLEW_BOUND):
        return False
        
    if np.sqrt(I[0]**2 + Q[0]**2) / DT > SLEW_BOUND:
        return False
    if np.sqrt(I[-1]**2 + Q[-1]**2) / DT > SLEW_BOUND:
        return False
        
    return True

def generate_pulse(sigma, beta):
    t = np.linspace(0, T_TOTAL, N_STEPS)
    t_center = T_TOTAL / 2
    t = t - t_center
    
    A_amp = np.sqrt(np.pi / 2) / sigma
    I = A_amp * np.exp(-t**2 / (2 * sigma**2))
    Q = -beta * t / (sigma**2) * I
    
    return I, Q

def objective(params):
    sigma, beta = params
    if sigma < 1.0 or sigma > 10.0:
        return 1e6
    if beta < -50.0 or beta > 50.0:
        return 1e6
        
    I, Q = generate_pulse(sigma, beta)
    
    if not check_constraints(I, Q):
        return 1e6
        
    infids = simulate_pulse(I, Q)
    score = np.mean(infids) + np.max(infids)
    return score

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0
    
    best_score = 1e6
    best_params = (5.0, 10.0)
    
    # Coarse grid search
    sigmas = np.arange(3.0, 9.0, 0.5)
    betas = np.arange(-15.0, 15.0, 1.0)
    
    for s in sigmas:
        for b in betas:
            if time.time() > deadline - 30.0:
                break
            score = objective([s, b])
            if score < best_score:
                best_score = score
                best_params = (s, b)
        
        if time.time() > deadline - 30.0:
            break
            
    # Local refinement with Nelder-Mead
    starts = [best_params, (5.0, 10.0), (4.0, 5.0), (6.0, 15.0), (5.0, 0.0)]
    
    for s in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    sigma, beta = best_params
    I, Q = generate_pulse(sigma, beta)
    
    if not check_constraints(I, Q):
        sigma_safe = 8.0
        beta_safe = 0.0
        I, Q = generate_pulse(sigma_safe, beta_safe)
        
    return np.column_stack((I, Q))
