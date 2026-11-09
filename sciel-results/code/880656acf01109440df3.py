import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import expm

# Precompute basis matrices
# 3 levels: 0, 1, 2
# a = |1><0| + sqrt(2)|2><1|
a_mat = np.zeros((3, 3), dtype=complex)
a_mat[1, 0] = 1.0
a_mat[2, 1] = np.sqrt(2.0)
a_dag = a_mat.conj().T
n_mat = a_dag @ a_mat

# Liouvillian construction
# rho is 3x3, vectorized to 9x1 (row-major or column-major? let's use column-major (Fortran) for consistency with Kronecker products if needed, or just manual)
# Let's use manual vectorization: rho_vec = rho.flatten(order='F')
# L(rho) = -i[H, rho] + ...
# Vectorized: L_vec = -i (I \otimes H^T - H \otimes I) + ...
# This is efficient.

def vec(rho):
    return rho.flatten(order='F')

def unvec(vec_rho):
    return vec_rho.reshape((3, 3), order='F')

# Precompute Liouvillian components
# H = Delta * n + 0.5 * alpha * n(n-1) + 0.5 * s * (I*(a+a_dag) + i*Q*(a-a_dag))
# Note: The drive term depends on I, Q. The detuning/anharmonicity part is constant for a given case.
# L1 = a / sqrt(T1)
# L2 = n * sqrt(2/Tphi)

T1 = 25000.0
Tphi = 40000.0
alpha = -2 * np.pi * 0.25

L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2.0 / Tphi)

# Lindblad term vectorized:
# sum_k (L_k rho L_k^dagger - 0.5 {L_k^dagger L_k, rho})
# Vectorized form for one L:
# (L \otimes L^*) - 0.5 (I \otimes (L^dagger L)^T) - 0.5 ((L^dagger L) \otimes I)
# Wait, standard formula:
# vec(L rho L^dagger) = (L \otimes L^*) vec(rho)
# vec(L^dagger L rho) = (I \otimes (L^dagger L)^T) vec(rho)
# vec(rho L^dagger L) = ((L^dagger L) \otimes I) vec(rho)

def make_lindblad_vec(L):
    L_dag_L = L.conj().T @ L
    term1 = np.kron(L, L.conj())
    term2 = 0.5 * np.kron(np.eye(3), L_dag_L.T)
    term3 = 0.5 * np.kron(L_dag_L, np.eye(3))
    return term1 - term2 - term3

Lind_vec_const = make_lindblad_vec(L1) + make_lindblad_vec(L2)

# Hamiltonian term vectorized: -i [H, rho]
# vec(H rho - rho H) = (I \otimes H^T - H \otimes I) vec(rho)
# So -i [H, rho] -> -i (I \otimes H^T - H \otimes I)

def get_liouvillian_vec(I_val, Q_val, detuning, scale):
    # H_static = detuning * n + 0.5 * alpha * (n @ (n - np.eye(3)))
    n_minus_1 = n_mat - np.eye(3)
    H_static = detuning * n_mat + 0.5 * alpha * (n_mat @ n_minus_1)
    
    # H_drive = 0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
    H_drive = 0.5 * scale * (I_val * (a_mat + a_dag) + 1j * Q_val * (a_mat - a_dag))
    
    H = H_static + H_drive
    
    H_term = -1j * (np.kron(np.eye(3), H.T) - np.kron(H, np.eye(3)))
    
    return H_term + Lind_vec_const

def simulate_pulse(I, Q, detuning, scale):
    dt = 0.5
    N = len(I)
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    rho_vec = vec(rho)
    
    for k in range(N):
        L_vec = get_liouvillian_vec(I[k], Q[k], detuning, scale)
        # Evolve for dt
        # rho_vec_next = exp(L_vec * dt) @ rho_vec
        # Use expm
        rho_vec = expm(L_vec * dt) @ rho_vec
        # Optional: normalize trace? Lindblad preserves trace, but numerical errors might creep in.
        # trace = np.trace(unvec(rho_vec))
        # rho_vec /= trace
        
    return unvec(rho_vec)

def calculate_infidelity(I, Q):
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Initial states
    # |0>, |1>, (|0>+|1>)/sqrt2, (|0>-|1>)/sqrt2, (|0>+i|1>)/sqrt2, (|0>-i|1>)/sqrt2
    # Target: X gate
    # X|0> = |1>, X|1> = |0>
    # X|+> = |->, X|-> = |+>
    # X|+i> = |-i>, X|-i> = |+i>
    
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    # Target states after X
    target_states = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    infidelities = []
    
    for d in detunings:
        for s in scales:
            case_infs = []
            for psi_init, psi_target in zip(states, target_states):
                # Start with density matrix of psi_init
                rho_init = np.outer(psi_init, psi_init.conj().T)
                
                # We need to simulate starting from rho_init, not |0><0|
                # My simulate_pulse above assumes |0><0|. Let's generalize.
                
                # Re-implement simulate for arbitrary initial rho
                dt = 0.5
                N = len(I)
                rho = rho_init.copy()
                rho_vec = vec(rho)
                
                for k in range(N):
                    L_vec = get_liouvillian_vec(I[k], Q[k], d, s)
                    rho_vec = expm(L_vec * dt) @ rho_vec
                
                rho_final = unvec(rho_vec)
                
                # Fidelity: Tr(rho_target rho_final)
                rho_target = np.outer(psi_target, psi_target.conj().T)
                fid = np.real(np.trace(rho_target @ rho_final))
                # Clamp to [0,1]
                if fid < 0: fid = 0
                if fid > 1: fid = 1
                
                inf = 1.0 - fid
                case_infs.append(inf)
            
            mean_inf = np.mean(case_infs)
            infidelities.append(mean_inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    
    return mean_inf + max_inf

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Parameters to optimize
    # sigma (width), drag_coeff (lambda), amp (scale)
    # Initial guess: sigma=6.67, drag=0.1, amp=1.0
    
    def objective(params):
        sigma, drag, amp = params
        if sigma <= 0:
            return 1e6
        
        t = np.arange(80) * 0.5 + 0.25
        t_center = 20.0
        
        # Gaussian
        I_gauss = amp * np.sqrt(np.pi / 2.0) / sigma * np.exp(-(t - t_center)**2 / (2.0 * sigma**2))
        
        # Derivative for DRAG
        dI_dt = np.gradient(I_gauss, 0.5)
        Q = drag * dI_dt
        
        # Check bounds
        # Amplitude
        max_amp = np.max(np.sqrt(I_gauss**2 + Q**2))
        if max_amp > 0.4:
            scale_f = 0.4 / max_amp
            I_gauss *= scale_f
            Q *= scale_f
        
        # Slew
        I_ext = np.concatenate([[0.0], I_gauss, [0.0]])
        Q_ext = np.concatenate([[0.0], Q, [0.0]])
        dI = np.diff(I_ext) / 0.5
        dQ = np.diff(Q_ext) / 0.5
        slew = np.sqrt(dI**2 + dQ**2)
        max_slew = np.max(slew)
        if max_slew > 0.12:
            scale_f = 0.12 / max_slew
            I_gauss *= scale_f
            Q *= scale_f
            
        # Final check
        max_amp = np.max(np.sqrt(I_gauss**2 + Q**2))
        if max_amp > 0.4:
            scale_f = 0.4 / max_amp
            I_gauss *= scale_f
            Q *= scale_f
            
        return calculate_infidelity(I_gauss, Q)
    
    # Optimize
    # Bounds: sigma in [2, 15], drag in [-0.5, 0.5], amp in [0.1, 1.0]
    # Note: amp is scaled by sqrt(pi/2)/sigma, so effective amp is large.
    # Let's just optimize sigma and drag, and fix the area to pi initially, then let the optimizer adjust?
    # No, the area must result in a Pi pulse.
    # If we change sigma, the peak changes to maintain area.
    # So I(t) is fixed in area.
    # We just optimize sigma and drag.
    
    # Let's use Nelder-Mead
    x0 = [6.67, 0.1]
    bounds = [(2.0, 15.0), (-0.5, 0.5)]
    
    # Wrap objective to handle bounds if needed, but NM doesn't support bounds directly.
    # We can penalize out-of-bounds.
    
    def obj_nm(params):
        sigma, drag = params
        if sigma < 2.0 or sigma > 15.0 or drag < -0.5 or drag > 0.5:
            return 1e6
        return objective([sigma, drag, 1.0]) # amp fixed to 1.0 (area pi)
        
    # Initial guess
    # Seed: sigma=40/6=6.67, drag=0.1
    res = minimize(obj_nm, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    best_sigma, best_drag = res.x
    
    # Generate final pulse
    t = np.arange(80) * 0.5 + 0.25
    t_center = 20.0
    
    I_gauss = np.sqrt(np.pi / 2.0) / best_sigma * np.exp(-(t - t_center)**2 / (2.0 * best_sigma**2))
    dI_dt = np.gradient(I_gauss, 0.5)
    Q = best_drag * dI_dt
    
    # Apply bounds
    max_amp = np.max(np.sqrt(I_gauss**2 + Q**2))
    if max_amp > 0.4:
        scale_f = 0.4 / max_amp
        I_gauss *= scale_f
        Q *= scale_f
        
    I_ext = np.concatenate([[0.0], I_gauss, [0.0]])
    Q_ext = np.concatenate([[0.0], Q, [0.0]])
    dI = np.diff(I_ext) / 0.5
    dQ = np.diff(Q_ext) / 0.5
    slew = np.sqrt(dI**2 + dQ**2)
    max_slew = np.max(slew)
    if max_slew > 0.12:
        scale_f = 0.12 / max_slew
        I_gauss *= scale_f
        Q *= scale_f
        
    return np.column_stack((I_gauss, Q))
