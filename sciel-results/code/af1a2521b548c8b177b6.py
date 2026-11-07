# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNINGS = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT

# Precompute matrices
# Basis: |0>, |1>, |2>
a_mat = np.array([
    [0, 1, 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
a_dag = a_mat.conj().T
n_mat = a_dag @ a_mat

# Lindblad operators
L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2 / TPHI)

# Precompute constant parts of Liouvillian
# L_const = sum_k (L_k^* \otimes L_k) - 0.5 * sum_k (L_k^dag L_k \otimes I + I \otimes (L_k^dag L_k)^T)
I3 = np.eye(3, dtype=complex)
L1_dag_L1 = L1.conj().T @ L1
L2_dag_L2 = L2.conj().T @ L2

L_const = np.kron(L1.conj(), L1) - 0.5 * (np.kron(L1_dag_L1, I3) + np.kron(I3, L1_dag_L1.T))
L_const += np.kron(L2.conj(), L2) - 0.5 * (np.kron(L2_dag_L2, I3) + np.kron(I3, L2_dag_L2.T))

# Precompute Hamiltonian components
a_plus_adag = a_mat + a_dag
a_minus_adag = a_mat - a_dag
n_nm1 = np.diag([0, 0, 2])
anharmonic_term = 0.5 * ANHARMONICITY * n_nm1

# Test states and targets
# Inputs: |0>, |1>, |+>, |->, |+i>, |-i>
states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, -1, 0],
    [1, 1j, 0],
    [1, -1j, 0]
], dtype=complex)
norms = np.linalg.norm(states, axis=1)
states = states / norms[:, np.newaxis]

# Targets: X gate applied
# X|0>=|1>, X|1>=|0>, X|+>=|+>, X|->=|->, X|+i>=|-i> (up to phase), X|-i>=|+i> (up to phase)
targets = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [1, -1j, 0],
    [1, 1j, 0]
], dtype=complex)
norms = np.linalg.norm(targets, axis=1)
targets = targets / norms[:, np.newaxis]

# Precompute initial density matrices and target projectors
rho_initial = [np.outer(s, s.conj()).flatten() for s in states]
target_projectors = [np.outer(t, t.conj()) for t in targets]

def _generate_pulse(params):
    """
    params: [beta, area_scale, sigma]
    Returns (80, 2) array of [I, Q]
    """
    beta, area_scale, sigma = params
    
    # Clip parameters to reasonable ranges to avoid numerical issues
    sigma = np.clip(sigma, 1.0, 20.0)
    area_scale = np.clip(area_scale, 0.5, 1.5)
    beta = np.clip(beta, -1.0, 1.0)
    
    t = np.linspace(0, T_TOTAL, N_STEPS, endpoint=False) + DT/2
    
    # Gaussian envelope
    # Area = A * sqrt(2*pi) * sigma = pi * area_scale
    # A = sqrt(pi/2) * area_scale / sigma
    A = np.sqrt(np.pi / 2) * area_scale / sigma
    
    I = A * np.exp(-t**2 / (2 * sigma**2))
    
    # Derivative of I
    dI_dt = -t / (sigma**2) * I
    
    # DRAG Q component
    Q = beta * dI_dt
    
    pulse = np.column_stack((I, Q))
    
    # Check validity and scale down if necessary
    amplitudes = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amplitudes)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
        
    # Check slew rate
    diffs = np.diff(pulse, axis=0)
    diffs = np.vstack([pulse[0], diffs, -pulse[-1]])
    norms = np.linalg.norm(diffs, axis=1)
    max_slew = np.max(norms) / DT
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        pulse *= scale_factor
        
    return pulse

def _simulate_pulse(pulse, detuning, scale):
    """
    Simulate the pulse and return mean fidelity for the 6 test states.
    """
    # We simulate all 6 states in parallel using a (6, 9) array of rho vectors
    rho_all = np.array(rho_initial, dtype=complex)
    
    for step in range(N_STEPS):
        I_val = pulse[step, 0]
        Q_val = pulse[step, 1]
        
        # Construct H
        drive_term = 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        H = detuning * n_mat + anharmonic_term + drive_term
        
        # Construct Hamiltonian part of Liouvillian
        # L_H = -i (I \otimes H^T) + i (H^T \otimes I)
        L_H = -1j * np.kron(I3, H.T) + 1j * np.kron(H.T, I3)
        
        # Total Liouvillian
        L_step = L_H + L_const
        
        # Exponentiate
        exp_L = expm(L_step * DT)
        
        # Apply to all 6 states
        # rho_all shape (6, 9), exp_L shape (9, 9)
        # We want rho_all_new[i] = exp_L @ rho_all[i]
        # This is (exp_L @ rho_all.T).T
        rho_all = (exp_L @ rho_all.T).T
        
    # Calculate fidelities
    fidelities = []
    for i in range(6):
        rho = rho_all[i].reshape(3, 3)
        # Fidelity with pure target state |psi_t>
        # F = <psi_t| rho |psi_t>
        psi_t = targets[i]
        F = np.real(np.vdot(psi_t, rho @ psi_t))
        # Clip to [0, 1]
        F = np.clip(F, 0.0, 1.0)
        fidelities.append(F)
        
    return np.mean(fidelities)

def _compute_score(params):
    """
    Compute the score for a given set of pulse parameters.
    Score = mean(infidelities) + max(infidelities)
    """
    pulse = _generate_pulse(params)
    
    infidelities = []
    for det in DE_TUNINGS:
        for scale in SCALES:
            fid = _simulate_pulse(pulse, det, scale)
            infidelities.append(1 - fid)
            
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def pulse(target_gate, hardware_model):
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 300.0 # 300s limit, keep margin
    
    # Initial guess
    # beta: DRAG coefficient. Typical value is around chi / (2 * Omega_0).
    # Omega_0 ~ A/sigma ~ (sqrt(pi/2)/sigma) * sigma ~ 1.25.
    # chi ~ -1.57.
    # beta ~ -1.57 / (2 * 1.25) ~ -0.6.
    # Let's start with 0.1 and let the optimizer find the best.
    x0 = np.array([0.1, 1.0, 40.0/6.0])
    
    best_params = x0
    best_score = _compute_score(x0)
    
    # Optimize
    if time.time() < deadline - 30: # Leave 30s margin
        try:
            # Use Nelder-Mead for local optimization
            # maxiter and xatol/fatol tuned for speed
            res = minimize(_compute_score, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500, 'adaptive': True})
            
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception as e:
            pass
            
    # Generate final pulse
    final_pulse = _generate_pulse(best_params)
    
    # Final validity check (redundant but safe)
    amplitudes = np.sqrt(final_pulse[:, 0]**2 + final_pulse[:, 1]**2)
    if np.any(amplitudes > AMPLITUDE_BOUND):
        scale_factor = AMPLITUDE_BOUND / np.max(amplitudes)
        final_pulse *= scale_factor
        
    diffs = np.diff(final_pulse, axis=0)
    diffs = np.vstack([final_pulse[0], diffs, -final_pulse[-1]])
    norms = np.linalg.norm(diffs, axis=1)
    max_slew = np.max(norms) / DT
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        final_pulse *= scale_factor
        
    return final_pulse

# EVOLVE-BLOCK-END
