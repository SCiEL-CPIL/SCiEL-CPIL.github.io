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
# a = lowering operator for 3 levels (truncated)
# |0>, |1>, |2>
# a|0> = 0
# a|1> = 1|0>
# a|2> = sqrt(2)|1>
# Matrix a (basis |0>, |1>, |2>):
# [[0, 1, 0],
#  [0, 0, sqrt(2)],
#  [0, 0, 0]]
a_mat = np.array([
    [0, 1, 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
a_dag = a_mat.conj().T
n_mat = a_dag @ a_mat

# Lindblad operators
# L1 = a / sqrt(T1)
# L2 = n * sqrt(2 / Tphi)  (Note: task says "n sqrt(2 / Tphi)", usually dephasing is proportional to n or n^2? 
# "Lindblad operators a / sqrt(T1) and n sqrt(2 / Tphi)"
# Standard dephasing for transmon is often proportional to n or n^2 depending on the noise source. 
# The prompt explicitly says "n sqrt(2 / Tphi)".
# Let's stick to the prompt.
L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2 / TPHI)

# Liouvillian superoperator construction
# d rho/dt = -i [H, rho] + sum L rho L^dag - 1/2 {L^dag L, rho}
# We can represent this as a matrix acting on vec(rho).
# Or simply use the Lindblad master equation form in the simulation step.
# For speed, we can precompute the terms if H is constant, but H varies with I, Q.
# H = delta * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I (a + a^dag) + i Q (a - a^dag))
# n(n-1) = n^2 - n.
# n = diag(0, 1, 2).
# n^2 = diag(0, 1, 4).
# n(n-1) = diag(0, 0, 2).
# So the anharmonic term is 0.5 * chi * diag(0, 0, 2) = diag(0, 0, chi).

def _generate_pulse(params):
    """
    params: [sigma, alpha, area_scale]
    Returns (80, 2) array of [I, Q]
    """
    sigma, alpha, area_scale = params
    # Ensure sigma is reasonable
    sigma = np.clip(sigma, 1.0, 20.0)
    
    t = np.linspace(0, T_TOTAL, N_STEPS, endpoint=False) + DT/2 # Center of steps
    # Gaussian envelope
    # Area = A * sqrt(2*pi) * sigma
    # We want Area = pi * area_scale
    # A = (pi * area_scale) / (sqrt(2*pi) * sigma) = sqrt(pi/2) * area_scale / sigma
    A = np.sqrt(np.pi / 2) * area_scale / sigma
    
    I = A * np.exp(-t**2 / (2 * sigma**2))
    
    # Derivative for DRAG
    # I'(t) = -t/sigma^2 * I(t)
    dI_dt = -t / (sigma**2) * I
    
    # DRAG Q component
    # Typically Q(t) = alpha * dI/dt
    # The sign and scaling of alpha depend on the specific DRAG formulation.
    # Standard DRAG: Q(t) = (chi / (2 * Omega_0)) * I'(t) ? 
    # Let's use alpha as a free parameter to be optimized.
    # A common scaling is Q = alpha * dI/dt.
    Q = alpha * dI_dt
    
    pulse = np.column_stack((I, Q))
    
    # Check validity
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.any(amplitudes > AMPLITUDE_BOUND):
        # Scale down if necessary
        scale_factor = AMPLITUDE_BOUND / np.max(amplitudes)
        pulse *= scale_factor
        
    # Check slew rate
    # p[0] - 0, p[i] - p[i-1], p[-1] - 0
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
    Simulate the pulse and return fidelities for the 5 test states.
    Returns mean fidelity.
    """
    # Precompute H components that are constant across steps for this detuning/scale
    n_nm1 = np.diag([0, 0, 2])
    anharmonic_term = 0.5 * ANHARMONICITY * n_nm1
    a_plus_adag = a_mat + a_dag
    a_minus_adag = a_mat - a_dag
    
    # Precompute constant parts of Liouvillian
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    I3 = np.eye(3, dtype=complex)
    
    # Constant Lindblad part of Liouvillian
    L_const = np.kron(L1.conj(), L1) - 0.5 * (np.kron(L1_dag_L1, I3) + np.kron(I3, L1_dag_L1.T))
    L_const += np.kron(L2.conj(), L2) - 0.5 * (np.kron(L2_dag_L2, I3) + np.kron(I3, L2_dag_L2.T))
    
    # Precompute initial density matrices
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
    
    # Precompute initial rho vectors for all 6 states
    rho_vecs = []
    for i in range(6):
        rho = np.outer(states[i], states[i].conj())
        rho_vecs.append(rho.flatten())
    
    # Precompute target density matrices
    target_rhos = []
    for i in range(6):
        rho_t = np.outer(targets[i], targets[i].conj())
        target_rhos.append(rho_t)
    
    # Simulate all 6 states in parallel (vectorized over states)
    # rho_vecs is a list of 6 vectors of length 9
    # We'll keep them as a (6, 9) array
    rho_all = np.array(rho_vecs, dtype=complex) # shape (6, 9)
    
    for step in range(N_STEPS):
        I_val = pulse[step, 0]
        Q_val = pulse[step, 1]
        
        # Construct H
        drive_term = 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        H = detuning * n_mat + anharmonic_term + drive_term
        
        # Construct Hamiltonian part of Liouvillian
        # L_H = -i (I ⊗ H^T) + i (H^T ⊗ I)
        L_H = -1j * np.kron(I3, H.T) + 1j * np.kron(H.T, I3)
        
        # Total Liouvillian for this step
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
        rho_t = target_rhos[i]
        
        # Fidelity for pure target and mixed state:
        # F = <psi_t| rho |psi_t>
        # Since target is pure, this is simpler and exact.
        psi_t = targets[i]
        F = np.real(np.vdot(psi_t, rho @ psi_t))
        # Clip to [0, 1] to handle numerical noise
        F = np.clip(F, 0.0, 1.0)
        fidelities.append(F)
        
    return np.mean(fidelities)
    # Initial states
    # |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # Wait, the task says: "|0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # That's 1 + 1 + 2 + 2 = 6 states?
    # "mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # Usually X gate maps |0>->|1>, |1>->|0>.
    # Fidelity is usually |<target|psi>|^2.
    # Target for |0> is |1>. Target for |1> is |0>.
    # Target for (|0>+|1>)/sqrt(2) is (|1>+|0>)/sqrt(2) = same state.
    # Target for (|0>-|1>)/sqrt(2) is (|1>-|0>)/sqrt(2) = - (|0>-|1>)/sqrt(2). Fidelity 1.
    # Target for (|0>+i|1>)/sqrt(2) is (|1>+i|0>)/sqrt(2) = i (|0>-i|1>)/sqrt(2)?
    # Let's just compute the unitary U and check fidelity to the ideal X gate.
    # Ideal X: X|0>=|1>, X|1>=|0>, X|2>=|2> (or whatever, but we only care about 0,1 subspace ideally).
    # Actually, we just need the fidelity of the evolution of the 5 (or 6) input states to their X-rotated counterparts.
    
    # Let's construct the 5 input states as density matrices or state vectors.
    # State vectors are easier.
    # |0> = [1, 0, 0]
    # |1> = [0, 1, 0]
    # |+> = [1, 1, 0] / sqrt(2)
    # |-> = [1, -1, 0] / sqrt(2)
    # |+i> = [1, i, 0] / sqrt(2)
    # |-i> = [1, -i, 0] / sqrt(2)
    
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
    
    # Targets (X gate applied)
    # X|0> = |1>
    # X|1> = |0>
    # X|+> = |+>
    # X|-> = |->
    # X|+i> = ? X [1, i] = [i, 1] = i [1, -i] = i |-i>. Global phase doesn't matter for fidelity.
    # So target for |+i> is |-i> (up to phase).
    # X|-i> = ? X [1, -i] = [-i, 1] = -i [1, i] = -i |+i>.
    # So target for |-i> is |+i>.
    
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
    
    # Precompute H components
    # H = detuning * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I (a + a^dag) + i Q (a - a^dag))
    # n(n-1) is diag(0, 0, 2)
    n_nm1 = np.diag([0, 0, 2])
    anharmonic_term = 0.5 * ANHARMONICITY * n_nm1
    
    # a + a^dag
    a_plus_adag = a_mat + a_dag
    # a - a^dag
    a_minus_adag = a_mat - a_dag
    
    # Lindblad terms in Liouvillian form?
    # It's easier to simulate the density matrix using the master equation.
    # d rho = -i [H, rho] + L1 rho L1^dag - 0.5 {L1^dag L1, rho} + L2 rho L2^dag - 0.5 {L2^dag L2, rho}
    
    # Precompute L1^dag L1 and L2^dag L2
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    # We will simulate each state separately.
    fidelities = []
    
    for i in range(6):
        psi = states[i]
        rho = np.outer(psi, psi.conj())
        
        for step in range(N_STEPS):
            I_val = pulse[step, 0]
            Q_val = pulse[step, 1]
            
            # Construct H
            drive_term = 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
            H = detuning * n_mat + anharmonic_term + drive_term
            
            # Commutator [H, rho]
            comm = H @ rho - rho @ H
            
            # Lindblad terms
            # L1 rho L1^dag
            L1_rho_L1_dag = L1 @ rho @ L1.conj().T
            # L2 rho L2^dag
            L2_rho_L2_dag = L2 @ rho @ L2.conj().T
            
            # Anti-commutators
            anti1 = L1_dag_L1 @ rho + rho @ L1_dag_L1
            anti2 = L2_dag_L2 @ rho + rho @ L2_dag_L2
            
            # d rho / dt
            d_rho = -1j * comm + L1_rho_L1_dag - 0.5 * anti1 + L2_rho_L2_dag - 0.5 * anti2
            
            # Integrate: rho_next = rho + dt * d_rho (Euler) 
            # The task says "each step exponentiates the Liouvillian".
            # This implies we should form the Liouvillian matrix L such that d vec(rho)/dt = L vec(rho)
            # and then rho_next = expm(L * dt) vec(rho).
            # This is more accurate than Euler.
            
            # Construct Liouvillian L (9x9)
            # vec(A B C) = (C^T \otimes A) vec(B)
            # Term -i [H, rho] = -i (H rho - rho H)
            # = -i (H \otimes I) vec(rho) + i (I \otimes H^T) vec(rho)
            # Note: vec(H rho) = (I \otimes H^T) vec(rho)? 
            # Let's check: vec(H rho) = (rho^T \otimes I) vec(H)? No.
            # Standard identity: vec(A X B) = (B^T \otimes A) vec(X).
            # So vec(H rho) = (I \otimes H^T) vec(rho).
            # vec(rho H) = (H^T \otimes I) vec(rho).
            # So -i [H, rho] -> (-i (I \otimes H^T) + i (H^T \otimes I)) vec(rho).
            
            # Lindblad L_k rho L_k^dag -> (L_k^* \otimes L_k) vec(rho).
            # -0.5 {L_k^dag L_k, rho} -> -0.5 ( (L_k^dag L_k) \otimes I + I \otimes (L_k^dag L_k)^T ) vec(rho).
            
            I9 = np.eye(9, dtype=complex)
            I3 = np.eye(3, dtype=complex)
            
            L_mat = np.zeros((9, 9), dtype=complex)
            
            # Hamiltonian part
            L_mat += -1j * (np.kron(I3, H.T)) + 1j * (np.kron(H.T, I3))
            
            # Lindblad 1
            L_mat += np.kron(L1.conj(), L1) - 0.5 * (np.kron(L1_dag_L1, I3) + np.kron(I3, L1_dag_L1.T))
            
            # Lindblad 2
            L_mat += np.kron(L2.conj(), L2) - 0.5 * (np.kron(L2_dag_L2, I3) + np.kron(I3, L2_dag_L2.T))
            
            # Exponentiate
            # L_mat is 9x9.
            exp_L = expm(L_mat * DT)
            
            vec_rho = rho.flatten()
            vec_rho_next = exp_L @ vec_rho
            rho = vec_rho_next.reshape((3, 3))
            
        # Calculate fidelity
        psi_final = rho # No, rho is density matrix.
        # We need the state vector? Or just fidelity between rho and target density matrix?
        # Fidelity F = Tr(sqrt(sqrt(rho) rho_target sqrt(rho)))
        # For pure states, F = |<psi_target|psi_final>|^2.
        # Since we start with pure states and the evolution is CPTP, the final state might be mixed due to decoherence.
        # So we must use the general fidelity formula.
        rho_target = np.outer(targets[i], targets[i].conj())
        
        # Compute fidelity
        # Use scipy or manual.
        # sqrtm(rho) @ rho_target @ sqrtm(rho)
        # This is expensive. 
        # Alternative: for small systems, we can use the formula F = Tr(sqrt(sqrt(rho) rho_t sqrt(rho)))
        # Or just use the eigenvalue method.
        # Given T1, Tphi are large, decoherence is small. The state is nearly pure.
        # We can extract the dominant eigenvector of rho and compute overlap with target.
        # This is an approximation but might be sufficient if decoherence is low.
        # However, "infidelity = 1 - mean fidelity".
        # Let's try to compute exact fidelity.
        
        # Efficient fidelity for 3x3:
        # F = Tr( sqrtm( sqrtm(rho) @ rho_t @ sqrtm(rho) ) )
        # This requires two sqrtm calls.
        # Since we do this 6 times per simulation, and simulations are ~3000, it's 18000 sqrtm calls.
        # Might be slow.
        
        # Let's check if we can just use the dominant eigenvector.
        # w, v = np.linalg.eigh(rho)
        # psi_final = v[:, -1]
        # F = abs(np.vdot(targets[i], psi_final))**2
        # This is much faster.
        
        w, v = np.linalg.eigh(rho)
        psi_final = v[:, -1]
        F = abs(np.vdot(targets[i], psi_final))**2
        fidelities.append(F)
        
    return np.mean(fidelities)

def _compute_score(params):
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
    deadline = start_time + 290.0 # 10s margin for 300s limit
    
    # Initial guess based on seed description
    # sigma = 40/6, alpha = ? , area = 1.0
    # Typical DRAG alpha for transmon: 
    # The leakage term is proportional to chi. 
    # Correction is Q ~ chi * I'.
    # Let's start with alpha = 0.1 and refine.
    
    x0 = np.array([40.0/6.0, 0.1, 1.0])
    
    best_params = x0
    best_score = _compute_score(x0)
    
    # Simple local search
    # Use Nelder-Mead or similar.
    
    if time.time() < deadline - 20:
        try:
            res = minimize(_compute_score, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200, 'adaptive': True})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception as e:
            pass
            
    # Generate final pulse
    final_pulse = _generate_pulse(best_params)
    
    # Final validity check
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
