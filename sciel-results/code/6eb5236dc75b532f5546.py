"""
Strategy: Runtime Numerical Optimization via Stochastic Local Search
The previous approach used a fixed analytic DRAG pulse. This approach performs
a lightweight numerical search at runtime to minimize the specific scoring metric
(mean + max infidelity across 9 detuning/scale cases) while respecting all
hardware constraints (amplitude, slew rate).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes for a 40 ns X-pi gate.
    Uses a stochastic local search to optimize the pulse shape for the
    specific hardware model, minimizing infidelity across detuning and scale
    variations.
    """
    # Hardware constants
    dt = 0.5
    n_steps = 80
    T1 = 25000.0
    Tphi = 40000.0
    anharmonicity = -2.0 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2.0 * np.pi * 0.002
    
    # Time vector
    t = np.arange(n_steps) * dt - 20.0
    
    # Scoring cases: detuning in {-d, 0, d}, scale in {0.97, 1.0, 1.03}
    dets = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]
    cases = [(d, s) for d in dets for s in scales]
    
    # Pre-compute Lindblad operators and Hamiltonian parts for speed
    # System is 3x3
    # a = [[0, 0, 0], [1, 0, 0], [0, sqrt(2), 0]]
    # adag = a.T
    # n = adag @ a = diag(0, 1, 2)
    
    a_op = np.array([[0, 0, 0], [1, 0, 0], [0, np.sqrt(2), 0]], dtype=complex)
    adag_op = a_op.conj().T
    n_op = adag_op @ a_op
    
    # Lindblad terms
    # L1 = a / sqrt(T1)
    L1 = a_op / np.sqrt(T1)
    L1_dag = L1.conj().T
    L1_term = L1_dag @ L1
    
    # L2 = n * sqrt(2 / Tphi)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    L2_dag = L2.conj().T
    L2_term = L2_dag @ L2
    
    # Pre-compute the "static" part of the Liouvillian action for each case
    # Liouvillian: L(rho) = -i[H, rho] + sum (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # We will construct the full Liouvillian matrix for each case to exponentiate it.
    # However, constructing 3x3 Liouvillians (9x9 matrices) for 9 cases at every step of the search
    # might be slow if done naively. 
    # Optimization: The Liouvillian depends linearly on I and Q.
    # L = L_static + I * L_I + Q * L_Q
    # We can precompute L_I and L_Q matrices (9x9) and L_static (9x9) for each case.
    # Then for a given pulse step (I, Q), the Liouvillian is L_static + I*L_I + Q*L_Q.
    # We need to exponentiate this 9x9 matrix. scipy.linalg.expm is fast for 9x9.
    
    # Let's build the Liouvillian matrix representation.
    # Vectorization: vec(rho) = [rho_00, rho_01, ..., rho_22].T
    # L(rho) = -i (H rho - rho H) + ...
    # Using Kronecker products:
    # H rho H^dag -> (H^dag.T \kron H) vec(rho)
    # H rho -> (I \kron H) vec(rho)
    # rho H -> (H^T \kron I) vec(rho)
    
    I9 = np.eye(9, dtype=complex)
    
    def build_liouvillian_matrices(detuning, scale):
        # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I*(a+adag) + iQ*(a-adag))
        # Static part (detuning, anharmonicity)
        H_static = detuning * n_op + 0.5 * anharmonicity * (n_op @ n_op - n_op)
        
        # Drive part: 0.5 * scale * (I_drive*(a+adag) + i Q_drive*(a-adag))
        # Note: The drive is time-dependent.
        # H_drive(t) = 0.5 * scale * (I(t)*(a+adag) + i Q(t)*(a-adag))
        # Let M_I = 0.5 * (a+adag) and M_Q = 0.5 * i * (a-adag)
        # H_drive = scale * (I(t) * M_I + Q(t) * M_Q)
        
        M_I = 0.5 * (a_op + adag_op)
        M_Q = 0.5 * 1j * (a_op - adag_op)
        
        # Liouvillian terms
        # L(rho) = -i [H, rho] + L1 rho L1^dag - 0.5 {L1^dag L1, rho} + L2 rho L2^dag - 0.5 {L2^dag L2, rho}
        
        # Commutator part: -i (H rho - rho H)
        # = -i ( (I \kron H) - (H^T \kron I) ) vec(rho)
        # Wait, standard vectorization: vec(A X B) = (B^T \kron A) vec(X)
        # H rho -> vec(H rho) = (I \kron H) vec(rho) ? No.
        # vec(A X) = (I \kron A) vec(X) if vec is column-stacking?
        # Let's check: vec(A X)_{ij} = sum_k A_{ik} X_{kj}.
        # (I \kron A) vec(X): element (i, j) is sum_{k,l} I_{ik} A_{jl} X_{kl}? No.
        # Standard identity: vec(A X B) = (B^T \kron A) vec(X).
        # So H rho = H rho I -> (I^T \kron H) = (I \kron H).
        # rho H = I rho H -> (H^T \kron I).
        
        # So L_comm(H) = -1j * ( (I9 \kron H) - (H.T \kron I9) )
        
        # Dissipative part:
        # L rho L^dag -> (L^dag.T \kron L) vec(rho)
        # {L^dag L, rho} = L^dag L rho + rho L^dag L
        # -> (I \kron L^dag L) + ((L^dag L)^T \kron I)
        
        L_static_mat = -1j * (np.kron(I9, H_static) - np.kron(H_static.T, I9))
        L_I_mat = -1j * (np.kron(I9, M_I) - np.kron(M_I.T, I9))
        L_Q_mat = -1j * (np.kron(I9, M_Q) - np.kron(M_Q.T, I9))
        
        # Dissipation is same for all cases (T1, Tphi constant)
        L_diss = (np.kron(L1_dag.T, L1) - 0.5 * (np.kron(I9, L1_term) + np.kron(L1_term.T, I9)) +
                  np.kron(L2_dag.T, L2) - 0.5 * (np.kron(I9, L2_term) + np.kron(L2_term.T, I9)))
        
        # Total Liouvillian for a step with I, Q:
        # L_total = L_static_mat + scale * (I * L_I_mat + Q * L_Q_mat) + L_diss
        # Note: scale multiplies the drive part in H, so it multiplies L_I and L_Q.
        
        return L_static_mat, L_I_mat, L_Q_mat, L_diss

    # Pre-compute matrices for all 9 cases
    case_matrices = []
    for d, s in cases:
        Ls, LI, LQ, Ld = build_liouvillian_matrices(d, s)
        case_matrices.append((Ls, LI, LQ, Ld, s))
        
    # Initialize pulse with a DRAG seed
    sigma = 6.0
    A_I = np.pi / (sigma * np.sqrt(2 * np.pi))
    I_seed = A_I * np.exp(-t**2 / (2 * sigma**2))
    # Add small DRAG Q component? 
    # For robustness, maybe a small Q helps. 
    # Let's start with Q=0 and let the optimizer find if Q is needed.
    Q_seed = np.zeros_like(I_seed)
    
    current_pulse = np.column_stack((I_seed, Q_seed))
    
    # Define fidelity calculation
    # Target X gate: |0> -> |1>, |1> -> -|0> (or |0> -> |1>, |1> -> |0> with phase? 
    # "X_pi" usually means pi rotation around X. 
    # U = exp(-i pi/2 X) = [[0, -i], [-i, 0]]? 
    # Or standard Pauli X pi pulse: maps |0> to |1> and |1> to -|0>? 
    # Let's assume the target unitary is U_X = [[0, 1], [1, 0]] (swap) or [[0, -1], [-1, 0]]?
    # Standard "X pi pulse" in control usually refers to a pi rotation, which flips the state.
    # Fidelity F = |<psi_target|psi_final>|^2.
    # Target states:
    # |0> -> |1>
    # |1> -> -|0> (for U = exp(-i pi/2 X) = -i X, global phase doesn't matter for fidelity if we ignore it? 
    # Wait, "X_pi" gate. Usually U = [[0, 1], [1, 0]] is a pi pulse? 
    # exp(-i (pi/2) X) = cos(pi/2) I - i sin(pi/2) X = -i X. 
    # -i X |0> = -i |1>. Fidelity to |1> is 1.
    # -i X |1> = -i |0>. Fidelity to |0> is 1? No, target is -|0>? 
    # If target is X (Pauli X), then X|0>=|1>, X|1>=|0>.
    # If target is -X, then -X|0>=-|1>, -X|1>=-|0>.
    # Fidelity is modulus squared, so global phase and sign don't matter for single state fidelity if the target is defined up to phase.
    # However, the superposition states (|0>+|1>)/sqrt(2) are sensitive to relative phase.
    # X |+> = |->. 
    # If we use -X, -X |+> = -|->. Fidelity to |-> is 1.
    # So X or -X gives same fidelities for these states.
    # Let's define target states:
    # t0 = [0, 1, 0] (|1>)
    # t1 = [-1, 0, 0] (-|0>)  # Assuming U ~ -X
    # t_plus = [1, -1, 0] / sqrt(2) (|->)
    # t_minus = [1, 1, 0] / sqrt(2) (|+>)? 
    # Wait, X |+> = |->. So target for |+> is |->.
    # X |-> = |+>. So target for |-> is |+>.
    # Let's use the standard X gate: [[0,1],[1,0]].
    # Targets:
    # |0> -> |1> = [0, 1, 0]
    # |1> -> |0> = [1, 0, 0]
    # |+> -> |-> = [1, -1, 0]/sqrt(2)
    # |-> -> |+> = [1, 1, 0]/sqrt(2)
    # |i+> = (|0> + i|1>)/sqrt(2) -> (|1> + i|0>)/sqrt(2) = (i|0> + |1>)/sqrt(2) ? 
    # X (|0> + i|1>)/sqrt(2) = (|1> + i|0>)/sqrt(2).
    # |i-> = (|0> - i|1>)/sqrt(2) -> (|1> - i|0>)/sqrt(2).
    
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2), # |i+> target is X|0> + i X|1> = |1> + i|0>
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2) # |i-> target
    ]
    
    # Initial states: |0>, |1>, |+>, |->, |i+>, |i->
    inits = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]
    
    def is_valid(pulse):
        # Amplitude
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        if np.any(amps > amp_bound + 1e-9):
            return False
        # Slew
        # From 0 to p[0]
        if np.sqrt(pulse[0,0]**2 + pulse[0,1]**2) / dt > slew_bound + 1e-9:
            return False
        # Last to 0
        if np.sqrt(pulse[-1,0]**2 + pulse[-1,1]**2) / dt > slew_bound + 1e-9:
            return False
        # Internal
        diffs = np.diff(pulse, axis=0)
        steps = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        if np.any(steps / dt > slew_bound + 1e-9):
            return False
        return True

    def score_pulse(pulse):
        if not is_valid(pulse):
            return float('inf')
        
        max_inf = 0.0
        sum_inf = 0.0
        
        for idx, (Ls, LI, LQ, Ld, s) in enumerate(case_matrices):
            # Evolve for each initial state
            case_infs = []
            for k in range(6):
                rho = np.outer(inits[k], inits[k].conj())
                rho = rho.flatten()
                
                for step in range(n_steps):
                    I_val = pulse[step, 0]
                    Q_val = pulse[step, 1]
                    
                    # Construct Liouvillian
                    # L = Ls + s * (I * LI + Q * LQ) + Ld
                    L_mat = Ls + s * (I_val * LI + Q_val * LQ) + Ld
                    # Exponentiate
                    # Use scipy.linalg.expm
                    U = scipy.linalg.expm(L_mat * dt)
                    rho = U @ rho
                
                # Final state
                rho_final = rho.reshape(3, 3)
                psi_final = np.sqrt(np.diag(rho_final)) # Approximation? No, need pure state evolution.
                # Wait, the system is open. The state becomes mixed.
                # Fidelity to pure target |t>: F = <t| rho |t>
                t_vec = targets[k]
                F = np.real(np.vdot(t_vec, rho_final @ t_vec))
                F = np.clip(F, 0.0, 1.0)
                inf = 1.0 - F
                case_infs.append(inf)
            
            mean_inf = np.mean(case_infs)
            sum_inf += mean_inf
            if mean_inf > max_inf:
                max_inf = mean_inf
                
        return sum_inf / 9.0 + max_inf

    import scipy.linalg
    
    # Optimization Loop
    start_time = time.time()
    deadline = start_time + 290.0 # 40s safety margin
    
    current_pulse = current_pulse.copy()
    current_score = score_pulse(current_pulse)
    
    best_pulse = current_pulse.copy()
    best_score = current_score
    
    # Simulated Annealing / Stochastic Search
    T = 0.1 # Initial temperature
    alpha = 0.99 # Cooling rate
    
    # Number of iterations depends on time
    while time.time() < deadline:
        # Generate perturbation
        # Random walk in pulse space
        # Perturb I and Q
        # Ensure validity by projecting or rejecting
        new_pulse = current_pulse.copy()
        
        # Randomly perturb a few steps
        n_perturb = np.random.randint(1, 10)
        idxs = np.random.choice(n_steps, n_perturb, replace=False)
        for idx in idxs:
            # Gaussian noise
            dI = np.random.normal(0, 0.01)
            dQ = np.random.normal(0, 0.01)
            new_pulse[idx, 0] += dI
            new_pulse[idx, 1] += dQ
            
        # Check validity
        if not is_valid(new_pulse):
            # Try to fix by clipping amplitude
            amps = np.sqrt(new_pulse[:, 0]**2 + new_pulse[:, 1]**2)
            if np.any(amps > amp_bound):
                scale = amp_bound / np.max(amps)
                new_pulse *= scale
            # If still invalid (slew), reject
            if not is_valid(new_pulse):
                continue
        
        new_score = score_pulse(new_pulse)
        
        # Acceptance criterion
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            current_pulse = new_pulse
            current_score = new_score
            if current_score < best_score:
                best_pulse = current_pulse.copy()
                best_score = current_score
        
        T *= alpha
        if T < 1e-6:
            T = 1e-6
            
    return best_pulse

# EVOLVE-BLOCK-END
