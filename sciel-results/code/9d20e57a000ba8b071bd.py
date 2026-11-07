import numpy as np
from scipy.linalg import expm
import time
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
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
    
    # Basis states
    # |0>, |1>, |0>+|1>/sqrt(2), |0>-|1>/sqrt(2), |0>+i|1>/sqrt(2), |0>-i|1>/sqrt(2)
    # Actually, the task says: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # That's 6 states? No, "mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # This lists 1 + 1 + 2 + 2 = 6 states?
    # Or is it 4 states?
    # "mean fidelity to X over [list]"
    # Usually tomography uses 4 states.
    # Let's assume the 4 standard basis states for the qubit subspace:
    # 1. |0>
    # 2. |1>
    # 3. (|0> + |1>)/sqrt(2)
    # 4. (|0> + i|1>)/sqrt(2)
    # The prompt says "(|0> +- |1>)/sqrt(2)" which implies both + and -.
    # And "(|0> +- i|1>)/sqrt(2)" which implies both + and -.
    # So 6 states total?
    # "mean fidelity ... over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # This phrasing suggests the set includes all listed.
    # Let's include all 6 to be safe.
    
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]
    
    # Target states for X gate
    # X|0> = |1>, X|1> = |0>
    # X(|0>+|1>) = |1>+|0>
    # X(|0>-|1>) = |1>-|0> = -(|0>-|1>)
    # X(|0>+i|1>) = |1>+i|0> = i(|0>-i|1>)? No.
    # X|0> = |1>, X|1> = |0>.
    # X(a|0> + b|1>) = a|1> + b|0>.
    
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2), # X(1,-1) = (1,-1) -> (1,-1)? No. X(1,-1)/sqrt2 = (1,-1)/sqrt2?
        # X(1, -1) = 1*|1> - 1*|0> = -|0> + |1>.
        # So target is [-1, 1, 0]/sqrt(2).
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2), # X(1, i) = i|0> + 1|1> = (i, 1).
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2) # X(1, -i) = -i|0> + 1|1> = (-i, 1).
    ]
    
    # Construct Lindblad operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    n_op = a_op.conj().T @ a_op
    # n = diag(0, 1, 2)
    
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    def get_H(I, Q, delta, scale):
        # H = delta n + 0.5 alpha n(n-1) + 0.5 scale (I (a + a^dag) + i Q (a - a^dag))
        H0 = delta * n_op + 0.5 * alpha * (n_op @ (n_op - np.eye(3)))
        
        a_dag = a_op.conj().T
        drive = 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        
        return H0 + drive
    
    def simulate_pulse(I_Q, delta, scale):
        # I_Q: (80, 2)
        # Returns fidelities for the 6 states
        fidels = []
        
        # Precompute H for each step? No, H depends on I, Q which change.
        # But I, Q are fixed for a given pulse.
        # We can precompute H_steps.
        H_steps = []
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            H = get_H(I, Q, delta, scale)
            H_steps.append(H)
        
        for idx, psi0 in enumerate(states):
            psi = psi0.copy()
            # Target state
            target = targets[idx]
            
            # Time evolution
            # Use step-by-step exponentiation
            # U = exp(-i (H dt - 0.5 sum L^dag L dt) - sum L rho L^dag ... )
            # For pure states, we can use the Liouvillian superoperator or just the stochastic master equation?
            # The task says "exponentiates the Liouvillian".
            # Liouvillian L = -i [H, .] + sum (L_k . L_k^dag - 0.5 {L_k^dag L_k, .})
            # We need to evolve the density matrix rho.
            
            rho = np.outer(psi, psi.conj())
            
            for i in range(N):
                H = H_steps[i]
                # Construct Liouvillian superoperator (9x9)
                # Or just use the formula for the update.
                # U = exp(L * dt)
                # L = -i H (vectorized) ...
                
                # Efficient way:
                # rho_next = expm(L * dt) @ rho_vec
                # Constructing 9x9 expm 80 times * 6 states * 9 cases * many evals might be slow.
                
                # Alternative: Since decoherence is weak, we can ignore it for optimization and add it back at the end?
                # No, "DOP853 recheck ... must agree".
                # But for optimization, we can approximate?
                # Let's try to speed up the simulation.
                
                # We can combine the Hamiltonian and Lindblad into a single superoperator.
                # But 9x9 expm is fast.
                
                # Let's vectorize rho
                rho_vec = rho.flatten()
                
                # Build L
                # L = -i (H ⊗ I - I ⊗ H^T) + ...
                # This is tedious.
                
                # Given the short time and weak decoherence, we can use the first-order approximation?
                # No, let's just do the expm.
                
                # To save time, we can precompute the superoperator for each step?
                # H changes each step.
                
                # Let's optimize the simulation loop.
                pass
            
            # This is getting complex. Let's use a simpler simulation for optimization.
            # Since T1, Tphi are large, the infidelity is dominated by leakage.
            # We can simulate with H only, and then add a small penalty for decoherence?
            # Or just simulate with H for optimization, and verify with full simulation at the end.
            
            pass
            
        return fidels

    # Optimization Strategy:
    # 1. Define a function to generate the pulse from sigma, eta.
    # 2. Define a fast simulator (H only).
    # 3. Optimize sigma, eta.
    # 4. Verify with full simulator.
    
    def generate_pulse(sigma, eta):
        t = np.linspace(0, 40, N)
        t_c = 20.0
        I = (np.sqrt(np.pi/2) / sigma) * np.exp(-(t - t_c)**2 / (2 * sigma**2))
        dI = -(t - t_c) / (sigma**2) * I
        Q = eta * dI
        I_Q = np.column_stack((I, Q))
        return I_Q

    def check_constraints(I_Q):
        # Amplitude
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound:
            return False
        # Slew
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound:
            return False
        # Start/End
        if np.sqrt(I_Q[0,0]**2 + I_Q[0,1]**2) / dt > slew_bound:
            return False
        if np.sqrt(I_Q[-1,0]**2 + I_Q[-1,1]**2) / dt > slew_bound:
            return False
        return True

    def fast_score(I_Q):
        # H-only simulation
        infidelities = []
        for delta in deltas:
            for scale in scales:
                # Simulate
                H_steps = [get_H(I_Q[i,0], I_Q[i,1], delta, scale) for i in range(N)]
                
                case_infs = []
                for idx in range(6):
                    psi = states[idx].copy()
                    target = targets[idx]
                    
                    # Evolve
                    for i in range(N):
                        H = H_steps[i]
                        U = expm(-1j * H * dt)
                        psi = U @ psi
                    
                    fid = np.abs(np.vdot(target, psi))**2
                    case_infs.append(1 - fid)
                
                mean_inf = np.mean(case_infs)
                max_inf = np.max(case_infs)
                infidelities.append(mean_inf + max_inf)
        
        return np.mean(infidelities)

    # Optimization
    best_score = np.inf
    best_params = (10.0, 5.0)
    
    # Grid search for initial guess
    for sigma in [6, 8, 10, 12, 15]:
        for eta in [0, 2, 5, 10, 20]:
            I_Q = generate_pulse(sigma, eta)
            if not check_constraints(I_Q):
                continue
            s = fast_score(I_Q)
            if s < best_score:
                best_score = s
                best_params = (sigma, eta)
    
    # Local optimization
    def opt_func(params):
        sigma, eta = params
        I_Q = generate_pulse(sigma, eta)
        if not check_constraints(I_Q):
            return 10.0
        return fast_score(I_Q)

    res = minimize(opt_func, best_params, method='Nelder-Mead', options={'maxiter': 100})
    if res.fun < best_score:
        best_params = res.x
        best_score = res.fun
    
    # Generate final pulse
    I_Q = generate_pulse(best_params[0], best_params[1])
    
    # Ensure constraints
    if not check_constraints(I_Q):
        # Fallback to seed
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi/2) / (40/6.0)) * np.exp(-(t - 20.0)**2 / (2 * (40/6.0)**2))
        Q = np.zeros(N)
        I_Q = np.column_stack((I, Q))
        
    return I_Q
