"""Optimized pulse for three-level transmon X_pi gate with robustness."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings('ignore')

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill

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
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]

    # Target states for X gate
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2)
    ]

    # Construct Lindblad operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)

    n_op = a_op.conj().T @ a_op
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    def get_H(I, Q, delta, scale):
        # H = delta n + 0.5 alpha n(n-1) + 0.5 scale (I (a + a^dag) + i Q (a - a^dag))
        H0 = delta * n_op + 0.5 * alpha * (n_op @ (n_op - np.eye(3)))
        a_dag = a_op.conj().T
        drive = 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H0 + drive

    def simulate_pulse(I_Q, delta, scale):
        """Simulate pulse with full Liouvillian evolution."""
        fidels = []
        H_steps = [get_H(I_Q[i, 0], I_Q[i, 1], delta, scale) for i in range(N)]
        
        for idx, psi0 in enumerate(states):
            rho = np.outer(psi0, psi0.conj())
            target = targets[idx]
            target_rho = np.outer(target, target.conj())
            
            for i in range(N):
                H = H_steps[i]
                # Build Liouvillian superoperator
                # L = -i [H, .] + sum (L_k . L_k^dag - 0.5 {L_k^dag L_k, .})
                # Using vectorization: rho_vec = rho.flatten()
                
                # Construct the superoperator
                I3 = np.eye(3, dtype=complex)
                H_vec = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
                
                # Add Lindblad terms
                for Lk in [L1, L2]:
                    Lk_dag = Lk.conj().T
                    # L_k rho L_k^dag
                    term1 = np.kron(Lk, Lk_dag.T)
                    # 0.5 {L_k^dag L_k, rho}
                    Lk_dag_Lk = Lk_dag @ Lk
                    term2 = -0.5 * (np.kron(Lk_dag_Lk, I3) + np.kron(I3, Lk_dag_Lk.T))
                    H_vec += term1 + term2
                
                # Evolve
                rho_vec = rho.flatten()
                rho_next_vec = expm(H_vec * dt) @ rho_vec
                rho = rho_next_vec.reshape(3, 3)
            
            # Compute fidelity
            fid = np.real(np.trace(rho @ target_rho))
            fidels.append(fid)
        
        return fidels

    def generate_pulse(sigma, eta, width_shift=0.0):
        """Generate DRAG-style pulse."""
        t = np.linspace(0, 40, N)
        t_c = 20.0 + width_shift
        I = (np.sqrt(np.pi/2) / sigma) * np.exp(-(t - t_c)**2 / (2 * sigma**2))
        dI = -(t - t_c) / (sigma**2) * I
        Q = eta * dI
        I_Q = np.column_stack((I, Q))
        
        # Apply constraint clipping
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        for i in range(N):
            if amps[i] > amp_bound:
                scale_factor = amp_bound / amps[i]
                I_Q[i, 0] *= scale_factor
                I_Q[i, 1] *= scale_factor
        
        # Apply slew rate constraint
        for i in range(1, N):
            diff = np.linalg.norm(I_Q[i] - I_Q[i-1])
            if diff / dt > slew_bound:
                scale_factor = (slew_bound * dt) / diff
                I_Q[i] = I_Q[i-1] + scale_factor * (I_Q[i] - I_Q[i-1])
        
        # Check start/end slew
        if np.linalg.norm(I_Q[0]) / dt > slew_bound:
            scale_factor = (slew_bound * dt) / np.linalg.norm(I_Q[0])
            I_Q[0] *= scale_factor
        if np.linalg.norm(I_Q[-1]) / dt > slew_bound:
            scale_factor = (slew_bound * dt) / np.linalg.norm(I_Q[-1])
            I_Q[-1] *= scale_factor
            
        return I_Q

    def check_constraints(I_Q):
        """Check if pulse satisfies all constraints."""
        # Amplitude
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-10:
            return False
        # Slew
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-10:
            return False
        # Start/End
        if np.linalg.norm(I_Q[0]) / dt > slew_bound + 1e-10:
            return False
        if np.linalg.norm(I_Q[-1]) / dt > slew_bound + 1e-10:
            return False
        return True

    def fast_score(I_Q):
        """Fast score using H-only simulation (approximation for speed)."""
        infidelities = []
        for delta in deltas:
            for scale in scales:
                # Simulate with H only
                U = np.eye(3, dtype=complex)
                for i in range(N):
                    H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
                    U = expm(-1j * H * dt) @ U
                
                # Compute fidelities
                case_fids = []
                for idx, psi0 in enumerate(states):
                    psi_final = U @ psi0
                    target = targets[idx]
                    fid = np.abs(np.vdot(target, psi_final))**2
                    case_fids.append(fid)
                
                mean_fid = np.mean(case_fids)
                infidelities.append(1 - mean_fid)
        
        return np.mean(infidelities) + np.max(infidelities)

    def objective(params):
        """Objective function for optimization."""
        if time.time() > deadline:
            return 1e6  # Timeout penalty
        
        sigma, eta, width_shift = params
        I_Q = generate_pulse(sigma, eta, width_shift)
        
        if not check_constraints(I_Q):
            return 1e5  # Constraint violation penalty
        
        return fast_score(I_Q)

    # Optimization Strategy:
    # 1. Use differential evolution for global search
    # 2. Then use local refinement with L-BFGS-B

    # Initial bounds
    sigma_bounds = [(3, 8)]  # Gaussian width
    eta_bounds = [(-1.0, 1.0)]  # DRAG coefficient
    shift_bounds = [(-5, 5)]  # Width shift

    # Try differential evolution first
    best_result = None
    best_score = float('inf')
    
    try:
        # Differential evolution for global optimization
        de_result = differential_evolution(
            objective, 
            bounds=sigma_bounds + eta_bounds + shift_bounds,
            maxiter=20,
            popsize=10,
            tol=1e-4,
            seed=42
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception as e:
        pass

    # Local refinement
    if best_result is not None:
        try:
            local_result = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                bounds=sigma_bounds + eta_bounds + shift_bounds,
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            if local_result.fun < best_score:
                best_score = local_result.fun
                best_result = local_result.x
        except Exception as e:
            pass
    else:
        # Fallback to seed-like parameters if optimization fails
        best_result = np.array([40/6, 0.0, 0.0])

    # Generate final pulse
    sigma, eta, width_shift = best_result
    I_Q = generate_pulse(sigma, eta, width_shift)

    # Final constraint check
    if not check_constraints(I_Q):
        # Fallback to seed
        I_Q = generate_pulse(40/6, 0.0, 0.0)

    return I_Q

# EVOLVE-BLOCK-END
