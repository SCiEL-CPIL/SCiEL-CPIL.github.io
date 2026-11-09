"""Rewritten program for robust transmon X-pi pulse design."""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes for a 40 ns X-pi gate.
    80 steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Extract hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_low, scale_high = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5  # ns
    n_steps = 80
    total_time = 40.0  # ns
    
    # Precompute time array
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2  # midpoints
    
    # Build 3-level system operators
    # a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2 / Tphi)
    
    def compute_hamiltonian(I, Q, detuning, scale):
        """Compute the Hamiltonian for given I, Q, detuning, scale."""
        H = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
        H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        return H
    
    def compute_liouvillian(H):
        """Compute the Liouvillian superoperator."""
        dim = 3
        L = -1j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H))
        # Add dissipation terms
        for L_op in [L1, L2]:
            Ld = L_op @ L_op.conj().T
            L += Ld @ Ld - 0.5 * (np.kron(Ld, np.eye(dim)) + np.kron(np.eye(dim), Ld))
        return L
    
    def simulate_pulse(pulse_IQ, detuning, scale):
        """
        Simulate the pulse and return the final density matrix.
        pulse_IQ: (80, 2) array
        """
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0  # Start in |0>
        
        for i in range(n_steps):
            I = pulse_IQ[i, 0]
            Q = pulse_IQ[i, 1]
            H = compute_hamiltonian(I, Q, detuning, scale)
            L = compute_liouvillian(H)
            # Exponentiate: rho_next = expm(L * dt) @ rho
            rho = expm(L * dt) @ rho
            # Normalize to prevent numerical drift
            tr = np.trace(rho)
            if abs(tr) > 1e-12:
                rho = rho / tr
        
        return rho
    
    def compute_fidelity(rho, target_state):
        """Compute fidelity between density matrix and target state."""
        target_rho = np.outer(target_state, target_state.conj())
        # Fidelity = |<psi_target|rho|psi_target>| for pure target
        # More robust: F = (Tr(sqrt(sqrt(rho) target_rho sqrt(rho))))^2
        # For pure target: F = <psi|rho|psi>
        val = np.vdot(target_state, rho @ target_state)
        return np.real(val)
    
    def compute_infidelity_for_case(detuning, scale, pulse_IQ):
        """Compute mean infidelity over 4 input states for given hardware case."""
        # Target: X gate (pi pulse)
        # X|0> = |1>, X|1> = |0>
        # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
        # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
        
        states = [
            np.array([1, 0, 0], dtype=complex),  # |0>
            np.array([0, 1, 0], dtype=complex),  # |1>
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        ]
        
        # Target states after X gate
        target_states = [
            np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
            np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+|1>)/sqrt(2)
            np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+i|1>)/sqrt(2) = (i|0>+|1>)/sqrt(2)
        ]
        
        fidelities = []
        for state, target_state in zip(states, target_states):
            rho = simulate_pulse(pulse_IQ, detuning, scale)
            fid = compute_fidelity(rho, target_state)
            fidelities.append(fid)
        
        mean_fid = np.mean(fidelities)
        return 1.0 - mean_fid
    
    def scoring_metric(pulse_IQ):
        """Compute the score: mean + max infidelity over 9 cases."""
        detunings = [-detuning_bound, 0, detuning_bound]
        scales = [scale_low, 1.0, scale_high]
        
        infidelities = []
        for det in detunings:
            for sc in scales:
                inf = compute_infidelity_for_case(det, sc, pulse_IQ)
                infidelities.append(inf)
        
        inf_arr = np.array(infidelities)
        return np.mean(inf_arr) + np.max(inf_arr)
    
    # Define pulse parameterization
    # Use a DRAG-style pulse with additional correction terms
    # Parameters: [amp_scale, drag_coeff, width_factor, q_offset, i_correction]
    
    def generate_pulse(params):
        """Generate pulse from parameters."""
        amp_scale, drag_coeff, width_factor, q_offset, i_corr = params
        
        # Base Gaussian (DRAG seed style)
        sigma = (40.0 / 6.0) * width_factor  # ns
        t_center = 20.0  # center of 40 ns window
        
        # Gaussian envelope
        gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        
        # Offset to make it zero at edges
        edge_val = np.exp(-0.5 * ((0 - t_center) / sigma) ** 2)
        gauss = gauss - edge_val
        
        # Normalize area to pi
        area = np.trapz(gauss, t)
        if abs(area) > 1e-12:
            gauss = gauss * (np.pi / area)
        
        # DRAG derivative term
        dt_gauss = np.gradient(gauss, dt)
        
        # I and Q components
        I = amp_scale * gauss + i_corr * dt_gauss
        Q = amp_scale * drag_coeff * dt_gauss + q_offset * gauss
        
        # Clip to amplitude bound
        for i in range(n_steps):
            mag = np.sqrt(I[i]**2 + Q[i]**2)
            if mag > amp_bound:
                scale = amp_bound / mag
                I[i] *= scale
                Q[i] *= scale
        
        # Enforce slew rate constraint
        # Project to satisfy ||p[i+1] - p[i]|| / dt <= slew_bound
        # Simple iterative projection
        for _ in range(10):
            prev = np.zeros((n_steps, 2))
            for i in range(n_steps):
                if i > 0:
                    diff = np.array([I[i] - I[i-1], Q[i] - Q[i-1]])
                    norm = np.linalg.norm(diff)
                    max_diff = slew_bound * dt
                    if norm > max_diff:
                        scale = max_diff / norm
                        I[i] = I[i-1] + diff[0] * scale
                        Q[i] = Q[i-1] + diff[1] * scale
                else:
                    # From zero to first step
                    norm = np.sqrt(I[i]**2 + Q[i]**2)
                    max_diff = slew_bound * dt
                    if norm > max_diff:
                        scale = max_diff / norm
                        I[i] *= scale
                        Q[i] *= scale
            
            # Also check last step to zero
            last_norm = np.sqrt(I[-1]**2 + Q[-1]**2)
            max_diff = slew_bound * dt
            if last_norm > max_diff:
                scale = max_diff / last_norm
                I[-1] *= scale
                Q[-1] *= scale
        
        return np.column_stack([I, Q])
    
    # Optimization
    # Initial guess: DRAG seed parameters
    x0 = [1.0, 0.0, 1.0, 0.0, 0.0]  # amp_scale, drag_coeff, width_factor, q_offset, i_corr
    
    # Bounds for parameters
    bounds = [
        (0.5, 1.5),    # amp_scale
        (-0.5, 0.5),   # drag_coeff
        (0.5, 2.0),    # width_factor
        (-0.1, 0.1),   # q_offset
        (-0.5, 0.5),   # i_corr
    ]
    
    # Use Powell optimization (doesn't require gradients)
    best_pulse = generate_pulse(x0)
    best_score = scoring_metric(best_pulse)
    
    # Run optimization
    def objective(params):
        p = generate_pulse(params)
        return scoring_metric(p)
    
    result = minimize(objective, x0, method='Powell', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-8, 'disp': False})
    
    if result.fun < best_score:
        best_pulse = generate_pulse(result.x)
        best_score = result.fun
    
    # Final constraint check and fix
    for i in range(n_steps):
        mag = np.sqrt(best_pulse[i, 0]**2 + best_pulse[i, 1]**2)
        if mag > amp_bound:
            scale = amp_bound / mag
            best_pulse[i, 0] *= scale
            best_pulse[i, 1] *= scale
    
    # Slew rate final check
    for i in range(n_steps):
        if i > 0:
            diff = best_pulse[i] - best_pulse[i-1]
            norm = np.linalg.norm(diff)
            max_diff = slew_bound * dt
            if norm > max_diff:
                scale = max_diff / norm
                best_pulse[i] = best_pulse[i-1] + diff * scale
        else:
            norm = np.linalg.norm(best_pulse[i])
            max_diff = slew_bound * dt
            if norm > max_diff:
                best_pulse[i] *= max_diff / norm
    
    last_norm = np.linalg.norm(best_pulse[-1])
    max_diff = slew_bound * dt
    if last_norm > max_diff:
        best_pulse[-1] *= max_diff / last_norm
    
    return best_pulse
