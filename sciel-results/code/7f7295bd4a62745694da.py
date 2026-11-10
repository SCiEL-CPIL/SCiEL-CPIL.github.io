"""Rewritten program for robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes for a 40 ns X-pi gate.
    Uses a DRAG-style Gaussian seed and applies small corrections to minimize infidelity.
    """
    if target_gate != 'X_pi':
        # For now, only handle X_pi
        raise ValueError("Only X_pi is supported")
    
    # Constants
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters
    chi = -2 * math.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_values = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
    scale_values = [0.97, 1.0, 1.03]
    T1 = 25000.0
    Tphi = 40000.0
    
    # Operators
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a  # diag(0, 1, 2)
    
    # Lindblad operators
    L1 = a / math.sqrt(T1)
    L2 = n_op * math.sqrt(2 / Tphi)
    
    def make_lindblad_superop(I, Q, detuning, scale):
        """Build the Liouvillian superoperator for given pulse parameters."""
        # Hamiltonian
        H = detuning * n_op + 0.5 * chi * n_op @ (n_op - 3 * np.eye(3)) 
        # Wait, the task says: H = detuning n + 0.5 anharmonicity n (n - 1)
        # Let me re-read: "H = detuning n + 0.5 anharmonicity n (n - 1)"
        # So it's 0.5 * chi * n @ (n - I), not n(n-1)
        H = detuning * n_op + 0.5 * chi * n_op @ (n_op - np.eye(3))
        
        # Drive term
        drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        H = H + drive
        
        # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
        # We need to build the superoperator matrix
        dim = 3
        total_dim = dim * dim
        L_super = np.zeros((total_dim, total_dim), dtype=complex)
        
        for i in range(total_dim):
            for j in range(total_dim):
                # rho_ej where e = i // dim, j_idx = i % dim
                # Actually, let's use vec convention: vec(rho) stacks columns
                # L(rho) = -i H rho + i rho H + sum_k (L_k rho L_k^dag - 0.5 L_k^dag L_k rho - 0.5 rho L_k^dag L_k)
                
                # Column j, row i of rho
                col = j
                row = i // dim
                # Actually, let me use a simpler approach: build the matrix by acting on basis vectors
                
                # Let's just build it directly
                pass
        
        # Better approach: build the superoperator using the formula
        # For each basis vector |i><j|, compute L(|i><j|) and place it in the matrix
        L_super = np.zeros((total_dim, total_dim), dtype=complex)
        
        for i in range(dim):
            for j in range(dim):
                # Basis operator |i><j|
                rho = np.zeros((dim, dim), dtype=complex)
                rho[i, j] = 1
                
                # Apply Liouvillian
                result = -1j * (H @ rho - rho @ H)
                for L in [L1, L2]:
                    Ldag = L.conj().T
                    result += L @ rho @ Ldag - 0.5 * (Ldag @ L @ rho + rho @ Ldag @ L)
                
                # Place in superoperator
                # vec(result) goes into column (j * dim + i) of L_super
                # Because vec stacks columns, and |i><j| has index j*dim+i in column-major
                col_idx = j * dim + i
                L_super[:, col_idx] = result.flatten(order='F')
        
        return L_super
    
    def simulate_pulse(pulse_IQ, detuning, scale):
        """Simulate the pulse and return the final density matrix."""
        L_super = make_lindblad_superop(pulse_IQ[0], pulse_IQ[1], detuning, scale)
        
        # Initial state: |0><0|
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1
        
        # Evolve
        for step in range(n_steps):
            I = pulse_IQ[step, 0]
            Q = pulse_IQ[step, 1]
            L_super = make_lindblad_superop(I, Q, detuning, scale)
            
            # Exponentiate: rho_next = exp(L_super * dt) @ vec(rho)
            # Use matrix exponential
            L_dt = L_super * dt
            U = sla.expm(L_dt)
            rho_vec = rho.flatten(order='F')
            rho_vec_new = U @ rho_vec
            rho = rho_vec_new.reshape((3, 3), order='F')
        
        return rho
    
    def compute_fidelity(rho, target_state):
        """Compute fidelity between density matrix and pure state."""
        # Fidelity = |<target|rho|target>| for pure target state
        target_vec = target_state.flatten(order='F')
        val = np.vdot(target_vec, rho @ target_vec)
        return np.real(val)
    
    def compute_infidelity(pulse_IQ):
        """Compute mean and max infidelity over the 9 cases."""
        target_states = [
            np.array([1, 0, 0], dtype=complex),  # |0>
            np.array([0, 1, 0], dtype=complex),  # |1>
            (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / math.sqrt(2),
            (np.array([1, 0, 0], dtype=complex) - np.array([0, 1, 0], dtype=complex)) / math.sqrt(2),
            (np.array([1, 0, 0], dtype=complex) + 1j * np.array([0, 1, 0], dtype=complex)) / math.sqrt(2),
            (np.array([1, 0, 0], dtype=complex) - 1j * np.array([0, 1, 0], dtype=complex)) / math.sqrt(2)
        ]
        
        # X gate maps |0> -> |1>, |1> -> -|0> (up to global phase)
        # Actually, X |0> = |1>, X |1> = |0>
        # So target for |0> is |1>, target for |1> is |0>
        # For superpositions, we need to apply X
        
        X_gate = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=complex)
        
        fidelities = []
        for detuning in detuning_values:
            for scale in scale_values:
                rho = simulate_pulse(pulse_IQ, detuning, scale)
                for target in target_states:
                    # Apply X gate to target
                    target_X = X_gate @ target
                    fid = compute_fidelity(rho, target_X)
                    fidelities.append(fid)
        
        fidelities = np.array(fidelities)
        infidelities = 1 - fidelities
        return np.mean(infidelities), np.max(infidelities)
    
    def check_validity(pulse_IQ):
        """Check amplitude and slew constraints."""
        # Amplitude constraint
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amp_bound + 1e-9):
            return False
        
        # Slew constraint: ||p[i+1] - p[i]|| / dt <= 0.12
        # From zero to step 1
        if np.linalg.norm(pulse_IQ[0]) / dt > slew_bound + 1e-9:
            return False
        
        for i in range(n_steps - 1):
            if np.linalg.norm(pulse_IQ[i+1] - pulse_IQ[i]) / dt > slew_bound + 1e-9:
                return False
        
        # From last step to zero
        if np.linalg.norm(pulse_IQ[-1]) / dt > slew_bound + 1e-9:
            return False
        
        return True
    
    # Generate DRAG-style Gaussian seed
    def generate_drag_pulse(sigma=40/6, area=math.pi):
        """Generate a DRAG-style Gaussian pulse with offset to 0 at edges."""
        t = np.linspace(0, T, n_steps)
        
        # Gaussian centered at T/2
        center = T / 2
        sigma_eff = sigma
        gaussian = np.exp(-0.5 * ((t - center) / sigma_eff)**2)
        
        # Normalize area to pi
        gaussian_area = np.trapz(gaussian, t)
        gaussian = gaussian * area / gaussian_area
        
        # DRAG correction: add derivative term
        # The derivative of the Gaussian
        dgaussian = -gaussian * (t - center) / sigma_eff**2
        
        # Scale the DRAG term
        drag_coeff = 0.0  # Start with no DRAG
        I = gaussian
        Q = drag_coeff * dgaussian
        
        # Offset to make edges zero
        # We need I[0] = I[-1] = 0 and Q[0] = Q[-1] = 0
        # Subtract a linear ramp
        I_offset = I - np.linspace(I[0], I[-1], n_steps)
        Q_offset = Q - np.linspace(Q[0], Q[-1], n_steps)
        
        return np.column_stack([I_offset, Q_offset])
    
    # Start with the seed
    pulse_IQ = generate_drag_pulse()
    
    # Check validity
    if not check_validity(pulse_IQ):
        # Scale down to meet constraints
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        max_amp = np.max(amplitudes)
        if max_amp > amp_bound:
            pulse_IQ = pulse_IQ * amp_bound / max_amp
        
        # Check slew again
        if not check_validity(pulse_IQ):
            # Apply a smoothing filter
            from scipy.signal import savgol_filter
            pulse_IQ[:, 0] = savgol_filter(pulse_IQ[:, 0], 5, 2)
            pulse_IQ[:, 1] = savgol_filter(pulse_IQ[:, 1], 5, 2)
    
    # Optimize: try to improve the pulse
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 10.0
    
    best_pulse = pulse_IQ.copy()
    best_score = np.inf
    
    # Simple local optimization: perturb the pulse and check if it improves
    np.random.seed(42)
    
    for iteration in range(1000):
        if time.time() - start_time > time_limit - safety_margin:
            break
        
        # Generate a random perturbation
        noise = np.random.randn(n_steps, 2) * 0.01
        candidate = best_pulse + noise
        
        # Clip to amplitude bound
        amplitudes = np.sqrt(candidate[:, 0]**2 + candidate[:, 1]**2)
        mask = amplitudes > amp_bound
        if np.any(mask):
            candidate[mask] = candidate[mask] * amp_bound / amplitudes[mask]
        
        # Check validity
        if not check_validity(candidate):
            continue
        
        # Compute score
        try:
            mean_inf, max_inf = compute_infidelity(candidate)
            score = mean_inf + max_inf
        except:
            continue
        
        if score < best_score:
            best_score = score
            best_pulse = candidate.copy()
    
    return best_pulse

# EVOLVE-BLOCK-END
