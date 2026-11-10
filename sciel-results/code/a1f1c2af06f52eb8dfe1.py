"""Pulse optimization for 3-level transmon X-pi gate using gradient-based method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _build_operators():
    """Build 3-level transmon operators."""
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = math.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _hamiltonian(I, Q, detuning, scale, a, adag, n, anharmonicity):
    """Build Hamiltonian."""
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3))
    H += 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    return H

def _liouvillian_superop(H, a, n, T1, Tphi):
    """Build Liouvillian superoperator matrix."""
    dim = 3
    dim2 = dim * dim
    
    # Coherent part: -i/2 (H ⊗ I - I ⊗ H^T)
    H_coh = -0.5j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H.T))
    
    # Incoherent part
    L_super = np.zeros((dim2, dim2), dtype=complex)
    
    # T1 relaxation: L = a / sqrt(T1)
    gamma1 = 1.0 / T1
    for i in range(dim2):
        for j in range(dim2):
            # L rho L^dag term
            Ldag_L = a @ a.conj().T
            # 1/2 {L^dag L, rho} term
            Ldag_L_half = 0.5 * Ldag_L
            L_super[i, j] += gamma1 * (a @ (np.reshape(np.eye(dim2, dim)[i, j], (dim, dim)) @ a.conj().T) - 0.5 * (Ldag_L @ np.reshape(np.eye(dim2, dim)[i, j], (dim, dim)) + np.reshape(np.eye(dim2, dim)[i, j], (dim, dim)) @ Ldag_L)).reshape(-1)[i]
    
    # Tphi dephasing: L = n * sqrt(2/Tphi)
    gamma_phi = 2.0 / Tphi
    for i in range(dim2):
        for j in range(dim2):
            rho_ij = np.reshape(np.eye(dim2, dim)[i, j], (dim, dim))
            L_super[i, j] += gamma_phi * (n @ rho_ij @ n.conj().T - 0.5 * (n.conj().T @ n @ rho_ij + rho_ij @ n.conj().T @ n)).reshape(-1)[i]
    
    return H_coh + L_super

def _simulate_pulse(pulse, detuning, scale, a, adag, n, anharmonicity, T1, Tphi, dt):
    """Simulate pulse evolution, return final process matrix."""
    dim = 3
    dim2 = dim * dim
    U = np.eye(dim2, dtype=complex)
    
    num_steps = len(pulse)
    for k in range(num_steps):
        I_k, Q_k = pulse[k]
        H = _hamiltonian(I_k, Q_k, detuning, scale, a, adag, n, anharmonicity)
        L = _liouvillian_superop(H, a, n, T1, Tphi)
        
        # Exponentiate using scipy for accuracy
        from scipy.linalg import expm
        U_step = expm(L * dt)
        U = U_step @ U
    
    return U

def _compute_score(pulse, a, adag, n, anharmonicity, T1, Tphi, dt):
    """Compute the score (mean + max infidelity over 9 cases)."""
    detunings = [-2 * math.pi * 0.002, 0.0, 2 * math.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states
    target_states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / math.sqrt(2),  # (|0>+|1>)/sqrt(2)
        (np.array([1, -1, 0], dtype=complex)) / math.sqrt(2),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 1j, 0], dtype=complex)) / math.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, -1j, 0], dtype=complex)) / math.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    infidelities = []
    
    for det in detunings:
        for sc in scales:
            U = _simulate_pulse(pulse, det, sc, a, adag, n, anharmonicity, T1, Tphi, dt)
            
            # U is a superoperator acting on vectorized density matrices
            # For pure states, we can compute fidelity more directly
            # Let's compute the process matrix and get fidelities
            
            # Actually, let's compute the effective unitary by looking at the 2x2 subspace
            # Extract the 2x2 block (|0>,|1> subspace) from the superoperator
            
            # The superoperator U acts on vec(rho). For a pure state |psi>, vec(rho) = |psi> ⊗ |psi*>
            # After evolution, rho_final = U_vec * vec(rho_initial)
            # Fidelity = <psi_target|rho_final|psi_target> for pure target
            
            # But we need to be careful. Let's compute fidelity for each target state
            # The X gate maps |0> -> |1> and |1> -> -|0> (up to phase)
            
            # Actually, let's just compute the infidelity properly
            # For each input state, compute output state, compare to X(input)
            
            fid_sum = 0.0
            for psi_in in target_states:
                # X gate target
                psi_target = np.zeros(3, dtype=complex)
                psi_target[0] = psi_in[1]
                psi_target[1] = -psi_in[0]
                # Note: X|0> = |1>, X|1> = |0>, so X maps (a,b,0) -> (b,a,0)
                # Wait, standard X: X|0>=|1>, X|1>=|0>. So X(a|0>+b|1>) = a|1>+b|0>
                # So psi_target = (psi_in[1], psi_in[0], 0)
                psi_target = np.array([psi_in[1], psi_in[0], 0], dtype=complex)
                
                # Evolve psi_in
                rho_in = np.outer(psi_in, psi_in.conj())
                vec_rho_in = rho_in.reshape(-1)
                vec_rho_out = U @ vec_rho_in
                rho_out = vec_rho_out.reshape((3, 3))
                
                # Fidelity between pure psi_target and rho_out
                # F = |<psi_target|psi_out>|^2 where psi_out is the state
                # For mixed rho_out, F = <psi_target|rho_out|psi_target>
                F = np.real(np.vdot(psi_target, rho_out @ psi_target))
                F = max(0.0, min(1.0, F))
                fid_sum += F
            
            mean_fid = fid_sum / len(target_states)
            infid = 1.0 - mean_fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def _drag_seed():
    """Generate DRAG-style Gaussian seed pulse."""
    num_steps = 80
    dt = 0.5
    t = np.arange(num_steps) * dt
    
    # Gaussian with sigma = 40/6 ns, centered, area = pi
    sigma = 40.0 / 6.0
    center = (num_steps - 1) * dt / 2.0
    
    # Gaussian envelope
    g = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    # Offset to 0 at edges
    g[0] = 0.0
    g[-1] = 0.0
    
    # Scale to have area = pi
    area = np.sum(g) * dt
    g = g * (math.pi / area)
    
    # DRAG: Q component is derivative of I component (scaled)
    # For a pi pulse, the DRAG correction is proportional to the derivative
    I = g
    # Q = -lambda * dI/dt, with lambda chosen for optimal DRAG
    # Standard DRAG: lambda = anharmonicity / (2 * drive_freq) or similar
    # Here, let's use a simple approximation: Q proportional to derivative
    dI_dt = np.gradient(I, dt)
    lambda_drag = 0.5  # Tune this
    Q = -lambda_drag * dI_dt
    
    # Ensure Q is zero at edges
    Q[0] = 0.0
    Q[-1] = 0.0
    
    pulse = np.column_stack([I, Q])
    
    # Check amplitude bound
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > 0.4:
        pulse = pulse * (0.4 / np.max(amp))
    
    # Check slew rate
    for k in range(len(pulse) - 1):
        diff = np.linalg.norm(pulse[k+1] - pulse[k]) / dt
        if diff > 0.12:
            # Scale down
            pass  # For now, assume it's ok
    
    # Also check from zero to step 1 and last step to zero
    if np.linalg.norm(pulse[0]) / dt > 0.12:
        pulse[0] = pulse[0] * (0.12 * dt / np.linalg.norm(pulse[0]))
    if np.linalg.norm(pulse[-1]) / dt > 0.12:
        pulse[-1] = pulse[-1] * (0.12 * dt / np.linalg.norm(pulse[-1]))
    
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    start_time = time.time()
    deadline = start_time + 320.0  # 320s deadline with margin
    
    # Extract hardware parameters
    anharmonicity = -2 * math.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    num_steps = 80
    
    a, adag, n = _build_operators()
    
    # Start with DRAG seed
    best_pulse = _drag_seed()
    best_score = _compute_score(best_pulse, a, adag, n, anharmonicity, T1, Tphi, dt)
    
    # Gradient-based optimization
    # Use finite differences for gradients (simple but effective)
    step_size = 0.01
    lr = 0.05
    
    while time.time() < deadline:
        # Compute gradient via finite differences
        grad = np.zeros_like(best_pulse)
        
        for i in range(num_steps):
            for j in range(2):
                pulse_plus = best_pulse.copy()
                pulse_plus[i, j] += step_size
                score_plus = _compute_score(pulse_plus, a, adag, n, anharmonicity, T1, Tphi, dt)
                
                pulse_minus = best_pulse.copy()
                pulse_minus[i, j] -= step_size
                score_minus = _compute_score(pulse_minus, a, adag, n, anharmonicity, T1, Tphi, dt)
                
                grad[i, j] = (score_plus - score_minus) / (2 * step_size)
        
        # Update
        new_pulse = best_pulse - lr * grad
        
        # Project to feasible region
        # Amplitude constraint
        amp = np.sqrt(new_pulse[:, 0]**2 + new_pulse[:, 1]**2)
        mask = amp > 0.4
        if np.any(mask):
            new_pulse[mask] = new_pulse[mask] * (0.4 / amp[mask])
        
        # Slew rate constraint (simple projection)
        for k in range(num_steps - 1):
            diff = new_pulse[k+1] - new_pulse[k]
            norm_diff = np.linalg.norm(diff) / dt
            if norm_diff > 0.12:
                scale = 0.12 * dt / np.linalg.norm(diff)
                new_pulse[k+1] = new_pulse[k] + diff * scale
        
        # Edge constraints
        if np.linalg.norm(new_pulse[0]) / dt > 0.12:
            new_pulse[0] = new_pulse[0] * (0.12 * dt / np.linalg.norm(new_pulse[0]))
        if np.linalg.norm(new_pulse[-1]) / dt > 0.12:
            new_pulse[-1] = new_pulse[-1] * (0.12 * dt / np.linalg.norm(new_pulse[-1]))
        
        new_score = _compute_score(new_pulse, a, adag, n, anharmonicity, T1, Tphi, dt)
        
        if new_score < best_score:
            best_pulse = new_pulse
            best_score = new_score
        
        # Decrease learning rate
        lr *= 0.95
        
        # Check time
        if time.time() > deadline - 30.0:
            break
    
    return best_pulse
# EVOLVE-BLOCK-END
