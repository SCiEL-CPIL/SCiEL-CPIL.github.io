"""Robust X-pi gate control pulse using low-dimensional parameterization and optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution
import time

def _build_hamiltonian(I_val, Q_val, detuning, scale, anharmonicity):
    """
    Build the Hamiltonian for the 3-level transmon.
    H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    """
    # a matrix (3x3)
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    
    # n = a^dag * a
    n = a.conj().T @ a
    
    # Anharmonicity term: 0.5 * anharmonicity * n * (n-1)
    # n*(n-1) is diagonal with entries [0, 0, 2] for levels 0,1,2
    n_n_minus_1 = n @ (n - np.eye(3))
    
    H = detuning * n + 0.5 * anharmonicity * n_n_minus_1
    H += 0.5 * scale * (I_val * (a + a.conj().T) + 1j * Q_val * (a - a.conj().T))
    
    return H

def _build_liouvillian(H, T1, Tphi):
    """
    Build the Liouvillian superoperator for Lindblad evolution.
    L = -i/ħ [H, ρ] + sum_k (L_k ρ L_k^† - 1/2 {L_k^† L_k, ρ})
    We work in vectorized form: vec(ρ) -> (L_vec) vec(ρ)
    """
    dim = 3
    # Commutator part: -i * (H ⊗ I - I ⊗ H^T) in vec convention
    # Actually, for vec(AρB) = (B^T ⊗ A) vec(ρ)
    # [H, ρ] = Hρ - ρH -> vec(Hρ) - vec(ρH) = (I ⊗ H) vec(ρ) - (H^T ⊗ I) vec(ρ)
    # So -i[H,ρ] -> -i * ((I ⊗ H) - (H^T ⊗ I)) vec(ρ)
    # But wait, standard convention: vec(AρB) = (B^T ⊗ A) vec(ρ)
    # Hρ = H ρ I -> (I^T ⊗ H) vec(ρ) = (I ⊗ H) vec(ρ)
    # ρH = I ρ H -> (H^T ⊗ I) vec(ρ)
    # So [H,ρ] -> (I ⊗ H - H^T ⊗ I) vec(ρ)
    # -i[H,ρ] -> -i(I ⊗ H - H^T ⊗ I)
    
    I_mat = np.eye(dim, dtype=complex)
    
    commutator = -1j * (np.kron(I_mat, H) - np.kron(H.T, I_mat))
    
    # Lindblad terms
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    n = a.conj().T @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    L_vec = commutator.copy()
    
    for L_op in [L1, L2]:
        # L_op ρ L_op^† -> (L_op^†)^T ⊗ L_op = conj(L_op) ⊗ L_op
        term1 = np.kron(L_op.conj(), L_op)
        # 1/2 {L_op^† L_op, ρ} = 1/2 (L_op^† L_op ρ + ρ L_op^† L_op)
        # -> 1/2 ((I ⊗ L_op^† L_op) + ((L_op^† L_op)^T ⊗ I))
        LdagL = L_op.conj().T @ L_op
        term2 = 0.5 * (np.kron(I_mat, LdagL) + np.kron(LdagL.T, I_mat))
        L_vec += term1 - term2
    
    return L_vec

def _evolve_pulse(pulse_IQ, detuning, scale, anharmonicity, T1, Tphi, dt):
    """
    Evolve the state through the pulse.
    pulse_IQ: (N, 2) array of [I, Q] amplitudes
    Returns the final process matrix (9,9) or we can compute fidelity directly.
    """
    N = len(pulse_IQ)
    dim = 3
    dim2 = dim * dim
    
    # Process matrix: U_total maps initial density matrix to final
    # We track the evolution of the identity operator basis to build the process matrix
    # Or simpler: evolve each basis state and superposition
    
    # For fidelity calculation, we need the process matrix E
    # E_{kl} = <k| U ρ_l U† > where ρ_l = |l><l| initially? No, that's not right.
    
    # Actually, fidelity for pure state |ψ> is |<ψ|U|ψ>|^2 for gate fidelity if we're comparing to ideal gate.
    # But we need to compare to X gate.
    
    # Let's compute the process matrix in the Liouville space.
    # The process matrix χ or E can be built by evolving the basis operators.
    
    # Easier: evolve the density matrix for each of the 4 test states, then compute fidelity.
    
    # But we need to do this for 9 detuning/scale cases, so let's be efficient.
    
    # Start with identity process: P = I (9x9)
    # Actually, let's just evolve the 4 test states and compute fidelities directly.
    
    # Test states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # But the task says: |0>, |1>, (|0> ± |1>)/sqrt(2), (|0> ± i|1>)/sqrt(2) - that's 6 states
    # Wait, "mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # That's 1 + 1 + 2 + 2 = 6 states
    
    # For each test state, evolve it and compute fidelity with X|ψ>
    
    # X gate: |0> -> |1>, |1> -> |0>
    
    test_states = []
    test_states.append(np.array([1, 0, 0], dtype=complex))  # |0>
    test_states.append(np.array([0, 1, 0], dtype=complex))  # |1>
    test_states.append(np.array([1, 1, 0], dtype=complex) / np.sqrt(2))  # (|0>+|1>)/sqrt(2)
    test_states.append(np.array([1, -1, 0], dtype=complex) / np.sqrt(2))  # (|0>-|1>)/sqrt(2)
    test_states.append(np.array([1, 1j, 0], dtype=complex) / np.sqrt(2))  # (|0>+i|1>)/sqrt(2)
    test_states.append(np.array([1, -1j, 0], dtype=complex) / np.sqrt(2))  # (|0>-i|1>)/sqrt(2)
    
    # X gate action on these states
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    fidelities = []
    
    for psi0 in test_states:
        # Evolve psi0
        psi = psi0.copy()
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            H = _build_hamiltonian(I_val, Q_val, detuning, scale, anharmonicity)
            
            # For pure state evolution without decoherence (or with), we can use the Liouvillian
            # But for pure states, we can just use the Schrödinger equation if we ignore decoherence
            # However, the task specifies Lindblad operators, so we should include them.
            # But for fidelity calculation, we need the density matrix.
            
            # Let's use the density matrix approach
            pass
        
        # This is getting complex. Let me use the Liouvillian approach properly.
        
    # Let me redo this with density matrices
    fidelities = []
    
    for psi0 in test_states:
        rho = np.outer(psi0, psi0.conj())
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            H = _build_hamiltonian(I_val, Q_val, detuning, scale, anharmonicity)
            L_vec = _build_liouvillian(H, T1, Tphi)
            
            # Evolve: rho_new = expm(L_vec * dt) vec(rho)
            rho_vec = rho.flatten()
            rho_vec_new = expm(L_vec * dt) @ rho_vec
            rho = rho_vec_new.reshape(dim, dim)
        
        # Compute fidelity with X|psi0>
        psi_target = X_gate @ psi0
        # Fidelity = <psi_target| rho |psi_target>
        fidelity = np.real(np.vdot(psi_target, rho @ psi_target))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def _make_pulse(params):
    """
    Generate pulse from parameters.
    params: [amplitude, sigma, drag_amplitude, drag_sigma, phase]
    """
    dt = 0.5
    n_steps = 80
    t = np.arange(n_steps) * dt
    
    amplitude = params[0]
    sigma = params[1]
    drag_amplitude = params[2]
    drag_sigma = params[3]
    phase = params[4]
    
    t_center = 20.0
    
    # Gaussian envelope
    gaussian = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # DRAG term: derivative of Gaussian
    drag_term = -((t - t_center) / sigma ** 2) * gaussian
    
    # Normalize to have area pi
    area_gaussian = np.sum(gaussian) * dt
    area_drag = np.sum(drag_term) * dt
    
    # I component: amplitude * gaussian / area_gaussian * pi
    # Q component: drag_amplitude * drag_term / area_drag * pi (if area_drag != 0)
    
    I = amplitude * gaussian / area_gaussian * np.pi
    if abs(area_drag) > 1e-10:
        Q = drag_amplitude * drag_term / area_drag * np.pi
    else:
        Q = np.zeros(n_steps)
    
    # Apply phase
    pulse_IQ = np.column_stack([I * np.cos(phase), I * np.sin(phase) + Q])
    
    # Enforce constraints
    # Amplitude bound: sqrt(I^2 + Q^2) <= 0.4
    for i in range(n_steps):
        r = np.sqrt(pulse_IQ[i, 0]**2 + pulse_IQ[i, 1]**2)
        if r > 0.4:
            pulse_IQ[i] *= 0.4 / r
    
    # Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # Also from zero to step 1 and last step to zero
    prev = np.zeros(2)
    for i in range(n_steps):
        diff = pulse_IQ[i] - prev
        norm_diff = np.linalg.norm(diff)
        if norm_diff / dt > 0.12:
            scale_factor = 0.12 * dt / norm_diff
            pulse_IQ[i] = prev + diff * scale_factor
        prev = pulse_IQ[i]
    
    # Check last step to zero
    diff = np.zeros(2) - prev
    norm_diff = np.linalg.norm(diff)
    if norm_diff / dt > 0.12:
        scale_factor = 0.12 * dt / norm_diff
        pulse_IQ[-1] = prev + (np.zeros(2) - prev) * scale_factor
    
    return pulse_IQ

def _objective(params):
    """
    Objective function: mean + max infidelity over 9 cases.
    """
    pulse_IQ = _make_pulse(params)
    
    # Check validity
    dt = 0.5
    for i in range(len(pulse_IQ)):
        if np.sqrt(pulse_IQ[i, 0]**2 + pulse_IQ[i, 1]**2) > 0.4 + 1e-9:
            return 1e10
    prev = np.zeros(2)
    for i in range(len(pulse_IQ)):
        diff = pulse_IQ[i] - prev
        if np.linalg.norm(diff) / dt > 0.12 + 1e-9:
            return 1e10
        prev = pulse_IQ[i]
    diff = np.zeros(2) - prev
    if np.linalg.norm(diff) / dt > 0.12 + 1e-9:
        return 1e10
    
    anharmonicity = -2 * np.pi * 0.25e-3  # rad/ns
    T1 = 25000.0
    Tphi = 40000.0
    
    detunings = [-2 * np.pi * 0.002e-3, 0.0, 2 * np.pi * 0.002e-3]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            fidelity = _evolve_pulse(pulse_IQ, det, sc, anharmonicity, T1, Tphi, dt)
            infidelity = 1.0 - fidelity
            infidelities.append(infidelity)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    
    return score

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    # Initial guess based on DRAG pulse
    sigma = 40.0 / 6.0
    initial_params = [0.08, sigma, 0.02, sigma * 0.5, 0.0]
    
    # Define bounds for parameters
    bounds = [
        (0.01, 0.4),      # amplitude
        (2.0, 10.0),      # sigma
        (-0.1, 0.1),      # drag_amplitude
        (1.0, 5.0),       # drag_sigma
        (-0.5, 0.5)       # phase
    ]
    
    # Use differential evolution
    start_time = time.time()
    timeout = 300.0
    
    # Run optimization with time limit
    def _time_limited_objective(params):
        if time.time() - start_time > timeout - 30:  # 30s safety margin
            return 1e10
        return _objective(params)
    
    result = differential_evolution(
        _time_limited_objective,
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-4,
        seed=42,
        polish=False
    )
    
    best_params = result.x
    best_pulse = _make_pulse(best_params)
    
    # Final validity check
    dt = 0.5
    valid = True
    for i in range(len(best_pulse)):
        if np.sqrt(best_pulse[i, 0]**2 + best_pulse[i, 1]**2) > 0.4 + 1e-9:
            valid = False
            break
    if valid:
        prev = np.zeros(2)
        for i in range(len(best_pulse)):
            diff = best_pulse[i] - prev
            if np.linalg.norm(diff) / dt > 0.12 + 1e-9:
                valid = False
                break
            prev = best_pulse[i]
        if valid:
            diff = np.zeros(2) - prev
            if np.linalg.norm(diff) / dt > 0.12 + 1e-9:
                valid = False
    
    if not valid:
        # Fallback to a simple Gaussian pulse
        t = np.arange(80) * dt
        t_center = 20.0
        gaussian = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        area = np.sum(gaussian) * dt
        I = 0.08 * gaussian / area * np.pi
        Q = np.zeros(80)
        best_pulse = np.column_stack([I, Q])
    
    return best_pulse
# EVOLVE-BLOCK-END
