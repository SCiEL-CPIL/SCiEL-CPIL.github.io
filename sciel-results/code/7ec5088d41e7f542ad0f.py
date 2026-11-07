"""Robust control pulses for a three-level transmon - DRAG + optimization."""
import numpy as np
from scipy.optimize import minimize
import time

# Constants from the task
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
T1 = 25000.0  # ns
T2PHI = 40000.0  # ns

# 3x3 ladder operators
def _build_operators():
    """Build a, a^dagger, n for 3-level system."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

A, ADAG, N_OP = _build_operators()

def _liouvillian(I, Q, detuning, scale):
    """Build the Liouvillian superoperator for given pulse parameters."""
    # Hamiltonian
    H = (detuning * N_OP + 
         0.5 * ANHARMONICITY * N_OP @ (N_OP - np.eye(3)) +
         0.5 * scale * (I * (A + ADAG) + 1j * Q * (A - ADAG)))
    
    # Liouvillian: L(rho) = -i[H, rho] + dissipation
    # For density matrix evolution, we use the superoperator form
    # L(rho) = -i*H@rho + i*rho@H + sum_k (L_k@rho@L_k^dag - 0.5*{L_k^dag L_k, rho})
    
    # Dissipation terms
    L1 = A / np.sqrt(T1)
    L2 = N_OP * np.sqrt(2.0 / T2PHI)
    
    # Build superoperator
    # Vectorize: rho -> rho.flatten()
    # L_super @ rho_flat = vec(-i[H,rho] + dissipation)
    
    # Coherent part: -i(H rho - rho H)
    # vec(H rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    
    I3 = np.eye(3, dtype=complex)
    L_coh = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipation: sum_k (L_k rho L_k^dag - 0.5 L_k^dag L_k rho - 0.5 rho L_k^dag L_k)
    L_diss = np.zeros((9, 9), dtype=complex)
    for L in [L1, L2]:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        # vec(L rho Ldag) = (Ldag^T ⊗ L) vec(rho)
        # vec(LdagL rho) = (I ⊗ LdagL) vec(rho)
        # vec(rho LdagL) = (LdagL^T ⊗ I) vec(rho)
        L_diss += np.kron(Ldag.T, L) - 0.5 * np.kron(I3, LdagL) - 0.5 * np.kron(LdagL.T, I3)
    
    L_super = L_coh + L_diss
    return L_super

def _evolve_pulse(I_arr, Q_arr, detuning, scale):
    """Evolve the state through the pulse. Returns final density matrix (3x3)."""
    # Initial state is |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        L = _liouvillian(I_arr[i], Q_arr[i], detuning, scale)
        # Exponentiate: rho(t+dt) = exp(L*dt) @ rho(t)
        # Use matrix exponential
        from scipy.linalg import expm
        U = expm(L * DT)
        rho_flat = rho.flatten()
        rho_new_flat = U @ rho_flat
        rho = rho_new_flat.reshape((3, 3))
    
    return rho

def _fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure target state."""
    # Fidelity = <psi| rho |psi>
    return np.real(np.vdot(target_state, rho @ target_state))

def _compute_score(I_arr, Q_arr):
    """Compute the scoring function (lower is better)."""
    # 9 cases: detuning in (-2*pi*0.002, 0, 2*pi*0.002), scale in (0.97, 1, 1.03)
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_ip = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_im = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_ip, psi_im]
    
    # X gate maps: |0>->|1>, |1>->|0>, superpositions accordingly
    # After X gate:
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    
    # Target states after X gate:
    target_after_x = [psi1, psi0, psi_plus, -psi_minus, 1j*psi_im, -1j*psi_ip]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
        # Evolve each initial state
            fidelities = []
            for init_state in target_states:
                rho = _evolve_pulse(I_arr, Q_arr, det, sc)
                # Wait, I need to evolve from init_state, not from |0>
                # Let me re-think: we need to compute fidelity of evolved state to target
                pass
            
            # Actually, let me restructure: for each initial state, evolve and check fidelity
            case_fidelities = []
            for init_state, target_state in zip(target_states, target_after_x):
                # Evolve init_state through the pulse
                rho = np.outer(init_state, init_state.conj())
                for i in range(N_STEPS):
                    L = _liouvillian(I_arr[i], Q_arr[i], det, sc)
                    from scipy.linalg import expm
                    U = expm(L * DT)
                    rho_flat = rho.flatten()
                    rho_new_flat = U @ rho_flat
                    rho = rho_new_flat.reshape((3, 3))
                
                fid = _fidelity(rho, target_state)
                case_fidelities.append(fid)
            
            mean_fid = np.mean(case_fidelities)
            infidelities.append(1.0 - mean_fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _is_valid(I_arr, Q_arr):
    """Check amplitude and slew rate constraints."""
    # Amplitude bound
    amps = np.sqrt(I_arr**2 + Q_arr**2)
    if np.any(amps > AMP_BOUND + 1e-9):
        return False
    
    # Slew rate: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # Include from zero to step 1 and last step to zero
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = np.array([I_arr[i], Q_arr[i]])
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    
    # From last step to zero
    if np.linalg.norm(prev - np.array([0.0, 0.0])) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True

def _seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT + DT/2  # step centers
    t_center = T_TOTAL / 2.0
    sigma = 40.0 / 6.0  # 6.67 ns
    
    # Gaussian I component with area pi
    I = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    # Normalize area to pi
    area = np.sum(I) * DT
    I = I * (np.pi / area)
    
    # DRAG Q component: proportional to derivative of I
    # Q = -beta * dI/dt
    dI_dt = np.gradient(I, DT)
    beta = sigma**2 * np.max(I) / 2.0  # typical DRAG beta
    Q = -beta * dI_dt
    
    # Clip to bounds
    for i in range(N_STEPS):
        amp = np.sqrt(I[i]**2 + Q[i]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            I[i] *= scale
            Q[i] *= scale
    
    return I, Q

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate supported")
    
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    # Generate seed pulse
    I_seed, Q_seed = _seed_pulse()
    
    # Check seed validity
    if not _is_valid(I_seed, Q_seed):
        # Scale down to satisfy constraints
        for i in range(N_STEPS):
            amp = np.sqrt(I_seed[i]**2 + Q_seed[i]**2)
            if amp > AMP_BOUND:
                scale = AMP_BOUND / amp
                I_seed[i] *= scale
                Q_seed[i] *= scale
        # Check slew rate
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            curr = np.array([I_seed[i], Q_seed[i]])
            delta = np.linalg.norm(curr - prev)
            if delta / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / delta
                I_seed[i] = prev[0] + scale * (I_seed[i] - prev[0])
                Q_seed[i] = prev[1] + scale * (Q_seed[i] - prev[1])
            prev = np.array([I_seed[i], Q_seed[i]])
        # Check last step to zero
        delta = np.linalg.norm(prev)
        if delta / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / delta
            I_seed[-1] *= scale
            Q_seed[-1] *= scale
    
    best_I = I_seed.copy()
    best_Q = Q_seed.copy()
    best_score = _compute_score(best_I, best_Q)
    
    # Parameterize the pulse for optimization
    # Use a small number of free parameters to keep optimization tractable
    # We'll optimize the DRAG beta and a few shape parameters
    
    def _params_to_pulse(params):
        """Convert optimization parameters to I, Q arrays."""
        # params[0]: beta (DRAG coefficient)
        # params[1]: gaussian sigma
        # params[2]: gaussian center offset
        # params[3]: amplitude scale
        beta = params[0]
        sigma = params[1]
        center_offset = params[2]
        amp_scale = params[3]
        
        t = np.arange(N_STEPS) * DT + DT/2
        t_center = T_TOTAL / 2.0 + center_offset
        
        I = np.exp(-0.5 * ((t - t_center) / sigma)**2) * amp_scale
        area = np.sum(I) * DT
        if area > 1e-12:
            I = I * (np.pi / area)
        
        dI_dt = np.gradient(I, DT)
        Q = -beta * dI_dt
        
        # Enforce constraints
        for i in range(N_STEPS):
            amp = np.sqrt(I[i]**2 + Q[i]**2)
            if amp > AMP_BOUND:
                scale = AMP_BOUND / amp
                I[i] *= scale
                Q[i] *= scale
        
        # Enforce slew rate
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            curr = np.array([I[i], Q[i]])
            delta = np.linalg.norm(curr - prev)
            if delta / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / delta
                I[i] = prev[0] + scale * (I[i] - prev[0])
                Q[i] = prev[1] + scale * (Q[i] - prev[1])
            prev = np.array([I[i], Q[i]])
        
        # Check last step to zero
        delta = np.linalg.norm(prev)
        if delta / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / delta
            I[-1] *= scale
            Q[-1] *= scale
        
        return I, Q
    
    def _objective(params):
        """Optimization objective."""
        I, Q = _params_to_pulse(params)
        if not _is_valid(I, Q):
            return 1e6
        return _compute_score(I, Q)
    
    # Initial parameters from seed
    sigma_init = 40.0 / 6.0
    beta_init = sigma_init**2 * np.max(I_seed) / 2.0
    x0 = np.array([beta_init, sigma_init, 0.0, 1.0])
    
    # Nelder-Mead optimization
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        if result.fun < best_score:
            I_opt, Q_opt = _params_to_pulse(result.x)
            if _is_valid(I_opt, Q_opt):
                best_I = I_opt
                best_Q = Q_opt
                best_score = result.fun
    except Exception:
        pass
    
    # Try multiple restarts
    restarts = [
        [beta_init * 0.5, sigma_init, 0.0, 1.0],
        [beta_init * 1.5, sigma_init, 0.0, 1.0],
        [beta_init, sigma_init * 0.8, 0.0, 1.0],
        [beta_init, sigma_init * 1.2, 0.0, 1.0],
        [beta_init, sigma_init, 0.5, 1.0],
        [beta_init, sigma_init, -0.5, 1.0],
    ]
    
    for x0_restart in restarts:
        if time.time() > deadline:
            break
        try:
            result = minimize(_objective, x0_restart, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
            if result.fun < best_score:
                I_opt, Q_opt = _params_to_pulse(result.x)
                if _is_valid(I_opt, Q_opt):
                    best_I = I_opt
                    best_Q = Q_opt
                    best_score = result.fun
        except Exception:
            pass
    
    # Final validity check
    if not _is_valid(best_I, best_Q):
        # Fall back to seed
        best_I, best_Q = _seed_pulse()
    
    return np.column_stack([best_I, best_Q])
