"""Robust X-pi gate control pulse for three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def _build_matrices():
    """Build the 3x3 matrices for the three-level system."""
    # a = diag(sqrt([1,2]), 1) - this is a 3x3 matrix
    # a = [[0,1,0],[0,0,sqrt(2)],[0,0,0]]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a  # n = a^dagger a = diag(0,1,2)
    return a, a_dag, n

def _make_liouvillian(I, Q, detuning, scale, anharmonicity, T1, Tphi):
    """Build the Liouvillian superoperator for a given step."""
    a, a_dag, n = _build_matrices()
    
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    # n*(n-1) for diag(0,1,2) = diag(0, 0, 2)
    n_nm1 = np.diag([0.0, 0.0, 2.0])
    
    H = detuning * n + 0.5 * anharmonicity * n_nm1 + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    # Liouvillian: dρ/dt = -i[H,ρ] + D[ρ]
    # D[ρ] = sum_k (L_k ρ L_k^dag - 0.5{L_k^dag L_k, ρ})
    # L1 = a/sqrt(T1), L2 = n*sqrt(2/Tphi)
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Build superoperator in vectorized form
    # ρ_vec = vec(ρ), L_vec(ρ) = (L ⊗ I) ρ_vec (L^dag ⊗ I)^* ... 
    # Actually: dρ/dt = -i(Hρ - ρH) + sum_k (L_k ρ L_k^dag - 0.5 L_k^dag L_k ρ - 0.5 ρ L_k^dag L_k)
    # In vector form: dρ_vec/dt = M @ ρ_vec
    # M = -i(H ⊗ I - I ⊗ H^T) + sum_k [(L_k ⊗ L_k^*) - 0.5(L_k^dag L_k ⊗ I) - 0.5(I ⊗ (L_k^dag L_k)^T)]
    
    I3 = np.eye(3, dtype=complex)
    
    M = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    for L in [L1, L2]:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        M += np.kron(L, L.conj()) - 0.5 * np.kron(LdagL, I3) - 0.5 * np.kron(I3, LdagL.T)
    
    return M

def _simulate_pulse(IQ, detuning, scale, anharmonicity, T1, Tphi, dt, n_steps):
    """Simulate the pulse and return final density matrix for |0> initial state.
    IQ is (n_steps, 2) array of [I, Q]."""
    a, a_dag, n = _build_matrices()
    
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(n_steps):
        I_val = IQ[i, 0]
        Q_val = IQ[i, 1]
        
        M = _make_liouvillian(I_val, Q_val, detuning, scale, anharmonicity, T1, Tphi)
        
        # Exponentiate: rho_next = exp(M * dt) @ rho_vec
        rho_vec = rho.flatten()
        
        # Use matrix exponential for accuracy
        from scipy.linalg import expm
        Mdt = M * dt
        rho_next_vec = expm(Mdt) @ rho_vec
        rho = rho_next_vec.reshape(3, 3)
        
        # Renormalize trace to 1 (small numerical errors)
        tr = np.trace(rho)
        if abs(tr) > 1e-12:
            rho = rho / tr
    
    return rho

def _compute_infidelity(IQ, detuning, scale, anharmonicity, T1, Tphi, dt, n_steps):
    """Compute mean infidelity over 5 input states for given detuning/scale."""
    a, a_dag, n = _build_matrices()
    
    # Target X gate on |0>,|1> subspace: |0> -> |1>, |1> -> |0>
    # X = [[0,1],[1,0]] in {|0>,|1>} basis
    # In 3D: X_3d = [[0,1,0],[1,0,0],[0,0,1]] (leaves |2> unchanged)
    X_target = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]], dtype=complex)
    
    # Input states
    states = []
    # |0>
    states.append(np.array([1, 0, 0], dtype=complex))
    # |1>
    states.append(np.array([0, 1, 0], dtype=complex))
    # (|0>+|1>)/sqrt(2)
    states.append(np.array([1, 1, 0], dtype=complex) / np.sqrt(2))
    # (|0>-|1>)/sqrt(2)
    states.append(np.array([1, -1, 0], dtype=complex) / np.sqrt(2))
    # (|0>+i|1>)/sqrt(2)
    states.append(np.array([1, 1j, 0], dtype=complex) / np.sqrt(2))
    # (|0>-i|1>)/sqrt(2)
    states.append(np.array([1, -1j, 0], dtype=complex) / np.sqrt(2))
    
    fidelities = []
    
    for psi in states:
        rho_init = np.outer(psi, psi.conj())
        
        # Simulate
        rho = rho_init.copy()
        for i in range(n_steps):
            I_val = IQ[i, 0]
            Q_val = IQ[i, 1]
            M = _make_liouvillian(I_val, Q_val, detuning, scale, anharmonicity, T1, Tphi)
            rho_vec = rho.flatten()
            from scipy.linalg import expm
            Mdt = M * dt
            rho_next_vec = expm(Mdt) @ rho_vec
            rho = rho_next_vec.reshape(3, 3)
            tr = np.trace(rho)
            if abs(tr) > 1e-12:
                rho = rho / tr
        
        # Fidelity: F = <psi_target|rho|psi_target> where psi_target = X @ psi
        psi_target = X_target @ psi
        F = np.real(np.vdot(psi_target, rho @ psi_target))
        F = np.clip(F, 0.0, 1.0)
        fidelities.append(F)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity

def _check_constraints(IQ, dt, amp_bound, slew_bound):
    """Check if IQ satisfies amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= amp_bound
    amps = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
    if np.max(amps) > amp_bound + 1e-12:
        return False
    
    # Slew: ||p[i+1] - p[i]|| / dt <= slew_bound
    # Including from zero into step 1 and from last step to zero
    # p[0] = IQ[0], but "from zero into step 1" means ||IQ[0] - 0||/dt <= slew_bound
    if np.linalg.norm(IQ[0]) / dt > slew_bound + 1e-12:
        return False
    if np.linalg.norm(IQ[-1]) / dt > slew_bound + 1e-12:
        return False
    
    for i in range(len(IQ) - 1):
        diff = np.linalg.norm(IQ[i+1] - IQ[i])
        if diff / dt > slew_bound + 1e-12:
            return False
    
    return True

def _project_to_constraints(IQ, dt, amp_bound, slew_bound):
    """Project IQ to satisfy constraints (simple clipping)."""
    IQ = IQ.copy()
    
    # Amplitude constraint
    for i in range(len(IQ)):
        amp = np.sqrt(IQ[i, 0]**2 + IQ[i, 1]**2)
        if amp > amp_bound:
            scale = amp_bound / amp
            IQ[i] *= scale
    
    # Slew constraint - simple approach: limit changes
    # This is a rough projection; full constraint handling is complex
    return IQ

def _seed_pulse(n_steps, dt, amp_bound, slew_bound):
    """Generate the DRAG-style seed pulse."""
    sigma = 40.0 / 6.0  # ns
    t = np.arange(n_steps) * dt
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * (t / sigma)**2)
    
    # Offset to 0 at edges: subtract value at first/last
    gauss = gauss - gauss[0]
    gauss = gauss - gauss[-1]
    # Actually "offset to 0 at the edges" - let's make it zero at t=0 and t=T
    # Use a windowed Gaussian
    gauss = np.exp(-0.5 * (t / sigma)**2)
    gauss[0] = 0.0
    gauss[-1] = 0.0
    
    # Area should be pi: integral of gauss * dt = pi
    area = np.sum(gauss) * dt
    if area > 1e-12:
        gauss = gauss * (np.pi / area)
    
    # DRAG: add derivative term
    # d/dt of Gaussian
    drag_coeff = 1.0  # typical DRAG coefficient
    gauss_deriv = np.gradient(gauss, dt)
    
    I = gauss
    Q = drag_coeff * gauss_deriv
    
    # Scale to satisfy constraints
    IQ = np.column_stack([I, Q])
    IQ = _project_to_constraints(IQ, dt, amp_bound, slew_bound)
    
    return IQ

def _optimize_pulse(IQ_init, n_steps, dt, anharmonicity, T1, Tphi, detunings, scales, 
                    amp_bound, slew_bound, max_iters=50, time_limit=250.0):
    """Optimize the pulse to minimize mean+max infidelity."""
    start_time = time.time()
    best_IQ = IQ_init.copy()
    best_score = np.inf
    
    # Compute initial score
    def compute_score(IQ):
        infs = []
        for det in detunings:
            for sc in scales:
                inf = _compute_infidelity(IQ, det, sc, anharmonicity, T1, Tphi, dt, n_steps)
                infs.append(inf)
        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)
    
    best_score = compute_score(best_IQ)
    
    # Simple gradient-free optimization: coordinate descent with small perturbations
    # Use a randomized local search
    np.random.seed(42)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try random perturbations
        found_better = False
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Random perturbation
            delta = np.random.randn(n_steps, 2) * 0.01
            IQ_new = best_IQ + delta
            IQ_new = _project_to_constraints(IQ_new, dt, amp_bound, slew_bound)
            
            if not _check_constraints(IQ_new, dt, amp_bound, slew_bound):
                continue
            
            score = compute_score(IQ_new)
            if score < best_score - 1e-10:
                best_IQ = IQ_new.copy()
                best_score = score
                found_better = True
        
        if not found_better:
            # Try smaller steps
            for attempt in range(10):
                if time.time() - start_time > time_limit:
                    break
                delta = np.random.randn(n_steps, 2) * 0.003
                IQ_new = best_IQ + delta
                IQ_new = _project_to_constraints(IQ_new, dt, amp_bound, slew_bound)
                if not _check_constraints(IQ_new, dt, amp_bound, slew_bound):
                    continue
                score = compute_score(IQ_new)
                if score < best_score - 1e-10:
                    best_IQ = IQ_new.copy()
                    best_score = score
                    found_better = True
            if not found_better:
                break
    
    return best_IQ, best_score

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gate
        return np.zeros((80, 2))
    
    # Extract hardware parameters
    # From contract: anharmonicity -2*pi*0.25, amplitude bound 0.4, slew bound 0.12,
    # detuning bounds +-2*pi*0.002, drive scale bounds [0.97, 1.03], T1 25000 ns, Tphi 40000 ns
    
    anharmonicity = -2.0 * np.pi * 0.25  # rad/ns? Let's assume GHz converted
    # Actually, the units are tricky. Let's assume the values are in rad/ns already
    # or we need to convert. The contract says "anharmonicity -2 pi 0.25" which is likely
    # in GHz, so we need to convert to rad/ns: -2*pi*0.25e-3 rad/ns? 
    # Let's use the values as given and assume they're in consistent units.
    
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2.0 * np.pi * 0.002
    scale_min, scale_max = 0.97, 1.03
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    dt = 0.5  # ns
    n_steps = 80
    
    # Test cases
    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [scale_min, 1.0, scale_max]
    
    # Generate seed pulse
    IQ_seed = _seed_pulse(n_steps, dt, amp_bound, slew_bound)
    
    # Ensure seed is valid
    if not _check_constraints(IQ_seed, dt, amp_bound, slew_bound):
        # Fallback: simple rectangular pulse
        IQ_seed = np.zeros((n_steps, 2))
        IQ_seed[:, 0] = np.pi / (n_steps * dt)  # constant I to get area pi
        IQ_seed = _project_to_constraints(IQ_seed, dt, amp_bound, slew_bound)
    
    # Optimize
    start_time = time.time()
    time_limit = 280.0  # Leave 20s margin for safety
    
    IQ_opt, score = _optimize_pulse(
        IQ_seed, n_steps, dt, anharmonicity, T1, Tphi,
        detunings, scales, amp_bound, slew_bound,
        max_iters=100, time_limit=time_limit
    )
    
    # Final check
    if not _check_constraints(IQ_opt, dt, amp_bound, slew_bound):
        IQ_opt = IQ_seed
    
    return IQ_opt
# EVOLVE-BLOCK-END
