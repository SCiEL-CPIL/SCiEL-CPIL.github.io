"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import linalg

def pulse(target_gate, hardware_model):
    """
    Generate a robust X_pi pulse for a 3-level transmon.

    Returns (80, 2) array of [I, Q] amplitudes in rad/ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    dt = 0.5  # ns
    N = 80
    T = N * dt  # 40 ns

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    # Time guard
    start_time = time.time()
    deadline = start_time + 280.0  # 280s safety margin

    # Time grid for constant steps
    t = np.linspace(0, T, N, endpoint=False) + dt / 2.0

    # Base Gaussian envelope (DRAG-style seed)
    sigma = T / 6.0
    center = T / 2.0

    def gaussian_env(center_shift=0.0):
        g = np.exp(-0.5 * ((t - center - center_shift) / sigma) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center - center_shift) / sigma) ** 2)
        g = g - edge_val
        g = np.maximum(g, 0.0)
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)
        return g

    g = gaussian_env()
    g_deriv = np.gradient(g, dt)

    def make_pulse(sigma2, amp_scale=1.0, center_shift=0.0):
        gg = gaussian_env(center_shift)
        gg_deriv = np.gradient(gg, dt)
        I = amp_scale * gg.copy()
        Q = amp_scale * sigma2 * gg_deriv.copy()

        # Enforce amplitude bound
        amp = np.sqrt(I ** 2 + Q ** 2)
        max_amp = np.max(amp) if amp.size else 0.0
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I *= factor
            Q *= factor

        # Enforce slew bound
        def check_slew(I, Q):
            slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
            slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
            m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
            return m <= slew_bound

        if not check_slew(I, Q):
            for _ in range(120):
                I *= 0.95
                Q *= 0.95
                if check_slew(I, Q):
                    break

        return I, Q

    # Hamiltonian / Liouvillian setup
    # a = diag(sqrt([1,2]), 1), n = a^dag a
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a
    n2 = n @ n

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Pre-compute constant parts of Liouvillian (Lindblad terms)
    # L(rho) = D1(rho) + D2(rho)
    # Vectorized form: vec(D(rho)) = (L^T \otimes L - 0.5 I \otimes L L^T - 0.5 L L^T \otimes I) vec(rho)
    # Note: The code uses column-stacking convention for vec.
    # Let's stick to the explicit construction in the step to avoid convention errors, 
    # but we can pre-compute the Lindblad part of the Liouvillian matrix.
    
    eye3 = np.eye(3)
    L1c = L1.conj().T
    L2c = L2.conj().T
    
    # Lindblad contribution to Liouvillian (9x9)
    Lindblad_part = np.zeros((9, 9), dtype=complex)
    for L, Lc in [(L1, L1c), (L2, L2c)]:
        Lindblad_part += (np.kron(Lc, L) - 0.5 * np.kron(L @ Lc, eye3)
                          - 0.5 * np.kron(eye3, L @ Lc))
    
    def hamiltonian_part(detuning, scale, I, Q):
        # H = detuning n + 0.5 anharmonicity n (n - 1) + 0.5 scale (I (a + a_dag) + i Q (a - a_dag))
        # The term -i[H, rho] in vectorized form is:
        # -i (I \otimes H - H^T \otimes I) if using row-major? 
        # Standard vectorization vec(A X B) = (B^T \otimes A) vec(X).
        # [H, rho] = H rho - rho H.
        # vec(H rho) = (I \otimes H) vec(rho).
        # vec(rho H) = (H^T \otimes I) vec(rho).
        # So vec([H, rho]) = (I \otimes H - H^T \otimes I) vec(rho).
        # The Liouvillian term is -i * this.
        H = (detuning * n
             + 0.5 * anharmonicity * (n @ n - n)
             + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
        Hc = H.conj().T
        # Note: In the previous code, it used Hc in the kron. Let's verify.
        # If vec is column-stacking, vec(A X B) = (B^T \otimes A) vec(X).
        # vec(H rho) = (I^T \otimes H) vec(rho) = (I \otimes H) vec(rho).
        # vec(rho H) = (H^T \otimes I) vec(rho).
        # So [H, rho] term is (I \otimes H - H^T \otimes I).
        # The previous code used: -1j * (np.kron(np.eye(3), H) - np.kron(Hc, np.eye(3)))
        # Hc is H^T if H is real? No, H is complex. Hc is H^\dagger.
        # For the commutator [H, rho], we need H^T, not H^\dagger.
        # However, in quantum mechanics, H is Hermitian, so H^T = H^* = H^\dagger.
        # So Hc is correct for H^T.
        return -1j * (np.kron(eye3, H) - np.kron(Hc, eye3))

    # States to simulate
    states = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]
    
    # Pre-compute target projectors and initial density matrices
    # rho0 for each state
    rho0s = np.array([np.outer(st, st.conj()).reshape(9) for st in states]) # Shape (6, 9)
    
    # Target states for fidelity
    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
    target_states = [X_gate @ st for st in states]
    # Projectors P_target = |psi_target><psi_target|
    # Fidelity = Tr(rho_final P_target) = <psi_target| rho_final |psi_target>
    # We can compute this as sum_{i,j} (P_target)_{ij} (rho_final)_{ji}
    # Or simply <psi_target| rho_final |psi_target>
    
    def score_pulse(I, Q):
        # Simulate all 6 states simultaneously
        # rho_batch shape (6, 9)
        rho_batch = rho0s.copy()
        
        for k in range(N):
            H_part = hamiltonian_part(0.0, 1.0, I[k], Q[k]) # Detuning and scale are 0,1 for nominal? 
            # Wait, the score is the mean over 9 cases of (detuning, scale).
            # We must simulate for each of the 9 cases!
            # This means we cannot just batch the 6 states. We have 9 * 6 = 54 trajectories.
            # We can batch all 54 into a (54, 9) matrix.
            pass
            
        # Re-implementing score_pulse to batch all 54 trajectories
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        # Pre-compute initial density matrices for all 54 cases
        # Actually, the initial state is the same for a given state index, regardless of detuning/scale.
        # So we have 6 unique initial states.
        # We can batch them into a (6, 9) matrix, but we need to apply different Liouvillians for each of the 9 cases.
        # This means we cannot simply do one big matrix multiplication unless we construct a block-diagonal Liouvillian, which is inefficient.
        # Instead, we loop over the 9 cases, and for each case, we simulate the 6 states batched.
        
        fids = []
        for d in detunings:
            for s in scales:
                # Simulate 6 states for this (d, s)
                rho_batch = rho0s.copy() # (6, 9)
                
                for k in range(N):
                    H_part = hamiltonian_part(d, s, I[k], Q[k])
                    L_full = Lindblad_part + H_part
                    U = linalg.expm(L_full * dt)
                    # Update all 6 states: rho_{k+1} = U @ rho_k
                    # rho_batch is (6, 9). U is (9, 9).
                    # We need U @ rho for each column.
                    # rho_batch.T is (9, 6). U @ rho_batch.T is (9, 6).
                    rho_batch = (U @ rho_batch.T).T
                
                # Calculate fidelity for each of the 6 states
                case_fids = []
                for i in range(6):
                    rho_final = rho_batch[i].reshape(3, 3)
                    psi_target = target_states[i]
                    fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
                    case_fids.append(fid)
                fids.append(np.mean(case_fids))
                
        fids = np.array(fids)
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    # Search over DRAG parameter and small shifts/scales
    best_I, best_Q = None, None
    best_score = float('inf')

    sigma2_candidates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    amp_scales = [0.95, 1.0, 1.05]
    center_shifts = [-1.0, 0.0, 1.0]

    for sigma2 in sigma2_candidates:
        for amp_scale in amp_scales:
            for cs in center_shifts:
                I, Q = make_pulse(sigma2, amp_scale, cs)
                sc = score_pulse(I, Q)
                if sc < best_score:
                    best_score = sc
                    best_I, best_Q = I, Q
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if best_I is None:
        drive_amp = np.max(g)
        optimal_sigma2 = abs(anharmonicity) / (2 * drive_amp * sigma) if drive_amp > 0 else 0.2
        I, Q = make_pulse(optimal_sigma2, 1.0, 0.0)
        best_I, best_Q = I, Q

    I, Q = best_I, best_Q

    # Final validity check
    amp = np.sqrt(I ** 2 + Q ** 2)
    assert np.all(amp <= amp_bound + 1e-10), "Amplitude constraint violated"

    def check_slew(I, Q):
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
        return m <= slew_bound

    assert check_slew(I, Q), "Slew rate constraint violated"

    return np.column_stack([I, Q])

# EVOLVE-BLOCK-END
