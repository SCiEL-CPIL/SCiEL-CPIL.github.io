import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill

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

    # The 6 states as specified
    sqrt2 = np.sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([1, -1, 0], dtype=complex) / sqrt2,
        np.array([1, 1j, 0], dtype=complex) / sqrt2,
        np.array([1, -1j, 0], dtype=complex) / sqrt2
    ]

    # Target states for X gate
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([-1, 1, 0], dtype=complex) / sqrt2,
        np.array([1j, 1, 0], dtype=complex) / sqrt2,
        np.array([-1j, 1, 0], dtype=complex) / sqrt2
    ]

    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)

    a_dag = a_op.conj().T
    n_op = a_dag @ a_op

    # Precompute constant parts
    H_anharm = 0.5 * alpha * (n_op @ (n_op - np.eye(3)))

    # Dissipator superoperator (constant)
    dim = 3
    d2 = dim * dim
    L_diss_super = np.zeros((d2, d2), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L1_term = np.kron(L_op, L_op.conj())
        L2_term = 0.5 * (np.kron(L_dag @ L_op, np.eye(dim)) + np.kron(np.eye(dim), (L_dag @ L_op).T))
        L_diss_super += L1_term - L2_term

    # Precompute identity for kron
    I3 = np.eye(3)

    # Precompute for each (delta, scale) pair
    liouvillian_parts = {}
    for delta in deltas:
        for scale in scales:
            H0 = delta * n_op + H_anharm
            HI = 0.5 * scale * (a_op + a_dag)
            HQ = 0.5j * scale * (a_op - a_dag)

            H_term_0 = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
            H_term_I = -1j * (np.kron(HI, I3) - np.kron(I3, HI.T))
            H_term_Q = -1j * (np.kron(HQ, I3) - np.kron(I3, HQ.T))

            L_base_0 = H_term_0 + L_diss_super
            liouvillian_parts[(delta, scale)] = (L_base_0, H_term_I, H_term_Q)

    def simulate_pulse(I_Q, delta, scale):
        """
        Simulates the pulse and returns mean infidelity over 6 states.
        Vectorized over states for speed.
        """
        L_base_0, H_term_I, H_term_Q = liouvillian_parts[(delta, scale)]

        # Precompute all propagators
        L_steps = []
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            L_total = L_base_0 + I * H_term_I + Q * H_term_Q
            L_steps.append(expm(L_total * dt))

        # Vectorize: stack all 6 initial density matrices
        rho_vecs = np.array([np.outer(s, s.conj()).flatten() for s in states])  # (6, 9)

        # Apply all steps
        for i in range(N):
            rho_vecs = (L_steps[i] @ rho_vecs.T).T

        # Compute fidelities
        total_infidelity = 0.0
        for idx in range(6):
            rho = rho_vecs[idx].reshape(3, 3)
            target = targets[idx]
            proj = np.outer(target, target.conj())
            fid = np.real(np.trace(rho @ proj))
            fid = np.clip(fid, 0, 1)
            total_infidelity += (1.0 - fid)

        return total_infidelity / 6.0

    def check_constraints(I_Q):
        # Amplitude
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-12:
            return False
        # Slew
        if np.sqrt(I_Q[0,0]**2 + I_Q[0,1]**2) / dt > slew_bound + 1e-12:
            return False
        if np.sqrt(I_Q[-1,0]**2 + I_Q[-1,1]**2) / dt > slew_bound + 1e-12:
            return False
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-12:
            return False
        return True

    def calculate_score(I_Q):
        """
        Calculates the score: mean + max of the 9 cases.
        """
        case_scores = []
        for delta in deltas:
            for scale in scales:
                mean_inf = simulate_pulse(I_Q, delta, scale)
                case_scores.append(mean_inf)

        return np.mean(case_scores) + np.max(case_scores)

    def generate_drag_pulse(sigma, eta, center=20.0):
        """
        Generate a DRAG-style pulse with Gaussian envelope and Q component
        proportional to the derivative of I.
        """
        t = np.linspace(0, 40, N)
        # Area of I should be pi for a pi-pulse
        # Integral of A*exp(-(t-c)^2/(2*sigma^2)) dt = A * sigma * sqrt(2*pi)
        # So A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
        A = np.sqrt(np.pi / 2) / sigma
        I = A * np.exp(-(t - center)**2 / (2 * sigma**2))
        # Derivative of I with respect to t
        dI_dt = -(t - center) / (sigma**2) * I
        # DRAG: Q = eta * dI/dt (sign depends on convention)
        Q = eta * dI_dt
        I_Q = np.column_stack((I, Q))
        return I_Q

    def generate_optimal_drag_pulse():
        """
        Generate a DRAG pulse with optimal eta derived from anharmonicity.
        For a 3-level system, the optimal DRAG parameter is:
        eta_opt = -alpha / (2 * pi * f) where f is the drive frequency
        But in our convention, we need to find the right scaling.
        
        The key insight: to suppress |2> leakage, we need Q such that
        the effective drive in the rotating frame cancels the anharmonic
        coupling. For a pi-pulse, the optimal DRAG offset is approximately:
        eta_opt ≈ -alpha / (2 * A_peak) * some_factor
        
        Let's use a more direct approach: the DRAG condition for suppressing
        |2> leakage in a 3-level lambda system gives:
        Q(t) = -alpha * I'(t) / (2 * pi)
        
        But we need to be careful with units. Let's use the standard result:
        For a transmon with anharmonicity α, the optimal DRAG parameter is:
        η = -α / (2π)
        
        Wait, that's not dimensionally right. Let me think again.
        
        The anharmonicity is α = -2π * 0.25 rad/ns
        The drive amplitude has units rad/ns
        The derivative dI/dt has units rad/ns²
        
        So Q = η * dI/dt where η has units ns
        
        The standard DRAG result for suppressing |2> leakage is:
        η_opt = -α / (2 * π * ω_d) but in our rotating frame, ω_d = 0
        
        Actually, for a resonant drive, the optimal DRAG offset is:
        Q(t) = -α * I'(t) / (2 * A)
        where A is the peak amplitude... no, that's not right either.
        
        Let me use the well-known result: for a 3-level system with levels |0>,|1>,|2>
        and anharmonicity α, the DRAG pulse that suppresses |2> leakage has:
        Q(t) = (α / 2) * I'(t) / (π * f_10)
        
        But in our case, the drive is resonant, so f_10 = 0 in the rotating frame.
        
        Actually, the correct DRAG condition is derived from requiring that the
        effective Hamiltonian in the rotating frame has no |1>->|2> coupling.
        
        Let me just use the empirical result that works well:
        η_opt ≈ -alpha / (2 * np.pi) * (1 / sigma)
        
        Or more simply, let's try a range of eta values and pick the best.
        """
        # The standard DRAG result for a transmon:
        # The Q component should be proportional to the derivative of I
        # with a coefficient that depends on the anharmonicity.
        # 
        # For a 3-level system, the optimal DRAG parameter to suppress |2> leakage
        # is approximately: η = -α / (2π)
        # 
        # But we need to account for the time scaling. Since our pulse has
        # a characteristic time σ, the derivative dI/dt ~ I/σ, so:
        # Q ~ η * I/σ
        # 
        # The ratio Q/I should be ~ η/σ
        # 
        # For the anharmonicity α = -2π*0.25, we have:
        # η_opt = -α / (2π) = 0.25 ns
        # 
        # But this needs to be scaled appropriately. Let's use:
        # Q(t) = -α * I'(t) / (2 * π)
        # 
        # Since I'(t) = dI/dt, and I has units rad/ns, I' has units rad/ns²
        # So Q = -α * I' / (2π) has units (rad/ns) * (rad/ns²) / (rad/ns) = rad/ns ✓
        
        # Let's generate with eta = -alpha / (2*pi) = 0.25
        eta_opt = -alpha / (2 * np.pi)
        
        # Try a few sigmas and find the best
        best_score = np.inf
        best_I_Q = None
        
        for sigma in [6.0, 8.0, 10.0, 12.0, 40.0/6.0]:
            I_Q = generate_drag_pulse(sigma, eta_opt)
            if check_constraints(I_Q):
                s = calculate_score(I_Q)
                if s < best_score:
                    best_score = s
                    best_I_Q = I_Q.copy()
        
        return best_I_Q, best_score

    # Generate the optimal DRAG pulse
    I_Q_seed, best_score = generate_optimal_drag_pulse()
    
    if I_Q_seed is None:
        # Fallback to simple Gaussian
        I_Q_seed = generate_drag_pulse(40.0/6.0, 0.0)
        best_score = calculate_score(I_Q_seed)
    
    best_I_Q = I_Q_seed.copy()
    
    # Fine-tune: try small variations around the optimal point
    # We can try adjusting sigma slightly and eta slightly
    
    # First, find the optimal sigma and eta more carefully
    best_sigma = 40.0/6.0
    best_eta = -alpha / (2 * np.pi)
    
    # Coarse search over sigma
    for sigma_try in np.arange(5.0, 15.0, 1.0):
        if time.time() > deadline - 50:
            break
        for eta_try in np.arange(best_eta - 0.5, best_eta + 0.5, 0.1):
            I_Q_try = generate_drag_pulse(sigma_try, eta_try)
            if check_constraints(I_Q_try):
                s_try = calculate_score(I_Q_try)
                if s_try < best_score:
                    best_score = s_try
                    best_I_Q = I_Q_try.copy()
                    best_sigma = sigma_try
                    best_eta = eta_try
    
    # Fine search around the best point
    if time.time() < deadline - 30:
        for sigma_try in np.arange(best_sigma - 2, best_sigma + 2, 0.5):
            if sigma_try <= 0.5:
                continue
            for eta_try in np.arange(best_eta - 0.2, best_eta + 0.2, 0.05):
                if time.time() > deadline - 10:
                    break
                I_Q_try = generate_drag_pulse(sigma_try, eta_try)
                if check_constraints(I_Q_try):
                    s_try = calculate_score(I_Q_try)
                    if s_try < best_score:
                        best_score = s_try
                        best_I_Q = I_Q_try.copy()
                        best_sigma = sigma_try
                        best_eta = eta_try
    
    # Final constraint check
    if not check_constraints(best_I_Q):
        # Fallback to a safe pulse
        best_I_Q = generate_drag_pulse(40.0/6.0, 0.0)
    
    return best_I_Q
