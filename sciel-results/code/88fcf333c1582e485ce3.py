import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin

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
    I3 = np.eye(3)

    # Precompute constant parts
    H_anharm = 0.5 * alpha * (n_op @ (n_op - I3))

    # Dissipator superoperator (constant)
    dim = 3
    d2 = dim * dim
    L_diss_super = np.zeros((d2, d2), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L1_term = np.kron(L_op, L_op.conj())
        L2_term = 0.5 * (np.kron(L_dag @ L_op, I3) + np.kron(I3, (L_dag @ L_op).T))
        L_diss_super += L1_term - L2_term

    # Precompute Liouvillian parts for each (delta, scale) pair
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

    # Precompute initial state vectors (flattened density matrices)
    rho0_vecs = np.zeros((d2, 6), dtype=complex)
    for i in range(6):
        rho0 = np.outer(states[i], states[i].conj())
        rho0_vecs[:, i] = rho0.flatten()

    # Precompute target projectors (flattened)
    target_proj_vecs = np.zeros((d2, 6), dtype=complex)
    for i in range(6):
        proj = np.outer(targets[i], targets[i].conj())
        target_proj_vecs[:, i] = proj.flatten()

    def simulate_pulse(I_Q):
        """
        Simulates the pulse for all 9 cases and returns the score (mean + max infidelity).
        I_Q: (80, 2) array.
        """
        case_infs = []
        for delta in deltas:
            for scale in scales:
                L_base_0, H_term_I, H_term_Q = liouvillian_parts[(delta, scale)]
                
                # Compute total propagator U_total
                U_total = np.eye(d2, dtype=complex)
                for i in range(N):
                    I_val = I_Q[i, 0]
                    Q_val = I_Q[i, 1]
                    L_total = L_base_0 + I_val * H_term_I + Q_val * H_term_Q
                    U_step = expm(L_total * dt)
                    U_total = U_step @ U_total
                
                # Apply U_total to all 6 initial states
                # rho_final_vecs = U_total @ rho0_vecs
                rho_final_vecs = U_total @ rho0_vecs
                
                # Calculate fidelity for each state
                # Fidelity = trace(rho_final @ target_proj)
                # In vector form: <vec(target_proj), vec(rho_final)> = sum(conj(target_proj_vec) * rho_final_vec)
                # Note: vec(A) @ vec(B) = trace(A^T B). 
                # Fidelity = Tr(rho P) = sum_ij rho_ij P_ji = sum_ij rho_ij conj(P_ij) (since P is Hermitian)
                # So F = sum(rho_vec * conj(P_vec))
                
                fids = np.sum(rho_final_vecs * np.conj(target_proj_vecs), axis=0)
                fids = np.real(fids)
                fids = np.clip(fids, 0, 1)
                
                infid = 1.0 - np.mean(fids)
                case_infs.append(infid)
                
        return np.mean(case_infs) + np.max(case_infs)

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

    def generate_pulse(params):
        """
        Generates pulse from parameters.
        params: [A1, t1, s1, A2, t2, s2, B1, u1, r1, B2, u2, r2]
        I(t) = A1*exp(-(t-t1)^2/2s1^2) + A2*exp(-(t-t2)^2/2s2^2)
        Q(t) = B1*exp(-(t-u1)^2/2r1^2) + B2*exp(-(t-u2)^2/2r2^2)
        """
        t = np.linspace(0, 40, N)
        
        A1, t1, s1, A2, t2, s2 = params[:6]
        B1, u1, r1, B2, u2, r2 = params[6:]
        
        # Avoid division by zero
        s1 = max(abs(s1), 0.5)
        s2 = max(abs(s2), 0.5)
        r1 = max(abs(r1), 0.5)
        r2 = max(abs(r2), 0.5)
        
        I = A1 * np.exp(-(t - t1)**2 / (2 * s1**2)) + A2 * np.exp(-(t - t2)**2 / (2 * s2**2))
        Q = B1 * np.exp(-(t - u1)**2 / (2 * r1**2)) + B2 * np.exp(-(t - u2)**2 / (2 * r2**2))
        
        I_Q = np.column_stack((I, Q))
        return I_Q

    def objective(params):
        if time.time() > deadline:
            return 1e10
        I_Q = generate_pulse(params)
        if not check_constraints(I_Q):
            return 1e10
        return simulate_pulse(I_Q)

    # Seed pulse: Single Gaussian + DRAG
    # A ~ 0.3, sigma ~ 6, eta ~ -2
    # I = A * exp(-(t-20)^2 / 2*6^2)
    # Q = eta * dI/dt
    
    # Initialize best
    best_score = 1e10
    best_params = [0.3, 20, 6, 0, 20, 6, -0.6, 20, 6, 0, 20, 6] # A1, t1, s1, A2, t2, s2, B1, u1, r1, B2, u2, r2
    # Note: For DRAG, Q is proportional to derivative. 
    # d/dt exp(-(t-20)^2/72) = -(t-20)/36 * exp(...)
    # So if I = A exp(...), Q ~ eta * A * -(t-20)/sigma^2 * exp(...)
    # This shape is not a Gaussian. 
    # My parameterization uses Gaussians for Q. 
    # A Gaussian centered at 20 is not a derivative. 
    # To approximate derivative, we can use two close Gaussians with opposite signs? 
    # Or just let the optimizer find the shape.
    
    # Let's start with a reasonable guess.
    # I: Main pulse at 20, width 6, amp 0.3
    # Q: Small component to cancel leakage. Often centered slightly off or same center.
    # Let's try Q centered at 20, width 6, small amp.
    
    initial_guesses = [
        [0.3, 20, 6, 0, 20, 6, 0.05, 20, 6, 0, 20, 6],
        [0.3, 20, 6, 0, 20, 6, -0.05, 20, 6, 0, 20, 6],
        [0.3, 20, 5, 0, 20, 5, 0.05, 20, 5, 0, 20, 5],
        [0.3, 20, 7, 0, 20, 7, -0.05, 20, 7, 0, 20, 7],
        [0.25, 20, 6, 0.05, 20, 6, 0.05, 20, 6, 0, 20, 6],
        [0.25, 20, 6, 0.05, 20, 6, -0.05, 20, 6, 0, 20, 6]
    ]

    # Evaluate initial guesses
    for guess in initial_guesses:
        score = objective(guess)
        if score < best_score:
            best_score = score
            best_params = guess

    # Optimize from best guess
    if time.time() < deadline - 50:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    # Try a few other local optimizations from diverse points if time remains
    if time.time() < deadline - 30:
        other_starts = [
            [0.35, 20, 5, 0, 20, 5, 0.08, 20, 5, 0, 20, 5],
            [0.25, 20, 8, 0, 20, 8, -0.04, 20, 8, 0, 20, 8]
        ]
        for start in other_starts:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                pass

    # Final check
    final_I_Q = generate_pulse(best_params)
    if not check_constraints(final_I_Q):
        # Fallback to seed
        t = np.linspace(0, 40, N)
        sigma = 40.0 / 6.0
        A = np.sqrt(np.pi / 2) / sigma
        I = A * np.exp(-(t - 20)**2 / (2 * sigma**2))
        Q = np.zeros(N)
        final_I_Q = np.column_stack((I, Q))
        # Ensure constraints
        while not check_constraints(final_I_Q):
            final_I_Q *= 0.9

    return final_I_Q
