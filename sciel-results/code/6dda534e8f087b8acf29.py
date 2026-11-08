import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill

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

    def simulate_pulse(I_Q, delta, scale):
        """
        Simulates the pulse and returns mean infidelity over 6 states.
        Uses matrix product of propagators for speed.
        """
        L_base_0, H_term_I, H_term_Q = liouvillian_parts[(delta, scale)]

        # Compute all propagators and their product
        U_total = np.eye(d2, dtype=complex)
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            L_total = L_base_0 + I * H_term_I + Q * H_term_Q
            U_step = expm(L_total * dt)
            U_total = U_step @ U_total

        # Simulate all 6 states using the total propagator
        total_infidelity = 0.0
        for idx in range(6):
            psi0 = states[idx]
            rho0 = np.outer(psi0, psi0.conj())
            rho_vec = rho0.flatten()

            rho_vec = U_total @ rho_vec
            rho = rho_vec.reshape(3, 3)
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

    def generate_pulse_two_gaussian(A1, t1, s1, A2, t2, s2):
        """
        Generate pulse as sum of two Gaussian components.
        Component 1: main X-pulse (I channel)
        Component 2: DRAG component (Q channel)
        """
        t = np.linspace(0, 40, N)

        # Gaussian 1: main pulse
        g1 = A1 * np.exp(-(t - t1)**2 / (2 * s1**2))
        # Gaussian 2: DRAG component (derivative-like)
        g2 = A2 * np.exp(-(t - t2)**2 / (2 * s2**2))

        I = g1
        Q = g2

        I_Q = np.column_stack((I, Q))
        return I_Q

    def generate_pulse_sigma_eta(sigma, eta, center=20.0):
        """
        Generate DRAG-style pulse with Gaussian envelope.
        """
        t = np.linspace(0, 40, N)
        # Normalize to have area pi
        A = np.sqrt(np.pi / 2) / sigma
        I = A * np.exp(-(t - center)**2 / (2 * sigma**2))
        dI = -(t - center) / (sigma**2) * I
        Q = eta * dI
        I_Q = np.column_stack((I, Q))
        return I_Q

    # Seed pulse
    sigma_seed = 40.0 / 6.0
    I_Q_seed = generate_pulse_sigma_eta(sigma_seed, 0.0)

    # Check seed
    if not check_constraints(I_Q_seed):
        # Adjust seed if necessary - scale down
        scale_factor = 0.9
        I_Q_seed = I_Q_seed * scale_factor
        if not check_constraints(I_Q_seed):
            scale_factor = 0.8
            I_Q_seed = I_Q_seed * scale_factor

    best_I_Q = I_Q_seed.copy()
    best_score = calculate_score(best_I_Q)

    # Objective function for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e10
        sigma, eta = params
        if sigma <= 0.5 or sigma > 50:
            return 1e10
        I_Q = generate_pulse_sigma_eta(sigma, eta)
        if not check_constraints(I_Q):
            return 1e10
        return calculate_score(I_Q)

    # Phase 1: Coarse Grid Search
    grid_deadline = start_time + 60.0

    sigmas = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
    etas = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]

    best_grid_params = [sigma_seed, 0.0]

    for s in sigmas:
        for e in etas:
            if time.time() > grid_deadline:
                break
            score = objective([s, e])
            if score < best_score:
                best_score = score
                best_I_Q = generate_pulse_sigma_eta(s, e)
                best_grid_params = [s, e]
        if time.time() > grid_deadline:
            break

    # Phase 2: Local Optimization (Nelder-Mead)
    local_deadline = start_time + 200.0

    # Start from best grid point and diverse points
    nm_starts = [
        best_grid_params,
        [6.0, -2.0],
        [7.0, -3.0],
        [8.0, -2.5],
        [5.0, -1.5],
        [9.0, -3.5],
        [10.0, -4.0],
        [sigma_seed, -1.0],
        [sigma_seed, -2.0],
        [sigma_seed, -3.0]
    ]

    # Remove duplicates
    unique_starts = []
    for sp in nm_starts:
        if sp not in unique_starts:
            unique_starts.append(sp)

    for sp in unique_starts:
        if time.time() > local_deadline:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(objective, sp, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_I_Q = generate_pulse_sigma_eta(res.x[0], res.x[1])
                best_grid_params = [res.x[0], res.x[1]]
        except Exception:
            pass

    # Phase 3: Final Polish with tighter tolerances
    if time.time() < deadline - 10:
        final_deadline = deadline - 10
        final_starts = [
            best_grid_params,
            [6.0, -2.5],
            [7.0, -3.0],
            [8.0, -3.5],
            [5.5, -2.0],
            [9.0, -4.0]
        ]

        unique_final = []
        for sp in final_starts:
            if sp not in unique_final:
                unique_final.append(sp)

        for sp in unique_final:
            if time.time() > final_deadline:
                break
            try:
                from scipy.optimize import minimize
                res = minimize(objective, sp, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_I_Q = generate_pulse_sigma_eta(res.x[0], res.x[1])
            except Exception:
                pass

    # Final Constraint Check
    if not check_constraints(best_I_Q):
        best_I_Q = I_Q_seed.copy()

    return best_I_Q
