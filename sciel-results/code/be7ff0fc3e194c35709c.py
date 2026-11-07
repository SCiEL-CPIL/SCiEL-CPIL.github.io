import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin before 330s kill

    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity (rad/ns)
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0

    # Detuning and scale cases
    deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    # Basis states for fidelity evaluation
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]

    # Target states after X gate
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2)
    ]

    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    n_nm1 = n_op @ (n_op - np.eye(3))

    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    def get_H(I, Q, delta, scale):
        H0 = delta * n_op + 0.5 * alpha * n_nm1
        drive = 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H0 + drive

    def build_liouvillian(H):
        # Build 9x9 Liouvillian superoperator
        # L = -i(H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 0.5(L_k^†L_k)⊗I - 0.5 I⊗(L_k^†L_k)^T)
        dim = 3
        I3 = np.eye(dim)
        L_super = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
        
        for Lk in [L1, L2]:
            Lk_dag_Lk = Lk.conj().T @ Lk
            L_super += np.kron(Lk, Lk.conj()) - 0.5 * np.kron(Lk_dag_Lk, I3) - 0.5 * np.kron(I3, Lk_dag_Lk.T)
        
        return L_super

    def simulate_full(I_Q, delta, scale):
        """Full simulation with decoherence using Liouvillian exponentiation."""
        # Precompute Liouvillians for each step
        L_steps = []
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            L_steps.append(build_liouvillian(H))
        
        expm_steps = [expm(L * dt) for L in L_steps]
        
        fidels = []
        for idx in range(6):
            psi0 = states[idx]
            target = targets[idx]
            rho = np.outer(psi0, psi0.conj()).flatten()
            
            for i in range(N):
                rho = expm_steps[i] @ rho
            
            rho_mat = rho.reshape(3, 3)
            fid = np.real(np.vdot(target, rho_mat @ target))
            # Fidelity for pure target: |<target|rho|target>|
            fid = np.abs(np.vdot(target, rho_mat @ target))
            fidels.append(1.0 - fid)
        
        return fidels

    def simulate_fast(I_Q, delta, scale):
        """Fast simulation with only Hamiltonian (no decoherence) for optimization."""
        H_steps = [get_H(I_Q[i, 0], I_Q[i, 1], delta, scale) for i in range(N)]
        U_steps = [expm(-1j * H * dt) for H in H_steps]
        
        fidels = []
        for idx in range(6):
            psi = states[idx].copy()
            target = targets[idx]
            
            for i in range(N):
                psi = U_steps[i] @ psi
            
            fid = np.abs(np.vdot(target, psi))**2
            fidels.append(1.0 - fid)
        
        return fidels

    def score_pulse(I_Q, use_fast=True):
        """Compute score: mean + max of infidelities over 9 cases."""
        case_scores = []
        for delta in deltas:
            for scale in scales:
                if use_fast:
                    fidels = simulate_fast(I_Q, delta, scale)
                else:
                    fidels = simulate_full(I_Q, delta, scale)
                mean_inf = np.mean(fidels)
                max_inf = np.max(fidels)
                case_scores.append(mean_inf + max_inf)
        
        return np.mean(case_scores)

    def check_constraints(I_Q):
        """Check amplitude and slew constraints."""
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-10:
            return False
        
        # Slew rate: ||p[i+1] - p[i]|| / dt <= slew_bound
        # Including from zero into step 0 and from last step to zero
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-10:
            return False
        
        return True

    def generate_pulse(params):
        """Generate pulse from parameters: [sigma_I, eta, sigma_Q, shift_Q]"""
        sigma_I, eta, sigma_Q, shift_Q = params
        t = np.linspace(0, 40, N)
        t_c = 20.0
        
        # Main Gaussian envelope
        I = (np.sqrt(np.pi / 2) / sigma_I) * np.exp(-(t - t_c)**2 / (2 * sigma_I**2))
        
        # DRAG component: proportional to derivative of I, with independent sigma and shift
        dI = -(t - t_c) / (sigma_I**2) * I
        Q = eta * (np.sqrt(np.pi / 2) / sigma_Q) * np.exp(-(t - t_c - shift_Q)**2 / (2 * sigma_Q**2)) * \
             np.sign(dI) * np.abs(dI) / (np.max(np.abs(dI)) + 1e-15)
        
        # Actually, let's use a cleaner formulation:
        # Q = eta * dI/dt normalized
        dI_dt = -(t - t_c) / (sigma_I**2) * I / dt
        Q = eta * dI_dt
        
        I_Q = np.column_stack((I, Q))
        return I_Q

    def generate_pulse_v2(params):
        """Generate pulse with two independent Gaussian envelopes."""
        amp_I, sigma_I, eta, sigma_Q, shift_Q = params
        t = np.linspace(0, 40, N)
        t_c = 20.0
        
        # Main I pulse
        I = amp_I * np.exp(-(t - t_c)**2 / (2 * sigma_I**2))
        
        # Q pulse (DRAG-like)
        Q = eta * np.exp(-(t - t_c - shift_Q)**2 / (2 * sigma_Q**2))
        
        I_Q = np.column_stack((I, Q))
        return I_Q

    def generate_pulse_v3(params):
        """Generate pulse: Gaussian I + derivative-coupled Q."""
        amp_I, sigma_I, eta = params
        t = np.linspace(0, 40, N)
        t_c = 20.0
        
        I = amp_I * np.exp(-(t - t_c)**2 / (2 * sigma_I**2))
        dI = -(t - t_c) / (sigma_I**2) * I
        Q = eta * dI
        
        I_Q = np.column_stack((I, Q))
        return I_Q

    # Seed pulse: DRAG-style Gaussian
    def seed_pulse():
        sigma = 40.0 / 6.0
        t = np.linspace(0, 40, N)
        t_c = 20.0
        I = (np.sqrt(np.pi / 2) / sigma) * np.exp(-(t - t_c)**2 / (2 * sigma**2))
        Q = np.zeros(N)
        return np.column_stack((I, Q))

    # Start with seed
    best_I_Q = seed_pulse()
    best_score = score_pulse(best_I_Q, use_fast=True)

    # Grid search for initial parameters
    best_params = None
    best_grid_score = best_score

    for amp_I in [0.2, 0.3, 0.35, 0.4]:
        for sigma_I in [4, 5, 6, 7, 8, 10]:
            for eta in [-5, -3, -1, 0, 1, 3, 5]:
                if time.time() > deadline - 50:
                    break
                I_Q = generate_pulse_v3([amp_I, sigma_I, eta])
                if not check_constraints(I_Q):
                    continue
                s = score_pulse(I_Q, use_fast=True)
                if s < best_grid_score:
                    best_grid_score = s
                    best_params = [amp_I, sigma_I, eta]
                    best_I_Q = I_Q

    if best_params is None:
        best_params = [0.3, 6.0, 0.0]

    # Local optimization
    def opt_func(params):
        if time.time() > deadline - 30:
            return 10.0
        amp_I, sigma_I, eta = params
        if amp_I < 0.05 or amp_I > 0.45 or sigma_I < 2 or sigma_I > 20 or abs(eta) > 10:
            return 10.0
        I_Q = generate_pulse_v3(params)
        if not check_constraints(I_Q):
            return 10.0
        return score_pulse(I_Q, use_fast=True)

    from scipy.optimize import minimize
    res = minimize(opt_func, best_params, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1e-5})
    if res.fun < best_grid_score:
        best_grid_score = res.fun
        best_I_Q = generate_pulse_v3(res.x)

    # Try a second optimization with different initial point
    if time.time() < deadline - 40:
        best_params2 = [0.35, 5.0, -3.0]
        res2 = minimize(opt_func, best_params2, method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 0.01, 'fatol': 1e-5})
        if res2.fun < best_grid_score:
            best_grid_score = res2.fun
            best_I_Q = generate_pulse_v3(res2.x)

    # Try optimizing with 5-parameter model if time allows
    if time.time() < deadline - 30:
        def opt_func5(params):
            if time.time() > deadline - 15:
                return 10.0
            amp_I, sigma_I, eta, sigma_Q, shift_Q = params
            if amp_I < 0.05 or amp_I > 0.45 or sigma_I < 2 or sigma_I > 20 or abs(eta) > 10:
                return 10.0
            I_Q = generate_pulse_v2(params)
            if not check_constraints(I_Q):
                return 10.0
            return score_pulse(I_Q, use_fast=True)

        init5 = [best_params[0], best_params[1], best_params[2], 5.0, 0.0]
        res5 = minimize(opt_func5, init5, method='Nelder-Mead',
                        options={'maxiter': 150, 'xatol': 0.01, 'fatol': 1e-5})
        if res5.fun < best_grid_score:
            best_grid_score = res5.fun
            best_I_Q = generate_pulse_v2(res5.x)

    # Final constraint check and fallback
    if not check_constraints(best_I_Q):
        best_I_Q = seed_pulse()

    return best_I_Q
