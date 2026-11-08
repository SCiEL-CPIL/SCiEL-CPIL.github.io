# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    dt = 0.5
    N = 80
    total_time = N * dt

    # 3-level operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a  # diag(0, 1, 2)
    I3 = np.eye(3)

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2

    # Precompute static Liouvillian parts
    L1_term = np.kron(L1, L1.T)
    L2_term = np.kron(L2, L2.T)
    L1_diss_left = np.kron(I3, L1_dag_L1)
    L1_diss_right = np.kron(L1_dag_L1.T, I3)
    L2_diss_left = np.kron(I3, L2_dag_L2)
    L2_diss_right = np.kron(L2_dag_L2.T, I3)

    static_Liouv = L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                  + L2_term - 0.5 * (L2_diss_left + L2_diss_right)

    # Anharmonicity contribution
    an_ham = 0.5 * anharmonicity * n_op @ (n_op - I3)

    # Cases: detuning x scale
    detunings = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
    scales = np.array([0.97, 1.0, 1.03])

    # Precompute case-specific static Liouvillians
    case_static_L = np.zeros((9, 9, 9), dtype=complex)
    case_scale = np.zeros(9)
    case_idx = 0
    for det in detunings:
        for sc in scales:
            H_static = det * n_op + an_ham
            H_left = np.kron(I3, H_static)
            H_right = np.kron(H_static.T, I3)
            case_static_L[case_idx] = -1j * (H_left - H_right) + static_Liouv
            case_scale[case_idx] = sc
            case_idx += 1

    # Test states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]

    # X_pi target: -i * swap |0> and |1>
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = -1j * s[1]
        t[1] = -1j * s[0]
        t[2] = s[2]
        target_states.append(t)

    def check_validity(pulse_IQ):
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-12:
                return False
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-12:
            return False
        return True

    def make_drag_pulse(alpha=0.0, sigma=40.0/6.0):
        t = np.arange(N) * dt
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-15:
            gaussian *= np.pi / area
        I = gaussian.copy()
        if alpha != 0:
            dI_dt = np.gradient(I, dt)
            Q = alpha * dI_dt
        else:
            Q = np.zeros(N)
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
        return pulse_IQ

    def simulate_pulse(pulse_IQ):
        """
        Simulate all 9 cases x 6 states.
        Returns rho_final of shape (9, 6, 9).
        """
        U_all = np.zeros((9, N, 9, 9), dtype=complex)

        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                I_val = pulse_IQ[step, 0]
                Q_val = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_all[ci, step] = expm(dt * L)

        rho_init = np.zeros((6, 9), dtype=complex)
        for si in range(6):
            psi = states[si]
            rho_mat = np.outer(psi, psi.conj())
            rho_init[si] = rho_mat.flatten()

        rho_final = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                rho_final[ci, si] = vec

        return rho_final, U_all

    def compute_fidelities(rho_final):
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                rho_mat = rho_final[ci, si].reshape(3, 3)
                psi_t = target_states[si]
                fids[ci, si] = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
        return fids

    def compute_score(fids):
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def compute_soft_score(fids, beta=50.0):
        infids = 1.0 - fids
        m = np.max(infids)
        soft_max = m + np.log(np.sum(np.exp(beta * (infids - m)))) / beta
        return np.mean(infids) + soft_max

    # Time management
    start_time = time.time()
    deadline = start_time + 300.0

    def time_remaining():
        return deadline - time.time()

    # Phase 1: DRAG alpha sweep to find a good starting point
    pulse_IQ = make_drag_pulse(0.0)
    best_pulse = pulse_IQ.copy()
    rho_best, _ = simulate_pulse(best_pulse)
    best_score = compute_soft_score(compute_fidelities(rho_best), beta=50.0)

    alphas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
              0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
              -0.05, -0.1, -0.15, -0.2, -0.3]
    for alpha in alphas:
        if time_remaining() < 40:
            break
        p_test = make_drag_pulse(alpha)
        if check_validity(p_test):
            rho_t, _ = simulate_pulse(p_test)
            s = compute_soft_score(compute_fidelities(rho_t), beta=50.0)
            if s < best_score:
                best_score = s
                best_pulse = p_test.copy()

    # Phase 2: Differential evolution for global search
    if time_remaining() > 150:
        # Use a subset of parameters for DE to keep it manageable
        # Optimize a lower-dimensional representation: use a few basis functions
        # Represent pulse as combination of Gaussian basis functions
        
        def basis_gaussian(t, center, width):
            return np.exp(-0.5 * ((t - center) / width)**2)
        
        t = np.arange(N) * dt
        # Use 4 Gaussian basis functions for I and Q
        centers = [total_time/4, 3*total_time/4, total_time/2, total_time/2]
        widths = [8.0, 8.0, 12.0, 6.0]
        
        def params_to_pulse(params):
            I = np.zeros(N)
            Q = np.zeros(N)
            for k in range(4):
                I += params[k] * basis_gaussian(t, centers[k], widths[k])
                Q += params[4+k] * basis_gaussian(t, centers[k], widths[k])
            # Normalize area for I
            area_I = np.sum(I) * dt
            if abs(area_I) > 1e-10:
                I *= np.pi / area_I
            pulse_IQ = np.column_stack([I, Q])
            if not check_validity(pulse_IQ):
                max_amp = np.max(np.sqrt(I**2 + Q**2))
                if max_amp > amplitude_bound:
                    pulse_IQ *= amplitude_bound / max_amp
            return pulse_IQ
        
        def de_objective(params):
            if time_remaining() < 20:
                return 1e10
            p = params_to_pulse(params)
            rho_f, _ = simulate_pulse(p)
            fids = compute_fidelities(rho_f)
            return compute_soft_score(fids, beta=50.0)
        
        bounds_de = [(-0.5, 0.5)] * 8
        try:
            result_de = differential_evolution(de_objective, bounds_de, maxiter=30,
                                               popsize=15, tol=1e-4, seed=42,
                                               polish=False, updating='deferred')
            if time_remaining() > 60:
                p_de = params_to_pulse(result_de.x)
                if check_validity(p_de):
                    rho_de, _ = simulate_pulse(p_de)
                    s_de = compute_soft_score(compute_fidelities(rho_de), beta=50.0)
                    if s_de < best_score:
                        best_score = s_de
                        best_pulse = p_de.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead coarse optimization on full pulse
    if time_remaining() > 80:
        x0 = best_pulse.flatten()

        def nm_obj(params):
            if time_remaining() < 15:
                return 1e10
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10
            rho_f, _ = simulate_pulse(p)
            fids = compute_fidelities(rho_f)
            return compute_soft_score(fids, beta=50.0)

        try:
            result = minimize(nm_obj, x0, method='Nelder-Mead',
                              options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 60})
            if time_remaining() > 40:
                p_nm = result.x.reshape(N, 2)
                if check_validity(p_nm):
                    rho_nm, _ = simulate_pulse(p_nm)
                    s_nm = compute_soft_score(compute_fidelities(rho_nm), beta=50.0)
                    if s_nm < best_score:
                        best_score = s_nm
                        best_pulse = p_nm.copy()
        except Exception:
            pass

    # Phase 4: L-BFGS-B with gradient
    if time_remaining() > 50:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def lbfgs_jac(params):
            if time_remaining() < 10:
                return 1e10, np.zeros(2*N)
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10, np.zeros(2*N)

            rho_final, U_all = simulate_pulse(p)
            fids = compute_fidelities(rho_final)
            s = compute_soft_score(fids, beta=50.0)

            # Compute gradient via adjoint method
            infids = 1.0 - fids
            m = np.max(infids)
            exp_vals = np.exp(50.0 * (infids - m))
            sum_exp = np.sum(exp_vals)
            d_s_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (sum_exp * 50.0)
            d_s_d_fid = -d_s_d_infid.reshape(9, 6)

            # Gradient w.r.t. rho_final
            grad_rho = np.zeros((9, 6, 9), dtype=complex)
            for ci in range(9):
                for si in range(6):
                    psi_t = target_states[si]
                    fid_grad = np.outer(np.conj(psi_t), psi_t).real.flatten()
                    grad_rho[ci, si] = d_s_d_fid[ci, si] * fid_grad

            grad_params = np.zeros(2*N)

            for ci in range(9):
                lam = np.zeros(6, 9)
                for si in range(6):
                    lam[si] = grad_rho[ci, si].real

                for step in range(N - 1, -1, -1):
                    U_step = U_all[ci, step]

                    # Compute rho at this step
                    rho_at_step = np.zeros(6, 9)
                    for si in range(6):
                        psi = states[si]
                        rho_vec = np.outer(psi, psi.conj()).flatten()
                        for s2 in range(step):
                            rho_vec = U_all[ci, s2] @ rho_vec
                        rho_at_step[si] = rho_vec

                    dS_dU = np.zeros((9, 9))
                    for si in range(6):
                        dS_dU += np.outer(lam[si], rho_at_step[si])

                    dL_dI = -1j * (np.kron(I3, 0.5 * case_scale[ci] * (a + a_dag)) -
                                   np.kron((0.5 * case_scale[ci] * (a + a_dag)).T, I3))
                    dL_dQ = -1j * (np.kron(I3, 0.5j * case_scale[ci] * (a - a_dag)) -
                                   np.kron((0.5j * case_scale[ci] * (a - a_dag)).T, I3))

                    dU_dI = dt * dL_dI
                    dU_dQ = dt * dL_dQ

                    grad_params[step] += np.real(np.sum(dS_dU * dU_dI.T))
                    grad_params[N + step] += np.real(np.sum(dS_dU * dU_dQ.T))

                    for si in range(6):
                        lam[si] = U_step.T @ lam[si]

            return s, grad_params

        try:
            result2 = minimize(lbfgs_jac, x0, method='L-BFGS-B', jac=True,
                               bounds=bounds,
                               options={'maxiter': 40, 'ftol': 1e-14, 'gtol': 1e-8, 'maxfev': 200})
            if time_remaining() > 20:
                p_bfgs = result2.x.reshape(N, 2)
                if check_validity(p_bfgs):
                    rho_b, _ = simulate_pulse(p_bfgs)
                    s_b = compute_soft_score(compute_fidelities(rho_b), beta=50.0)
                    if s_b < best_score:
                        best_score = s_b
                        best_pulse = p_bfgs.copy()
        except Exception:
            pass

    # Phase 5: Final polish
    if time_remaining() > 20:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def polish_jac(params):
            if time_remaining() < 5:
                return 1e10, np.zeros(2*N)
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10, np.zeros(2*N)

            rho_final, U_all = simulate_pulse(p)
            fids = compute_fidelities(rho_final)
            s = compute_soft_score(fids, beta=100.0)

            infids = 1.0 - fids
            m = np.max(infids)
            exp_vals = np.exp(100.0 * (infids - m))
            sum_exp = np.sum(exp_vals)
            d_s_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (sum_exp * 100.0)
            d_s_d_fid = -d_s_d_infid.reshape(9, 6)

            grad_rho = np.zeros((9, 6, 9), dtype=complex)
            for ci in range(9):
                for si in range(6):
                    psi_t = target_states[si]
                    fid_grad = np.outer(np.conj(psi_t), psi_t).real.flatten()
                    grad_rho[ci, si] = d_s_d_fid[ci, si] * fid_grad

            grad_params = np.zeros(2*N)

            for ci in range(9):
                lam = np.zeros(6, 9)
                for si in range(6):
                    lam[si] = grad_rho[ci, si].real

                for step in range(N - 1, -1, -1):
                    U_step = U_all[ci, step]

                    rho_at_step = np.zeros(6, 9)
                    for si in range(6):
                        psi = states[si]
                        rho_vec = np.outer(psi, psi.conj()).flatten()
                        for s2 in range(step):
                            rho_vec = U_all[ci, s2] @ rho_vec
                        rho_at_step[si] = rho_vec

                    dS_dU = np.zeros((9, 9))
                    for si in range(6):
                        dS_dU += np.outer(lam[si], rho_at_step[si])

                    dL_dI = -1j * (np.kron(I3, 0.5 * case_scale[ci] * (a + a_dag)) -
                                   np.kron((0.5 * case_scale[ci] * (a + a_dag)).T, I3))
                    dL_dQ = -1j * (np.kron(I3, 0.5j * case_scale[ci] * (a - a_dag)) -
                                   np.kron((0.5j * case_scale[ci] * (a - a_dag)).T, I3))

                    dU_dI = dt * dL_dI
                    dU_dQ = dt * dL_dQ

                    grad_params[step] += np.real(np.sum(dS_dU * dU_dI.T))
                    grad_params[N + step] += np.real(np.sum(dS_dU * dU_dQ.T))

                    for si in range(6):
                        lam[si] = U_step.T @ lam[si]

            return s, grad_params

        try:
            result3 = minimize(polish_jac, x0, method='L-BFGS-B', jac=True,
                               bounds=bounds,
                               options={'maxiter': 30, 'ftol': 1e-16, 'gtol': 1e-10, 'maxfev': 100})
            if time_remaining() > 5:
                p_pol = result3.x.reshape(N, 2)
                if check_validity(p_pol):
                    rho_pol, _ = simulate_pulse(p_pol)
                    s_pol = compute_soft_score(compute_fidelities(rho_pol), beta=100.0)
                    if s_pol < best_score:
                        best_score = s_pol
                        best_pulse = p_pol.copy()
        except Exception:
            pass

    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = make_drag_pulse(0.0)

    return best_pulse
# EVOLVE-BLOCK-END
