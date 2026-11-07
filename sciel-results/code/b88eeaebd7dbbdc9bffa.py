# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
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
    n_op = a_dag @ a
    I3 = np.eye(3)

    # Precompute static parts of Liouvillian
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2

    L1_term = np.kron(L1, L1.T)
    L2_term = np.kron(L2, L2.T)
    L1_diss_left = np.kron(I3, L1_dag_L1)
    L1_diss_right = np.kron(L1_dag_L1.T, I3)
    L2_diss_left = np.kron(I3, L2_dag_L2)
    L2_diss_right = np.kron(L2_dag_L2.T, I3)

    static_Liouv = L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                  + L2_term - 0.5 * (L2_diss_left + L2_diss_right)

    an_ham = 0.5 * anharmonicity * n_op @ (n_op - I3)

    detunings = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
    scales = np.array([0.97, 1.0, 1.03])

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

    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]

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

    def make_drag_pulse(alpha=0.0):
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
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

    def simulate_all(pulse_IQ):
        rho_init = np.zeros((6, 9), dtype=complex)
        for si in range(6):
            psi = states[si]
            rho_mat = np.outer(psi, psi.conj())
            rho_init[si] = rho_mat.flatten()

        rho_final = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            sc = case_scale[ci]
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    I_val = pulse_IQ[step, 0]
                    Q_val = pulse_IQ[step, 1]
                    H_drive = 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
                    H_left = np.kron(I3, H_drive)
                    H_right = np.kron(H_drive.T, I3)
                    L = case_static_L[ci] - 1j * (H_left - H_right)
                    vec = expm(dt * L) @ vec
                rho_final[ci, si] = vec
        return rho_final

    def compute_fidelities(rho_final):
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                rho_mat = rho_final[ci, si].reshape(3, 3)
                psi_t = target_states[si]
                fid = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
                fids[ci, si] = fid
        return fids

    def compute_exact_score(fids):
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def compute_soft_score(fids, beta=50.0):
        infids = 1.0 - fids
        m = np.max(infids)
        soft_max = m + np.log(np.sum(np.exp(beta * (infids - m)))) / beta
        return np.mean(infids) + soft_max

    def objective_with_grad(params, beta=50.0):
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            return 1e10, np.zeros(2*N)

        rho_final = simulate_all(pulse_IQ)
        fids = compute_fidelities(rho_final)
        score = compute_soft_score(fids, beta=beta)

        eps = 1e-6
        grad = np.zeros(2*N)
        for j in range(2*N):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += eps
            params_minus[j] -= eps

            p_plus = np.column_stack([params_plus[:N], params_plus[N:2*N]])
            p_minus = np.column_stack([params_minus[:N], params_minus[N:2*N]])

            if check_validity(p_plus):
                r_plus = simulate_all(p_plus)
                f_plus = compute_fidelities(r_plus)
                s_plus = compute_soft_score(f_plus, beta=beta)
            else:
                s_plus = score + eps * 1000

            if check_validity(p_minus):
                r_minus = simulate_all(p_minus)
                f_minus = compute_fidelities(r_minus)
                s_minus = compute_soft_score(f_minus, beta=beta)
            else:
                s_minus = score + eps * 1000

            grad[j] = (s_plus - s_minus) / (2 * eps)

        return score, grad

    start_time = time.time()
    deadline = start_time + 280.0

    def time_remaining():
        return deadline - time.time()

    # Phase 1: DRAG alpha sweep
    pulse_IQ = make_drag_pulse(0.0)
    best_pulse = pulse_IQ.copy()
    best_score = compute_exact_score(compute_fidelities(simulate_all(best_pulse)))

    for alpha in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5,
                  -0.05, -0.1, -0.15, -0.2, -0.3]:
        if time_remaining() < 30:
            break
        p_test = make_drag_pulse(alpha)
        if check_validity(p_test):
            f = compute_fidelities(simulate_all(p_test))
            s = compute_exact_score(f)
            if s < best_score:
                best_score = s
                best_pulse = p_test.copy()

    # Phase 2: Nelder-Mead for coarse optimization
    if time_remaining() > 60:
        x0 = best_pulse.flatten()

        def nm_obj(params):
            if time_remaining() < 10:
                return 1e10
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10
            f = compute_fidelities(simulate_all(p))
            return compute_soft_score(f, beta=50.0)

        result = minimize(nm_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 100})
        if time_remaining() > 30:
            p_nm = result.x.reshape(N, 2)
            if check_validity(p_nm):
                f_nm = compute_fidelities(simulate_all(p_nm))
                s_nm = compute_exact_score(f_nm)
                if s_nm < best_score:
                    best_score = s_nm
                    best_pulse = p_nm.copy()

    # Phase 3: L-BFGS-B with numerical gradient
    if time_remaining() > 40:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def lbfgs_obj(params):
            if time_remaining() < 10:
                return 1e10, np.zeros(2*N)
            return objective_with_grad(params, beta=50.0)

        result2 = minimize(lbfgs_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfev': 300})
        if time_remaining() > 10:
            p_bfgs = result2.x.reshape(N, 2)
            if check_validity(p_bfgs):
                f_bfgs = compute_fidelities(simulate_all(p_bfgs))
                s_bfgs = compute_exact_score(f_bfgs)
                if s_bfgs < best_score:
                    best_score = s_bfgs
                    best_pulse = p_bfgs.copy()

    # Phase 4: Final polish
    if time_remaining() > 15:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2*N)

        def polish_obj(params):
            if time_remaining() < 5:
                return 1e10, np.zeros(2*N)
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10, np.zeros(2*N)
            f = compute_fidelities(simulate_all(p))
            s = compute_soft_score(f, beta=100.0)
            eps = 1e-7
            grad = np.zeros(2*N)
            for j in range(2*N):
                pp = params.copy()
                pm = params.copy()
                pp[j] += eps
                pm[j] -= eps
                ppp = np.column_stack([pp[:N], pp[N:2*N]])
                ppm = np.column_stack([pm[:N], pm[N:2*N]])
                if check_validity(ppp):
                    fp = compute_fidelities(simulate_all(ppp))
                    sp = compute_soft_score(fp, beta=100.0)
                else:
                    sp = s + 1000*eps
                if check_validity(ppm):
                    fm = compute_fidelities(simulate_all(ppm))
                    sm = compute_soft_score(fm, beta=100.0)
                else:
                    sm = s + 1000*eps
                grad[j] = (sp - sm) / (2*eps)
            return s, grad

        result3 = minimize(polish_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfev': 150})
        if time_remaining() > 5:
            p_pol = result3.x.reshape(N, 2)
            if check_validity(p_pol):
                f_pol = compute_fidelities(simulate_all(p_pol))
                s_pol = compute_exact_score(f_pol)
                if s_pol < best_score:
                    best_score = s_pol
                    best_pulse = p_pol.copy()

    if not check_validity(best_pulse):
        best_pulse = make_drag_pulse(0.0)

    return best_pulse
# EVOLVE-BLOCK-END
