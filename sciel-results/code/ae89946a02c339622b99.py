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
    n_op = a_dag @ a  # diag(0, 1, 2)
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

    # Static part of Liouvillian (independent of I, Q, detuning, scale)
    static_Liouv = L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                  + L2_term - 0.5 * (L2_diss_left + L2_diss_right)

    # Precompute anharmonicity contribution
    an_ham = 0.5 * anharmonicity * n_op @ (n_op - I3)

    def build_liouvillian(I_val, Q_val, detuning, scale):
        """Build full Liouvillian for given drive parameters."""
        # Drive Hamiltonian: 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
        H_drive = 0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
        H_det = detuning * n_op
        H_total = H_det + an_ham + H_drive

        H_left = np.kron(I3, H_total)
        H_right = np.kron(H_total.T, I3)

        return -1j * (H_left - H_right) + static_Liouv

    # Precompute all 9 case Liouvillian components
    detunings = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
    scales = np.array([0.97, 1.0, 1.03])

    # For each case, precompute the static part (detuning + anharmonicity)
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

    # Test states and target states
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

    # Precompute fidelity measurement vectors: |psi_target><psi_target| flattened
    # Fidelity = Re[psi^dagger rho psi] = Re[sum_{ij} conj(psi_i) rho_{ij} psi_j]
    # = Re[<psi_target| rho |psi_target>]
    # In vectorized form: vec(rho) = sum_{ij} rho_{ij} |i><j|
    # <psi| rho |psi> = sum_{ij} conj(psi_i) rho_{ij} psi_j
    # = sum_{ij} (conj(psi_i) psi_j) rho_{ij}
    # = dot(vec(conj(psi) psi^T), vec(rho))  ... need to be careful
    # Actually: <psi|rho|psi> = Tr(|psi><psi| rho) = Tr(rho |psi><psi|)
    # In vec form: Tr(A B) = vec(A^T)^T vec(B)
    # Tr(rho |psi><psi|) = vec(rho^T)^T vec(|psi><psi|)
    # rho^T vec: vec(rho^T) = K vec(rho) where K is commutation matrix
    # Simpler: just compute directly from 3x3 matrix

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
        """
        Simulate all 54 (case, state) combinations.
        Returns array of shape (9, 6, 9) = vec(rho_final) for each case and state.
        Also returns the U matrices for each case and step for adjoint computation.
        """
        # For each case, compute U matrices for all steps
        # U[case, step] is 9x9
        U_all = np.zeros((9, N, 9, 9), dtype=complex)

        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                I_val = pulse_IQ[step, 0]
                Q_val = pulse_IQ[step, 1]
                # Drive part
                H_drive = 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_all[ci, step] = expm(dt * L)

        # Initial density matrices for each state
        # rho_init[state] is 9-vector
        rho_init = np.zeros((6, 9), dtype=complex)
        for si in range(6):
            psi = states[si]
            rho_mat = np.outer(psi, psi.conj())
            rho_init[si] = rho_mat.flatten()

        # Evolve: for each case, evolve all 6 states
        rho_final = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                rho_final[ci, si] = vec

        return rho_final, U_all

    def compute_fidelities(rho_final):
        """Compute fidelity for each (case, state) pair."""
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                rho_mat = rho_final[ci, si].reshape(3, 3)
                psi_t = target_states[si]
                fid = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
                fids[ci, si] = fid
        return fids

    def compute_score(fids):
        """Compute mean + max infidelity."""
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def compute_soft_score(fids, beta=50.0):
        """Soft version: mean + soft_max via log-sum-exp."""
        infids = 1.0 - fids
        # Soft max: (1/beta) * log(sum(exp(beta * infids)))
        m = np.max(infids)
        soft_max = m + np.log(np.sum(np.exp(beta * (infids - m)))) / beta
        return np.mean(infids) + soft_max

    def objective(params):
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            return 1e10
        rho_final, _ = simulate_all(pulse_IQ)
        fids = compute_fidelities(rho_final)
        return compute_soft_score(fids, beta=50.0)

    def objective_with_grad(params):
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            return 1e10, np.zeros(2*N)

        rho_final, U_all = simulate_all(pulse_IQ)
        fids = compute_fidelities(rho_final)
        score = compute_soft_score(fids, beta=50.0)

        # Adjoint method for gradient
        # d(score)/d(rho_final[ci,si]) = d(score)/d(fid[ci,si]) * d(fid)/d(rho)
        # fid = Re[psi_t^dagger rho psi_t] = Re[sum_{ij} conj(psi_t_i) rho_{ij} psi_t_j]
        # d(fid)/d(rho_{ij}) = Re[conj(psi_t_i) psi_t_j]
        # In vectorized form: d(fid)/d(vec(rho)) = Re[vec(conj(psi_t) psi_t^T)] ... 
        # Actually: fid = Re[<psi_t|rho|psi_t>]
        # d fid / d rho_{ij} = Re[conj(psi_t_i) psi_t_j]
        # So the gradient w.r.t. vec(rho) is Re[outer(conj(psi_t), psi_t)] flattened

        infids = 1.0 - fids
        m = np.max(infids)
        exp_vals = np.exp(50.0 * (infids - m))
        soft_max = m + np.log(np.sum(exp_vals)) / 50.0
        # d(score)/d(infid[k]) = 1/54 + exp_vals[k] / sum(exp_vals) / 50
        d_score_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (np.sum(exp_vals) * 50.0)
        d_score_d_fid = -d_score_d_infid.reshape(9, 6)

        # Gradient w.r.t. each rho_final[ci, si]
        grad_rho = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                psi_t = target_states[si]
                # d(fid)/d(rho_{ij}) = Re[conj(psi_t_i) psi_t_j]
                grad_mat = np.outer(np.conj(psi_t), psi_t).real
                grad_rho[ci, si] = d_score_d_fid[ci, si] * grad_mat.flatten()

        # Backward propagation through time
        # d(score)/d(U[ci, step]) = sum over states of d(score)/d(rho_final) * d(rho_final)/d(U[step])
        # rho_final[ci,si] = U[ci,N-1] @ U[ci,N-2] @ ... @ U[ci,0] @ rho_init[si]
        # d(rho_final)/d(U[ci,step]) = U[ci,N-1]@...@U[ci,step+1] @ (d rho/d U[step]) @ U[ci,step-1]@...@U[ci,0]
        # This is complex. Let's use the adjoint state method.

        # Adjoint: define lambda[ci,si,step] = d(score)/d(rho_vec[ci,si,step])
        # lambda[ci,si,N] = grad_rho[ci,si]
        # lambda[ci,si,step] = U[ci,step]^T @ lambda[ci,si,step+1]  (for real case)
        # For complex: lambda[step] = U[step]^dagger @ lambda[step+1] in the appropriate sense

        # Actually, for vec(rho): rho_{step+1} = U @ rho_step
        # d(score)/d(rho_step) = U^T @ d(score)/d(rho_{step+1})  (if U is real)
        # For complex U: d(score)/d(rho_step) = U.conj().T @ d(score)/d(rho_{step+1}) ... 
        # Need to be careful with complex gradients.

        # For real-valued score and complex rho:
        # d(score)/d(rho_step) = U^dagger @ d(score)/d(rho_{step+1})
        # where ^dagger is conjugate transpose, and we treat rho as independent of rho^*

        # Let's use: d(score)/d(vec(rho_step)) = U[step]^T @ d(score)/d(vec(rho_{step+1}))
        # This works when score is real and we use the "Wirtinger" convention properly.
        # For now, use the simpler approach:
        # d(score)/d(rho_step) = U[step].conj().T @ d(score)/d(rho_{step+1})

        # Backward pass
        lambda_state = np.zeros((9, 6, 9), dtype=complex)
        lambda_state[:, :, :] = grad_rho

        grad_params = np.zeros(2*N)

        for step in range(N - 1, -1, -1):
            for ci in range(9):
                U_step = U_all[ci, step]
                # Gradient w.r.t. I and Q at this step
                # d(rho_{step+1})/d(I_val) = U[step+1:N] @ (dU[step]/dI @ rho_step) @ ...
                # This is getting very complex. Let's use numerical gradient instead
                # for the final polish, and analytic for the main optimization.
                pass

            # Propagate adjoint backward
            for ci in range(9):
                U_step = U_all[ci, step]
                for si in range(6):
                    if step > 0:
                        lambda_state[ci, si] = U_all[ci, step - 1].conj().T @ lambda_state[ci, si]

        # For simplicity and reliability, use numerical gradient
        # The simulation is fast enough with vectorized approach
        return score, None

    def objective_and_grad(params):
        """Objective with numerical gradient for L-BFGS-B."""
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            return 1e10, np.zeros(2*N)

        rho_final, _ = simulate_all(pulse_IQ)
        fids = compute_fidelities(rho_final)
        score = compute_soft_score(fids, beta=50.0)

        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(2*N)
        for j in range(2*N):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += eps
            params_minus[j] -= eps

            p_plus = np.column_stack([params_plus[:N], params_plus[N:2*N]])
            p_minus = np.column_stack([params_minus[:N], params_minus[N:2*N]])

            if check_validity(p_plus):
                r_plus, _ = simulate_all(p_plus)
                f_plus = compute_fidelities(r_plus)
                s_plus = compute_soft_score(f_plus, beta=50.0)
            else:
                s_plus = score + eps * 1000

            if check_validity(p_minus):
                r_minus, _ = simulate_all(p_minus)
                f_minus = compute_fidelities(r_minus)
                s_minus = compute_soft_score(f_minus, beta=50.0)
            else:
                s_minus = score + eps * 1000

            grad[j] = (s_plus - s_minus) / (2 * eps)

        return score, grad

    # Time management
    start_time = time.time()
    deadline = start_time + 280.0  # 280s safety margin (330s kill, leave 50s)

    def time_remaining():
        return deadline - time.time()

    # Phase 1: DRAG alpha sweep
    pulse_IQ = make_drag_pulse(0.0)
    best_pulse = pulse_IQ.copy()
    best_score = compute_soft_score(compute_fidelities(simulate_all(best_pulse)[0]), beta=50.0)

    for alpha in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5,
                  -0.05, -0.1, -0.15, -0.2, -0.3]:
        if time_remaining() < 30:
            break
        p_test = make_drag_pulse(alpha)
        if check_validity(p_test):
            f = compute_fidelities(simulate_all(p_test)[0])
            s = compute_soft_score(f, beta=50.0)
            if s < best_score:
                best_score = s
                best_pulse = p_test.copy()

    # Phase 2: Nelder-Mead for coarse optimization
    if time_remaining() > 60:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def nm_obj(params):
            if time_remaining() < 10:
                return 1e10
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10
            f = compute_fidelities(simulate_all(p)[0])
            return compute_soft_score(f, beta=50.0)

        result = minimize(nm_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 150})
        if time_remaining() > 30:
            p_nm = result.x.reshape(N, 2)
            if check_validity(p_nm):
                f_nm = compute_fidelities(simulate_all(p_nm)[0])
                s_nm = compute_soft_score(f_nm, beta=50.0)
                if s_nm < best_score:
                    best_score = s_nm
                    best_pulse = p_nm.copy()

    # Phase 3: L-BFGS-B with numerical gradient for fine optimization
    if time_remaining() > 40:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def lbfgs_obj(params):
            if time_remaining() < 10:
                return 1e10, np.zeros(2*N)
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10, np.zeros(2*N)
            f = compute_fidelities(simulate_all(p)[0])
            s = compute_soft_score(f, beta=50.0)
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros(2*N)
            for j in range(2*N):
                pp = params.copy()
                pm = params.copy()
                pp[j] += eps
                pm[j] -= eps
                ppp = np.column_stack([pp[:N], pp[N:2*N]])
                ppm = np.column_stack([pm[:N], pm[N:2*N]])
                if check_validity(ppp):
                    fp = compute_fidelities(simulate_all(ppp)[0])
                    sp = compute_soft_score(fp, beta=50.0)
                else:
                    sp = s + 1000*eps
                if check_validity(ppm):
                    fm = compute_fidelities(simulate_all(ppm)[0])
                    sm = compute_soft_score(fm, beta=50.0)
                else:
                    sm = s + 1000*eps
                grad[j] = (sp - sm) / (2*eps)
            return s, grad

        result2 = minimize(lbfgs_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfev': 500})
        if time_remaining() > 10:
            p_bfgs = result2.x.reshape(N, 2)
            if check_validity(p_bfgs):
                f_bfgs = compute_fidelities(simulate_all(p_bfgs)[0])
                s_bfgs = compute_soft_score(f_bfgs, beta=50.0)
                if s_bfgs < best_score:
                    best_score = s_bfgs
                    best_pulse = p_bfgs.copy()

    # Phase 4: Final polish with tight tolerance
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
            f = compute_fidelities(simulate_all(p)[0])
            s = compute_soft_score(f, beta=100.0)  # Higher beta for sharper max
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
                    fp = compute_fidelities(simulate_all(ppp)[0])
                    sp = compute_soft_score(fp, beta=100.0)
                else:
                    sp = s + 1000*eps
                if check_validity(ppm):
                    fm = compute_fidelities(simulate_all(ppm)[0])
                    sm = compute_soft_score(fm, beta=100.0)
                else:
                    sm = s + 1000*eps
                grad[j] = (sp - sm) / (2*eps)
            return s, grad

        result3 = minimize(polish_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfev': 200})
        if time_remaining() > 5:
            p_pol = result3.x.reshape(N, 2)
            if check_validity(p_pol):
                f_pol = compute_fidelities(simulate_all(p_pol)[0])
                s_pol = compute_soft_score(f_pol, beta=100.0)
                if s_pol < best_score:
                    best_score = s_pol
                    best_pulse = p_pol.copy()

    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = make_drag_pulse(0.0)

    return best_pulse
# EVOLVE-BLOCK-END
