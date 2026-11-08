"""Robust X_pi control pulse: piecewise-constant unitary evolution + B-spline + adjoint L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # --- Constants ---
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    total_time = N * dt  # 40 ns

    # --- 3-level operators ---
    a_op = np.zeros((3, 3), dtype=complex)
    a_op[0, 1] = 1.0
    a_op[1, 2] = np.sqrt(2.0)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op  # diag(0,1,2)
    I3 = np.eye(3)

    # Static part of Hamiltonian (detuning-independent, anharmonicity only)
    H_static = 0.5 * anharmonicity * (n_op @ n_op - n_op)  # 0.5*anharm*n*(n-1)

    # Drive terms
    # H_drive = 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    # = 0.5*scale*(I*(a+a_dag) + i*Q*a - i*Q*a_dag)
    # Let's define:
    #   sigma_x_part = a + a_dag  (real symmetric)
    #   sigma_y_part = a - a_dag  (imaginary antisymmetric)
    # H_drive = 0.5*scale*(I*sigma_x_part + i*Q*sigma_y_part)

    sigma_x = a_op + a_dag  # 3x3
    sigma_y = a_op - a_dag  # 3x3 (purely imaginary, antisymmetric)

    def build_unitary(I_val, Q_val, detuning, scale):
        """Build the unitary for one step."""
        H = H_static + detuning * n_op + 0.5 * scale * (I_val * sigma_x + 1j * Q_val * sigma_y)
        # H is Hermitian (check: sigma_x is Hermitian, i*sigma_y = i*(a-a_dag) is Hermitian since a-a_dag is anti-Hermitian)
        # Actually: a-a_dag is anti-Hermitian, so i*(a-a_dag) is Hermitian. Good.
        # Use scipy.linalg.expm for 3x3
        from scipy.linalg import expm
        return expm(-1j * dt * H)

    def simulate_pulse(pulse_IQ, detuning, scale):
        """Simulate piecewise-constant unitary evolution. Returns final rho (3x3) and list of intermediate U's."""
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        Us = []
        for i in range(N):
            U_i = build_unitary(pulse_IQ[i, 0], pulse_IQ[i, 1], detuning, scale)
            Us.append(U_i)
            rho = U_i @ rho @ U_i.conj().T
        return rho, Us

    def compute_score(pulse_IQ, detunings, scales, states_test):
        """Compute score = mean + max infidelity over 9 cases x 6 states."""
        infidelities = []
        for det in detunings:
            for scale in scales:
                rho_final, _ = simulate_pulse(pulse_IQ, det, scale)
                for state in states_test:
                    # Target: X_pi applied to state
                    target = np.zeros(3, dtype=complex)
                    target[0] = -1j * state[1]
                    target[1] = -1j * state[0]
                    target[2] = state[2]
                    psi = target.flatten()
                    fid = np.real(np.conj(psi) @ rho_final @ psi)
                    infidelities.append(1.0 - fid)
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)

    def compute_gradient(pulse_IQ, detunings, scales, states_test):
        """
        Compute gradient of score w.r.t. pulse_IQ (80,2) using adjoint method.
        For each (det, scale) case, we track the adjoint (costate) lambda_i = dC/d(rho_i) as a 3x3 matrix.
        """
        grad_I = np.zeros(N)
        grad_Q = np.zeros(N)
        n_inf = 9 * 6  # 54 infidelities
        weights = np.ones(n_inf) / n_inf

        # First pass: compute all infidelities to find max
        all_infs = []
        all_rhos = []  # list of (rho_final, Us) for each case
        for det in detunings:
            for scale in scales:
                rho_final, Us = simulate_pulse(pulse_IQ, det, scale)
                all_rhos.append((rho_final, Us, det, scale))
                for state in states_test:
                    target = np.zeros(3, dtype=complex)
                    target[0] = -1j * state[1]
                    target[1] = -1j * state[0]
                    target[2] = state[2]
                    psi = target.flatten()
                    fid = np.real(np.conj(psi) @ rho_final @ psi)
                    all_infs.append(1.0 - fid)

        all_infs = np.array(all_infs)
        max_idx = np.argmax(all_infs)
        weights[max_idx] += 1.0  # extra weight for max

        # Second pass: for each case, compute adjoint and accumulate gradient
        case_idx = 0
        for det in detunings:
            for scale in scales:
                rho_final, Us, det_c, scale_c = all_rhos[case_idx]

                # Build the cost gradient w.r.t. rho_final
                # C_case = sum_j w_{case,j} * (1 - <psi_j|rho|psi_j>)
                # dC/d(rho) = -sum_j w_{case,j} * |psi_j><psi_j|
                dC_drho = np.zeros((3, 3), dtype=complex)
                for j in range(6):
                    state = states_test[j]
                    target = np.zeros(3, dtype=complex)
                    target[0] = -1j * state[1]
                    target[1] = -1j * state[0]
                    target[2] = state[2]
                    psi = target.flatten()
                    w_j = weights[case_idx * 6 + j]
                    dC_drho -= w_j * np.outer(psi, psi.conj())

                # Adjoint: lambda_N = dC/d(rho_N)
                # rho_{i+1} = U_i @ rho_i @ U_i^dag
                # d(rho_{i+1}) = U_i @ d(rho_i) @ U_i^dag
                # dC = Tr(dC_drho_{i+1} @ d(rho_{i+1}))
                #     = Tr(dC_drho_{i+1} @ U_i @ d(rho_i) @ U_i^dag)
                #     = Tr(U_i^dag @ dC_drho_{i+1} @ U_i @ d(rho_i))
                # So lambda_i = U_i^dag @ lambda_{i+1} @ U_i
                # And dC/d(I_i) = Tr(lambda_{i+1} @ dU_i/dI_i @ rho_i @ U_i^dag) + Tr(U_i @ rho_i @ dU_i^dag/dI_i @ lambda_{i+1})
                # But since U_i depends on I_i, Q_i, we need dU_i/dI_i.
                #
                # Alternative: use the fact that for a unitary U = expm(-i*dt*H),
                # dU/dI = -i*dt * integral_0^1 expm(-i*dt*H*(1-s)) * (dH/dI) * expm(-i*dt*H*s) ds
                # This is expensive. Instead, use the first-order perturbation:
                # dU ≈ -i*dt*H*U + ... no, that's not right.
                #
                # Better approach: since we're doing piecewise-constant, the gradient w.r.t. pulse[i]
                # comes from U_i. We can compute dU_i/dI_i using the formula:
                # dU/dI = -i * dt * sum_{k=0}^{M} ... (too expensive)
                #
                # Simpler: use finite-difference-free adjoint with the identity:
                # For rho_{i+1} = U_i rho_i U_i^dag, treating U_i as a function of (I_i, Q_i):
                # dC/dI_i = Re[Tr(lambda_{i+1} @ (dU_i/dI_i) @ rho_i @ U_i^dag)]
                #          + Re[Tr(U_i @ rho_i @ (dU_i/dI_i)^dag @ lambda_{i+1})]
                #
                # To avoid computing dU/dI explicitly, note that:
                # dU/dI = -i*dt * ∫_0^1 U(1-s) (dH/dI) U(s) ds  where U(s) = expm(-i*dt*H*s)
                # For a 3x3 matrix, we can approximate this integral with a few quadrature points.
                # But even better: since dt is small (0.5 ns) and the Hamiltonian is nearly constant,
                # we can use the approximation dU/dI ≈ -i*dt*H*U * (dH/dI)/(something)... 
                #
                # Actually, the cleanest approach for piecewise-constant:
                # Let's just compute the gradient by propagating forward and backward,
                # and at each step compute dU/dI using a 3-point quadrature on the integral.
                #
                # For 3x3 matrices, expm is fast. Let's do:
                # dU/dI = -i*dt * (1/3) * [U(2/3)(dH/dI)U(1/3) + U(1/3)(dH/dI)U(2/3) + ...]
                # Actually, the exact formula is:
                # dU/dI = -i*dt * ∫_0^1 expm(-i*dt*H*(1-s)) (dH/dI) expm(-i*dt*H*s) ds
                # Use midpoint: ≈ -i*dt * expm(-i*dt*H/2) (dH/dI) expm(-i*dt*H/2)
                # This is the first-order approximation, good for small dt.

                # Propagate adjoint backward
                lambda_mat = dC_drho.copy()
                # We need rho_i for each step. Let's re-simulate forward to store rho_i.
                rho_i_list = [np.zeros((3, 3), dtype=complex)]
                rho_i_list[0][0, 0] = 1.0
                for i in range(N):
                    rho_next = Us[i] @ rho_i_list[-1] @ Us[i].conj().T
                    rho_i_list.append(rho_next)

                for i in range(N - 1, -1, -1):
                    U_i = Us[i]
                    rho_i = rho_i_list[i]
                    I_val = pulse_IQ[i, 0]
                    Q_val = pulse_IQ[i, 1]

                    # Compute dH/dI and dH/dQ
                    dH_dI = 0.5 * scale_c * sigma_x
                    dH_dQ = 0.5 * scale_c * 1j * sigma_y

                    # H for this step
                    H_i = H_static + det_c * n_op + 0.5 * scale_c * (I_val * sigma_x + 1j * Q_val * sigma_y)

                    # dU/dI ≈ -i*dt * expm(-i*dt*H_i/2) @ dH_dI @ expm(-i*dt*H_i/2)
                    from scipy.linalg import expm
                    U_half = expm(-1j * dt * H_i / 2.0)
                    dU_dI = -1j * dt * (U_half @ dH_dI @ U_half)
                    dU_dQ = -1j * dt * (U_half @ dH_dQ @ U_half)

                    # dC/dI_i = Re[Tr(lambda @ dU_dI @ rho_i @ U_i^dag)] + Re[Tr(U_i @ rho_i @ dU_dI^dag @ lambda)]
                    term1 = np.trace(lambda_mat @ dU_dI @ rho_i @ U_i.conj().T)
                    term2 = np.trace(U_i @ rho_i @ dU_dI.conj().T @ lambda_mat)
                    grad_I[i] += np.real(term1 + term2)

                    term1q = np.trace(lambda_mat @ dU_dQ @ rho_i @ U_i.conj().T)
                    term2q = np.trace(U_i @ rho_i @ dU_dQ.conj().T @ lambda_mat)
                    grad_Q[i] += np.real(term1q + term2q)

                    # Update adjoint: lambda_i = U_i^dag @ lambda_{i+1} @ U_i
                    lambda_mat = U_i.conj().T @ lambda_mat @ U_i

                case_idx += 1

        return np.column_stack([grad_I, grad_Q])

    def check_validity(pulse_IQ):
        """Check amplitude and slew constraints."""
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-10):
            return False
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            diff = np.linalg.norm(curr - prev)
            if diff / dt > slew_bound + 1e-10:
                return False
            prev = curr
        diff = np.linalg.norm(prev)
        if diff / dt > slew_bound + 1e-10:
            return False
        return True

    def enforce_constraints(pulse_IQ):
        """Project pulse to satisfy amplitude and slew constraints."""
        # Amplitude constraint
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        for i in range(N):
            if amps[i] > amplitude_bound:
                scale = amplitude_bound / amps[i]
                pulse_IQ[i] *= scale
                amps[i] = amplitude_bound

        # Slew constraint: forward pass
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i].copy()
            diff = curr - prev
            norm_diff = np.linalg.norm(diff)
            max_diff = slew_bound * dt
            if norm_diff > max_diff:
                curr = prev + diff * (max_diff / norm_diff)
            pulse_IQ[i] = curr
            prev = curr

        # Backward pass (to handle cases where forward pass isn't enough)
        prev = np.zeros(2)
        for i in range(N - 1, -1, -1):
            curr = pulse_IQ[i].copy()
            diff = prev - curr
            norm_diff = np.linalg.norm(diff)
            max_diff = slew_bound * dt
            if norm_diff > max_diff:
                curr = prev - diff * (max_diff / norm_diff)
            pulse_IQ[i] = curr
            prev = curr

        # Re-check amplitude
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        for i in range(N):
            if amps[i] > amplitude_bound:
                scale = amplitude_bound / amps[i]
                pulse_IQ[i] *= scale

        return pulse_IQ

    # --- Test states ---
    states_test = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),
    ]

    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    # --- B-spline parameterization ---
    # Use cubic B-spline with 16 knots for I and Q separately
    n_knots = 16
    knot_times = np.linspace(0, total_time, n_knots)

    def knots_to_pulse(knot_vals):
        """Convert 32 knot values (16 for I, 16 for Q) to (80,2) pulse."""
        I_knots = knot_vals[:n_knots]
        Q_knots = knot_vals[n_knots:]

        # Enforce zero at boundaries for smooth start/end
        I_knots = I_knots.copy()
        Q_knots = Q_knots.copy()
        I_knots[0] = 0.0
        I_knots[-1] = 0.0
        Q_knots[0] = 0.0
        Q_knots[-1] = 0.0

        # Build cubic spline
        cs_I = CubicSpline(knot_times, I_knots, bc_type='natural')
        cs_Q = CubicSpline(knot_times, Q_knots, bc_type='natural')

        t_eval = np.arange(N) * dt + dt / 2.0  # midpoints of each step
        I_vals = cs_I(t_eval)
        Q_vals = cs_Q(t_eval)

        pulse_IQ = np.column_stack([I_vals, Q_vals])
        pulse_IQ = enforce_constraints(pulse_IQ)
        return pulse_IQ

    def make_initial_knots():
        """Create initial DRAG-style knot values."""
        t = knot_times
        center = total_time / 2.0
        sigma = total_time / 6.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        gaussian[0] = 0.0
        gaussian[-1] = 0.0

        # Normalize area to pi
        # Approximate area using trapezoidal rule
        area = np.sum(gaussian) * (total_time / (n_knots - 1))
        if area > 1e-12:
            gaussian *= np.pi / area

        I_knots = gaussian.copy()
        # DRAG: Q proportional to derivative of I
        dI_dt = np.gradient(I_knots, total_time / (n_knots - 1))
        Q_knots = 0.15 * dI_dt  # small DRAG coefficient
        Q_knots[0] = 0.0
        Q_knots[-1] = 0.0

        return np.concatenate([I_knots, Q_knots])

    # --- Objective and gradient for L-BFGS-B ---
    def objective_and_grad(knot_vals):
        pulse_IQ = knots_to_pulse(knot_vals)
        score = compute_score(pulse_IQ, detunings, scales, states_test)

        # Gradient w.r.t. knot values: need to backprop through spline
        # For simplicity, use finite difference on knots (32 params, cheap)
        # Actually, let's use the analytic gradient w.r.t. pulse and chain through spline
        grad_pulse = compute_gradient(pulse_IQ, detunings, scales, states_test)  # (80,2)

        # Chain rule: d(score)/d(knot_j) = sum_i d(score)/d(pulse[i]) * d(pulse[i])/d(knot_j)
        # For cubic spline, d(pulse[i])/d(knot_j) is the spline basis function value
        # Approximate with finite difference on knots
        grad_knots = np.zeros(2 * n_knots)
        eps = 1e-6
        for j in range(2 * n_knots):
            kv_plus = knot_vals.copy()
            kv_plus[j] += eps
            pulse_plus = knots_to_pulse(kv_plus)
            # Compute score difference
            score_plus = compute_score(pulse_plus, detunings, scales, states_test)
            grad_knots[j] = (score_plus - score) / eps

        return score, grad_knots

    # --- Optimization ---
    start_time = time.time()
    time_limit = 280.0  # safety margin

    best_pulse = None
    best_score = np.inf

    # Multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 60:
            break

        if r == 0:
            init_knots = make_initial_knots()
        else:
            # Perturb initial knots
            init_knots = make_initial_knots()
            rng = np.random.default_rng(r * 42)
            init_knots += rng.normal(0, 0.05, size=init_knots.shape)

        def obj_wrapper(kv):
            s, g = objective_and_grad(kv)
            return s, g

        # Use L-BFGS-B with analytic gradient
        # Bounds: keep knots within [-0.5, 0.5] to ensure pulse stays within amplitude bound
        bounds = [(-0.5, 0.5)] * (2 * n_knots)

        try:
            result = minimize(
                lambda kv: objective_and_grad(kv)[0],
                init_knots,
                jac=lambda kv: objective_and_grad(kv)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
        except Exception:
            continue

        final_knots = result.x
        final_pulse = knots_to_pulse(final_knots)

        if not check_validity(final_pulse):
            final_pulse = enforce_constraints(final_pulse)

        final_score = compute_score(final_pulse, detunings, scales, states_test)

        if final_score < best_score:
            best_score = final_score
            best_pulse = final_pulse.copy()

        if time.time() - start_time > time_limit - 30:
            break

    if best_pulse is None:
        # Fallback: DRAG pulse
        t = np.arange(N) * dt
        center = total_time / 2.0
        sigma = total_time / 6.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        gaussian[0] = 0.0
        gaussian[-1] = 0.0
        area = np.sum(gaussian) * dt
        if area > 1e-12:
            gaussian *= np.pi / area
        I = gaussian.copy()
        Q = np.zeros(N)
        best_pulse = np.column_stack([I, Q])
        best_pulse = enforce_constraints(best_pulse)

    return best_pulse
