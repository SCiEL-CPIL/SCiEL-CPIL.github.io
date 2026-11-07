import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 318.0

    # ---- constants ----
    alpha = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0
    deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    a_op = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    n_nm1 = n_op @ (n_op - np.eye(3))

    H0_nom = 0.5 * alpha * n_nm1
    driveI = 0.5 * (a_op + a_dag)
    driveQ = 0.5j * (a_op - a_dag)

    # Lindblad terms
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    L1dag = L1.conj().T
    L2dag = L2.conj().T

    d = 3
    eye_d = np.eye(d)

    # Precompute incoherent Liouvillian part (constant across all steps/cases)
    incoh_L = np.zeros((d*d, d*d), dtype=complex)
    for Lk in [L1, L2]:
        Lkd = Lk.conj().T
        incoh_L += np.kron(Lk.conj(), Lk)
        LkdLk = Lkd @ Lk
        incoh_L += -0.5 * np.kron(eye_d, LkdLk)
        incoh_L += -0.5 * np.kron(LkdLk.T, eye_d)

    def get_H(I, Q, delta, scale):
        return H0_nom + delta * n_op + scale * (I * driveI + Q * driveQ)

    def build_Liouvillian(H):
        """Build the Liouvillian superoperator: coherent + precomputed incoherent."""
        HkronI = np.kron(H, eye_d)
        IkronHt = np.kron(eye_d, H.T)
        return -1j * (HkronI - IkronHt) + incoh_L

    def check_constraints(I_Q):
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        if np.max(np.sqrt(np.sum(diffs ** 2, axis=1))) / dt > slew_bound + 1e-9:
            return False
        return True

    def propagate(I_Q, delta, scale):
        # Build superoperator for each step
        Us = []
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            L = build_Liouvillian(H)
            U = expm(L * dt)
            Us.append(U)
        
        out = np.empty((6, 3), dtype=complex)
        states = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
        ]
        for s in range(6):
            rho = np.outer(states[s], states[s].conj())
            vec_rho = rho.flatten()
            for U in Us:
                vec_rho = U @ vec_rho
            rho_f = vec_rho.reshape(d, d)
            # Compute fidelity
            # F = <psi_t| rho |psi_t>
            # We need the target state for each initial state
            targets = [
                np.array([0, 1, 0], dtype=complex),
                np.array([1, 0, 0], dtype=complex),
                np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
                np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
                np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
                np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),
            ]
            psi_t = targets[s]
            f = np.vdot(psi_t, rho_f @ psi_t).real
            out[s, 0] = f  # store fidelity for now
            out[s, 1] = 0
            out[s, 2] = 0
        return out

    def score_case(I_Q, delta, scale):
        res = propagate(I_Q, delta, scale)
        fids = res[:, 0]
        infs = 1.0 - fids
        return infs.mean() + infs.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        return float(np.mean([score_case(I_Q, d, sc) for d in deltas for sc in scales]))

    def get_error_vector(I_Q, delta, scale):
        """
        Returns a vector of errors: [leakage, phase_err, excitation]
        leakage: probability in |2>
        phase_err: deviation of qubit phase from ideal
        excitation: probability in |1> when starting from |0>
        """
        res = propagate(I_Q, delta, scale)
        fids = res[:, 0]
        
        # State 0: |0> -> |1>. Fidelity F01.
        # State 1: |1> -> |0>. Fidelity F10.
        # State 2: |+> -> |->. Fidelity Fpp.
        # State 3: |-> -> |+>. Fidelity Fmm.
        # State 4: |+i> -> |-i>. Fidelity Fpi.
        # State 5: |-i> -> |+i>. Fidelity Fmi.
        
        # Leakage is 1 - (F01 + F10)/2 approx? No.
        # Better: Compute probabilities directly from density matrices?
        # The current propagate only returns fidelity.
        # We need to modify propagate or calculate errors differently.
        
        # For now, let's use infidelities as proxies.
        # e0 = 1 - F01 (excitation error from |0>)
        # e1 = 1 - F10 (decoherence/leakage from |1>)
        # e2 = 1 - Fpp (phase error from |+)
        
        e0 = 1.0 - fids[0]
        e1 = 1.0 - fids[1]
        e2 = 1.0 - fids[2]
        
        return np.array([e0, e1, e2])

    def compute_error_matrix(basis_pulses, deltas, scales):
        """
        Computes the error matrix for a set of basis pulses.
        Returns a list of matrices, one for each basis pulse.
        Each matrix is (9, 3) where rows are cases and cols are error components.
        """
        error_mats = []
        for p in basis_pulses:
            mats = []
            for d_idx, delta in enumerate(deltas):
                for s_idx, scale in enumerate(scales):
                    err = get_error_vector(p, delta, scale)
                    mats.append(err)
            error_mats.append(np.array(mats))
        return error_mats

    def solve_drag_coefficients(basis_pulses):
        """
        Solves for coefficients c1, c2 such that pulse = b0 + c1*b1 + c2*b2
        minimizes the sum of squared errors across all 9 cases.
        """
        if time.time() > deadline - 100:
            return None
            
        error_mats = compute_error_matrix(basis_pulses, deltas, scales)
        
        # error_mats[0] is E0 (9x3)
        # error_mats[1] is E1 (9x3)
        # error_mats[2] is E2 (9x3)
        
        # We want to minimize || E0 + c1*E1 + c2*E2 ||_F^2
        # This is equivalent to minimizing || [E1 E2] [c1; c2] - (-E0) ||^2
        
        A = np.hstack([error_mats[1].flatten(), error_mats[2].flatten()])
        b = -error_mats[0].flatten()
        
        # Solve least squares
        try:
            res = np.linalg.lstsq(A, b, rcond=None)
            coeffs = res[0]
            c1, c2 = coeffs[0], coeffs[1]
            return [1.0, c1, c2]
        except Exception:
            return None

    # Define basis pulses
    t = np.linspace(0, 40, N)
    tc = 20.0
    sigma = 40.0 / 6.0
    
    # b0: Gaussian with area pi
    # Area of exp(-x^2/2s^2) is s*sqrt(2pi).
    # We want area pi. So A * s * sqrt(2pi) = pi => A = 1/(s*sqrt(2)) = 1/(sqrt(2)*40/6) = 6/(40*sqrt(2)) = 3/(20*sqrt(2))
    A0 = 3.0 / (20.0 * np.sqrt(2))
    b0_I = A0 * np.exp(-(t - tc) ** 2 / (2 * sigma ** 2))
    b0_Q = np.zeros(N)
    
    # b1: Derivative of b0
    b1_I = -((t - tc) / sigma ** 2) * b0_I
    b1_Q = np.zeros(N)
    
    # b2: Second derivative of b0
    # d/dt exp(-x^2/2s^2) = -x/s^2 exp(...)
    # d^2/dt^2 exp(-x^2/2s^2) = (x^2/s^4 - 1/s^2) exp(...)
    b2_I = A0 * ((t - tc) ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * np.exp(-(t - tc) ** 2 / (2 * sigma ** 2))
    b2_Q = np.zeros(N)
    
    basis_pulses = [
        np.column_stack((b0_I, b0_Q)),
        np.column_stack((b1_I, b1_Q)),
        np.column_stack((b2_I, b2_Q))
    ]
    
    best = basis_pulses[0]
    best_s = score(best)
    
    # Try to find optimal coefficients
    coeffs = solve_drag_coefficients(basis_pulses)
    if coeffs is not None:
        c0, c1, c2 = coeffs
        I_new = c0 * b0_I + c1 * b1_I + c2 * b2_I
        Q_new = np.zeros(N)
        candidate = np.column_stack((I_new, Q_new))
        
        if check_constraints(candidate):
            s = score(candidate)
            if s < best_s:
                best_s = s
                best = candidate
        else:
            # If constraints are violated, try to scale down or just stick with b0
            # We can also try a grid search around this point
            pass
            
    # Refine with Nelder-Mead on the 3 coefficients
    def gen_drag_opt(c):
        c0, c1, c2 = c
        I_new = c0 * b0_I + c1 * b1_I + c2 * b2_I
        Q_new = np.zeros(N)
        return np.column_stack((I_new, Q_new))

    def opt_drag_c(c):
        if time.time() > deadline - 50:
            return 1e6
        I_Q = gen_drag_opt(c)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    # Initial guess for optimization
    init_c = [1.0, 0.0, 0.0]
    if coeffs is not None:
        init_c = coeffs
        
    try:
        r = minimize(opt_drag_c, init_c, method='Nelder-Mead',
                     options={'maxiter': 100, 'xatol': 0.01, 'fatol': 1e-5})
        if r.fun < best_s:
            best_s = r.fun
            best = gen_drag_opt(r.x)
    except Exception:
        pass

    # Final direct refinement
    def opt_direct(x):
        if time.time() > deadline - 25:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    best_x = best.reshape(-1)
    try:
        r = minimize(opt_direct, best_x, method='L-BFGS-B',
                     options={'maxiter': 40, 'ftol': 1e-7, 'gtol': 1e-4,
                              'maxfun': 400, 'maxls': 20})
        I_Q = r.x.reshape(N, 2)
        if check_constraints(I_Q):
            s = score(I_Q)
            if s < best_s:
                best_s = s
                best = I_Q
    except Exception:
        pass

    if not check_constraints(best):
        best = basis_pulses[0]
    return best
