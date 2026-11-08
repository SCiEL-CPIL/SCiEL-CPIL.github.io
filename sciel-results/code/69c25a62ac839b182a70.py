import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 325.0

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

    d = 3
    eye_d = np.eye(d)

    # Precompute incoherent Liouvillian part (constant)
    incoh_L = np.zeros((d * d, d * d), dtype=complex)
    for Lk in [L1, L2]:
        Lkd = Lk.conj().T
        incoh_L += np.kron(Lk.conj(), Lk)
        LkdLk = Lkd @ Lk
        incoh_L += -0.5 * np.kron(eye_d, LkdLk)
        incoh_L += -0.5 * np.kron(LkdLk.T, eye_d)

    def get_H(I, Q, delta, scale):
        return H0_nom + delta * n_op + scale * (I * driveI + Q * driveQ)

    # Precompute coherent superoperator components
    coh_base = -1j * (np.kron(H0_nom, eye_d) - np.kron(eye_d, H0_nom.T))
    coh_delta = -1j * (np.kron(n_op, eye_d) - np.kron(eye_d, n_op.T))
    coh_I = -1j * (np.kron(driveI, eye_d) - np.kron(eye_d, driveI.T))
    coh_Q = -1j * (np.kron(driveQ, eye_d) - np.kron(eye_d, driveQ.T))

    # For each (delta, scale) pair, precompute base Liouvillian and drive coefficients
    L_bases = np.zeros((9, d * d, d * d), dtype=complex)
    L_I_scales = np.zeros((9, d * d, d * d), dtype=complex)
    L_Q_scales = np.zeros((9, d * d, d * d), dtype=complex)
    case_idx = 0
    for d_val in deltas:
        for sc_val in scales:
            L_bases[case_idx] = coh_base + d_val * coh_delta + incoh_L
            L_I_scales[case_idx] = sc_val * coh_I
            L_Q_scales[case_idx] = sc_val * coh_Q
            case_idx += 1

    def check_constraints(I_Q):
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        if np.max(np.sqrt(np.sum(diffs ** 2, axis=1))) / dt > slew_bound + 1e-9:
            return False
        return True

    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),
    ]

    # Precompute fidelity measurement matrices
    M_mats = []
    for s in range(6):
        M = np.outer(targets[s].conj(), targets[s])
        M_mats.append(M.flatten())
    M_mats = np.array(M_mats)  # shape (6, 9)

    # Precompute initial density matrix vectors
    rho_vecs_init = np.array([np.outer(states[s], states[s].conj()).flatten() for s in range(6)])
    rho_vecs_init = rho_vecs_init.T  # shape (9, 6)

    def propagate_all(I_Q, delta, scale):
        case_idx = deltas.index(delta) * 3 + scales.index(scale)
        rho_vecs = rho_vecs_init.copy()
        for i in range(N):
            L = L_bases[case_idx] + I_Q[i, 0] * L_I_scales[case_idx] + I_Q[i, 1] * L_Q_scales[case_idx]
            U = expm(L * dt)
            rho_vecs = U @ rho_vecs
        fids = np.real(np.sum(M_mats * rho_vecs.T, axis=1))
        return fids

    def score_case(I_Q, delta, scale):
        res = propagate_all(I_Q, delta, scale)
        infs = 1.0 - res
        return infs.mean() + infs.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        total = 0.0
        for d in deltas:
            for sc in scales:
                total += score_case(I_Q, d, sc)
        return total / 9.0

    # ---- DRAG pulse generator with flexible shape ----
    def gen_drag(p):
        """p = [A, sI, eta, offset, width_q, amp_q, center_q]"""
        A, sI, eta, offset, width_q, amp_q, center_q = p
        t = np.linspace(0, 40, N)
        tc = 20.0
        I = A * np.exp(-(t - tc) ** 2 / (2 * sI ** 2))
        dI = -(t - tc) / sI ** 2 * I
        Q = eta * dI + amp_q * np.exp(-(t - center_q - offset) ** 2 / (2 * width_q ** 2))
        return np.column_stack((I, Q))

    def gen_drag2(p):
        """Two-Gaussian Q for spectral nulling: p = [A, sI, eta, Aq1, wq1, cq1, Aq2, wq2, cq2]"""
        A, sI, eta, Aq1, wq1, cq1, Aq2, wq2, cq2 = p
        t = np.linspace(0, 40, N)
        tc = 20.0
        I = A * np.exp(-(t - tc) ** 2 / (2 * sI ** 2))
        dI = -(t - tc) / sI ** 2 * I
        Q = (eta * dI +
             Aq1 * np.exp(-(t - tc - cq1) ** 2 / (2 * wq1 ** 2)) +
             Aq2 * np.exp(-(t - tc - cq2) ** 2 / (2 * wq2 ** 2)))
        return np.column_stack((I, Q))

    def seed_pulse():
        s = 40.0 / 6.0
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi / 2) / s) * np.exp(-(t - 20.0) ** 2 / (2 * s ** 2))
        return np.column_stack((I, np.zeros(N)))

    best = seed_pulse()
    best_s = score(best)

    # ---- Coarse grid: DRAG family ----
    for A in [0.22, 0.26, 0.30, 0.34, 0.37]:
        for sI in [4.5, 5.5, 6.5, 7.5, 9.0]:
            for eta in [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
                if time.time() > deadline - 100:
                    break
                I_Q = gen_drag([A, sI, eta, 0.0, 5.0, 0.0, 20.0])
                if not check_constraints(I_Q):
                    continue
                s = score(I_Q)
                if s < best_s:
                    best_s, best = s, I_Q

    # ---- Coarse grid: two-Gaussian Q (spectral nulling) ----
    for A in [0.26, 0.30, 0.34]:
        for sI in [5.0, 6.0, 7.0]:
            for eta in [-2.0, -1.0, 0.0]:
                for Aq in [0.03, 0.06, 0.10]:
                    if time.time() > deadline - 80:
                        break
                    I_Q = gen_drag2([A, sI, eta, Aq, 3.0, 4.0, -Aq, 3.0, -4.0])
                    if not check_constraints(I_Q):
                        continue
                    s = score(I_Q)
                    if s < best_s:
                        best_s, best = s, I_Q

    # ---- Nelder-Mead refinement on DRAG family ----
    def opt_drag(p):
        if time.time() > deadline - 50:
            return 1e6
        I_Q = gen_drag(p)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    inits_drag = [
        [0.28, 5.5, -1.5, 0.0, 5.0, 0.0, 20.0],
        [0.30, 6.0, -1.0, 0.0, 5.0, 0.0, 20.0],
        [0.26, 5.0, -2.0, 0.0, 5.0, 0.0, 20.0],
        [0.32, 6.5, -1.0, 0.0, 5.0, 0.0, 20.0],
        [0.30, 5.5, -1.5, 0.0, 4.0, 0.05, 24.0],
        [0.30, 5.5, -1.5, 0.0, 4.0, -0.05, 16.0],
    ]
    for init in inits_drag:
        if time.time() > deadline - 50:
            break
        try:
            r = minimize(opt_drag, init, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 0.005, 'fatol': 1e-8})
            if r.fun < best_s:
                I_Q = gen_drag(r.x)
                if check_constraints(I_Q):
                    best_s, best = r.fun, I_Q
        except Exception:
            pass

    # ---- Nelder-Mead on two-Gaussian Q ----
    def opt_drag2(p):
        if time.time() > deadline - 30:
            return 1e6
        I_Q = gen_drag2(p)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    inits_drag2 = [
        [0.30, 5.5, -1.5, 0.05, 3.0, 4.0, -0.05, 3.0, -4.0],
        [0.30, 6.0, -1.0, 0.08, 3.5, 5.0, -0.08, 3.5, -5.0],
        [0.28, 5.0, -2.0, 0.06, 2.5, 3.0, -0.06, 2.5, -3.0],
        [0.32, 6.0, -1.0, 0.04, 4.0, 6.0, -0.04, 4.0, -6.0],
    ]
    for init in inits_drag2:
        if time.time() > deadline - 30:
            break
        try:
            r = minimize(opt_drag2, init, method='Nelder-Mead',
                         options={'maxiter': 250, 'xatol': 0.005, 'fatol': 1e-8})
            if r.fun < best_s:
                I_Q = gen_drag2(r.x)
                if check_constraints(I_Q):
                    best_s, best = r.fun, I_Q
        except Exception:
            pass

    # ---- Direct 80-step L-BFGS-B refinement ----
    def opt_direct(x):
        if time.time() > deadline - 10:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    best_x = best.reshape(-1)
    for k in range(6):
        if time.time() > deadline - 10:
            break
        try:
            r = minimize(opt_direct, best_x, method='L-BFGS-B',
                         options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-7,
                                  'maxfun': 1000, 'maxls': 50})
            I_Q = r.x.reshape(N, 2)
            if check_constraints(I_Q):
                s = score(I_Q)
                if s < best_s:
                    best_s, best, best_x = s, I_Q, r.x
            else:
                break
        except Exception:
            break

    # ---- Final polish ----
    def opt_polish(x):
        if time.time() > deadline - 5:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    best_x = best.reshape(-1)
    try:
        r = minimize(opt_polish, best_x, method='L-BFGS-B',
                     options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-9,
                              'maxfun': 300, 'maxls': 40})
        I_Q = r.x.reshape(N, 2)
        if check_constraints(I_Q):
            s = score(I_Q)
            if s < best_s:
                best_s, best = s, I_Q
    except Exception:
        pass

    # Final constrained cleanup
    if not check_constraints(best):
        I_Q = best.copy()
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        mask = amps > amp_bound
        if np.any(mask):
            scale = amp_bound / amps[mask]
            I_Q[mask, 0] *= scale
            I_Q[mask, 1] *= scale
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        for i in range(len(p) - 1):
            diff = p[i + 1] - p[i]
            nd = np.sqrt(np.sum(diff ** 2))
            max_step = slew_bound * dt
            if nd > max_step:
                scale = max_step / nd
                p[i + 1] = p[i] + diff * scale
        I_Q = p[1:-1]
        if check_constraints(I_Q):
            best = I_Q
        else:
            best = seed_pulse()
    return best
