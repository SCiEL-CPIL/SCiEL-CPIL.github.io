import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    t0 = time.time()
    deadline = t0 + 325.0

    alpha = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0

    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    sqrt2 = np.sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([1, -1, 0], dtype=complex) / sqrt2,
        np.array([1, 1j, 0], dtype=complex) / sqrt2,
        np.array([1, -1j, 0], dtype=complex) / sqrt2
    ]
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([-1, 1, 0], dtype=complex) / sqrt2,
        np.array([1j, 1, 0], dtype=complex) / sqrt2,
        np.array([-1j, 1, 0], dtype=complex) / sqrt2
    ]

    a_op = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    I3 = np.eye(3)

    H_anharm = 0.5 * alpha * (n_op @ (n_op - I3))

    # Build dissipator superoperator (vectorized via commutators)
    L_diss = np.zeros((9, 9), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    for L_op in [L1, L2]:
        Ld = L_op.conj().T
        LdL = Ld @ L_op
        # vec(L rho)
        L_r = np.kron(L_op, I3)
        # vec(rho Ld)
        r_Ld = np.kron(I3, Ld.T)
        # vec(LdL rho)
        LdL_r = np.kron(LdL, I3)
        # vec(rho LdL)
        r_LdL = np.kron(I3, LdL.T)
        L_diss += L_r - 0.5 * LdL_r - 0.5 * r_LdL

    # Precompute Liouvillian parts for each (delta, scale)
    liouv_parts = {}
    for d in deltas:
        for sc in scales:
            H0 = d * n_op + H_anharm
            HI = 0.5 * sc * (a_op + a_dag)
            HQ = 0.5j * sc * (a_op - a_dag)
            # H superoperator: -1j (kron(H, I) - kron(I, H^T))
            H0s = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
            HIs = -1j * (np.kron(HI, I3) - np.kron(I3, HI.T))
            HQs = -1j * (np.kron(HQ, I3) - np.kron(I3, HQ.T))
            liouv_parts[(d, sc)] = (H0s + L_diss, HIs, HQs)

    # Precompute initial density matrices
    rho_vecs_init = np.array([np.outer(s, s.conj()).flatten() for s in states])  # (6, 9)

    def simulate_pulse(IQ, delta, scale):
        L_base, HI_s, HQ_s = liouv_parts[(delta, scale)]
        # Compute propagators
        # Use a fast approach: precompute all exponentials
        rho_vecs = rho_vecs_init.copy()
        for i in range(N):
            I_val = IQ[i, 0]
            Q_val = IQ[i, 1]
            L_total = L_base + I_val * HI_s + Q_val * HQ_s
            U = expm(L_total * dt)
            rho_vecs = (U @ rho_vecs.T).T
        # Compute infidelity
        total_inf = 0.0
        for idx in range(6):
            rho = rho_vecs[idx].reshape(3, 3)
            tgt = targets[idx]
            proj = np.outer(tgt, tgt.conj())
            fid = np.real(np.trace(rho @ proj))
            fid = np.clip(fid, 0, 1)
            total_inf += (1.0 - fid)
        return total_inf / 6.0

    def calc_score(IQ):
        case_scores = []
        for d in deltas:
            for sc in scales:
                case_scores.append(simulate_pulse(IQ, d, sc))
        return np.mean(case_scores) + np.max(case_scores)

    def check_constraints(IQ):
        amps = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-12:
            return False
        if np.sqrt(IQ[0,0]**2 + IQ[0,1]**2) / dt > slew_bound + 1e-12:
            return False
        if np.sqrt(IQ[-1,0]**2 + IQ[-1,1]**2) / dt > slew_bound + 1e-12:
            return False
        diffs = np.diff(IQ, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-12:
            return False
        return True

    def generate_pulse(sigma, eta, center=20.0, shape='gauss'):
        t = np.linspace(0, 40, N)
        if shape == 'gauss':
            A = np.sqrt(np.pi / 2) / sigma
            I = A * np.exp(-(t - center)**2 / (2 * sigma**2))
            dI_dt = -(t - center) / (sigma**2) * I
        elif shape == 'sech':
            A = np.sqrt(np.pi / 2) / sigma
            I = A / np.cosh((t - center) / sigma)
            dI_dt = -np.tanh((t - center) / sigma) / sigma * I
        else:
            A = np.sqrt(np.pi / 2) / sigma
            I = A * np.exp(-(t - center)**2 / (2 * sigma**2))
            dI_dt = -(t - center) / (sigma**2) * I
        Q = -eta * dI_dt
        return np.column_stack((I, Q))

    # Seed pulse
    sigma_seed = 40.0 / 6.0
    IQ_seed = generate_pulse(sigma_seed, 0.0)
    if not check_constraints(IQ_seed):
        # Scale down if needed
        max_amp = np.max(np.sqrt(IQ_seed[:, 0]**2 + IQ_seed[:, 1]**2))
        if max_amp > amp_bound:
            IQ_seed *= amp_bound / max_amp
        max_slew = max(
            np.sqrt(IQ_seed[0,0]**2 + IQ_seed[0,1]**2) / dt,
            np.sqrt(IQ_seed[-1,0]**2 + IQ_seed[-1,1]**2) / dt,
            np.max(np.sqrt(np.sum(np.diff(IQ_seed, axis=0)**2, axis=1))) / dt
        )
        if max_slew > slew_bound:
            IQ_seed *= slew_bound / max_slew

    best_IQ = IQ_seed.copy()
    best_score = calc_score(best_IQ)

    # Try DRAG seeds
    for sig_try in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
        for eta_try in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
            if time.time() > deadline - 5:
                break
            IQ_try = generate_pulse(sig_try, eta_try)
            if check_constraints(IQ_try):
                s_try = calc_score(IQ_try)
                if s_try < best_score:
                    best_score = s_try
                    best_IQ = IQ_try.copy()

    # Also try sech shape
    for sig_try in [5.0, 6.0, 7.0, 8.0, 10.0]:
        for eta_try in [0.0, 1.0, 2.0, 3.0, 4.0]:
            if time.time() > deadline - 5:
                break
            IQ_try = generate_pulse(sig_try, eta_try, shape='sech')
            if check_constraints(IQ_try):
                s_try = calc_score(IQ_try)
                if s_try < best_score:
                    best_score = s_try
                    best_IQ = IQ_try.copy()

    # Objective for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e10
        sigma, eta = params
        if sigma <= 0.5 or sigma > 50:
            return 1e10
        IQ = generate_pulse(sigma, eta)
        if not check_constraints(IQ):
            return 1e10
        return calc_score(IQ)

    # Phase 1: Grid search
    grid_deadline = t0 + 60.0
    sigmas_grid = [5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
    etas_grid = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    best_params = [sigma_seed, 0.0]

    for s in sigmas_grid:
        for e in etas_grid:
            if time.time() > grid_deadline:
                break
            score = objective([s, e])
            if score < best_score:
                best_score = score
                best_IQ = generate_pulse(s, e)
                best_params = [s, e]
        if time.time() > grid_deadline:
            break

    # Phase 2: Nelder-Mead local optimization from multiple starts
    local_deadline = t0 + 200.0
    nm_starts = [
        best_params,
        [8.0, 1.5],
        [10.0, 2.0],
        [7.0, 1.0],
        [9.0, 2.5],
        [6.0, 2.0],
        [11.0, 1.0],
        [12.0, 1.5],
        [8.0, 3.0],
        [10.0, 1.0],
    ]

    unique_starts = []
    for sp in nm_starts:
        if sp not in unique_starts:
            unique_starts.append(sp)

    for sp in unique_starts:
        if time.time() > local_deadline:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score and check_constraints(generate_pulse(res.x[0], res.x[1])):
                best_score = res.fun
                best_IQ = generate_pulse(res.x[0], res.x[1])
                best_params = [res.x[0], res.x[1]]
        except Exception:
            pass

    # Phase 3: Final polish
    if time.time() < deadline - 10:
        final_starts = [
            best_params,
            [best_params[0] + 1, best_params[1] - 0.5],
            [best_params[0] - 1, best_params[1] + 0.5],
            [8.0, 1.5],
            [9.0, 2.0],
            [10.0, 2.5],
            [7.0, 1.2],
        ]
        unique_final = []
        for sp in final_starts:
            if sp not in unique_final:
                unique_final.append(sp)
        for sp in unique_final:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, sp, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if res.fun < best_score and check_constraints(generate_pulse(res.x[0], res.x[1])):
                    best_score = res.fun
                    best_IQ = generate_pulse(res.x[0], res.x[1])
            except Exception:
                pass

    # Final constraint check
    if not check_constraints(best_IQ):
        best_IQ = IQ_seed.copy()

    return best_IQ
