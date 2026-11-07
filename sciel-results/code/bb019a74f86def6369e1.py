import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 315.0  # Safety margin for 330s limit

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
        """Build the Liouvillian superoperator for a given Hamiltonian."""
        coh_L = -1j * (np.kron(H, eye_d) - np.kron(eye_d, H.T))
        return coh_L + incoh_L

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

    # Precompute initial density matrices for efficiency
    rho0s = [np.outer(states[s], states[s].conj()).flatten() for s in range(6)]

    def propagate(I_Q, delta, scale):
        """
        Propagate the system for a given pulse and hardware parameters.
        Returns array of 6 fidelities.
        """
        # We can pre-compute the superoperators for all steps for this specific delta/scale
        # But delta/scale varies in the 9 cases. 
        # However, I and Q are fixed. 
        # Actually, H depends on delta and scale. So we must build Liouvillians for each case.
        
        # Optimization: Instead of looping 6 states * 80 steps * 9 cases,
        # we can loop 9 cases * 80 steps, applying the superoperator to all 6 states at once?
        # The superoperator is 9x9. The states are 9x1 vectors.
        # We can stack the 6 initial density vectors into a 9x6 matrix.
        # Then U @ vec_rho becomes U @ R0, where R0 is 9x6.
        
        # Let's try this vectorization.
        
        # Build superoperators for this delta, scale
        Us = []
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            L = build_Liouvillian(H)
            Us.append(expm(L * dt))
        
        # Stack initial density matrices: shape (9, 6)
        R0 = np.column_stack(rho0s)
        
        # Apply evolution
        R = R0
        for U in Us:
            R = U @ R
            
        # Calculate fidelities
        # F = <psi_t| rho_f |psi_t> = sum_{ij} psi_t[i]^* rho_f[i,j] psi_t[j]
        # This is a bilinear form.
        # We can precompute the "target" vectors in the Liouville space?
        # Or just do it for each of the 6 states.
        
        fids = np.zeros(6)
        for s in range(6):
            rho_f = R[:, s].reshape(d, d)
            psi_t = targets[s]
            f = np.vdot(psi_t, rho_f @ psi_t).real
            fids[s] = f
        return fids

    def score_case(I_Q, delta, scale):
        fids = propagate(I_Q, delta, scale)
        infs = 1.0 - fids
        return infs.mean() + infs.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        total = 0.0
        for d in deltas:
            for sc in scales:
                total += score_case(I_Q, d, sc)
        return total / 9.0

    # ---- Parametric DRAG Family ----
    def gen_drag(p):
        A, sI, eta, sQ, shQ, c, q0 = p
        t = np.linspace(0, 40, N)
        tc = 20.0
        I = A * np.exp(-(t - tc) ** 2 / (2 * sI ** 2))
        dI = -(t - tc) / sI ** 2 * I
        Q = eta * dI + c * np.exp(-(t - tc - shQ) ** 2 / (2 * sQ ** 2)) + q0
        return np.column_stack((I, Q))

    def seed_pulse():
        # Standard DRAG seed
        # Area pi => Integral A exp(-t^2/2s^2) dt = A s sqrt(2pi) = pi
        # A = sqrt(pi/2) / s
        s = 40.0 / 6.0  # sigma ~ 6.6 ns
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi / 2) / s) * np.exp(-(t - 20.0) ** 2 / (2 * s ** 2))
        # Derivative term for DRAG
        # Q = eta * dI/dt
        # eta is typically around 1-5 depending on definition. 
        # Here dI/dt = -(t-tc)/s^2 * I
        eta = 2.0 # Initial guess
        dI = -(t - 20.0) / s ** 2 * I
        Q = eta * dI
        return np.column_stack((I, Q))

    best = seed_pulse()
    best_s = score(best)

    # ---- Coarse Grid Search on DRAG Parameters ----
    # Parameters: A, sI (sigma), eta (DRAG coeff)
    # We fix the Gaussian shape for Q to be derivative of I for simplicity in first pass,
    # or allow a separate Gaussian for Q.
    
    # Let's use the 7-parameter gen_drag but restrict search space.
    # A in [0.2, 0.4], sI in [4, 10], eta in [-5, 5]
    
    # To save time, we will do a coarse scan first.
    
    # We can speed up score() by reusing the superoperators if I_Q doesn't change,
    # but here I_Q changes in the grid search.
    
    # Optimization: 
    # The score function calls propagate 9 times. 
    # Each propagate calls expm 80 times.
    # Total expm calls per score: 720.
    # If we do 100 grid points, that's 72000 expm calls.
    # expm on 9x9 is fast (~1ms?). 72 seconds. This is too slow.
    
    # We need a faster approximation or fewer evaluations.
    # Strategy: 
    # 1. Evaluate only the nominal case (delta=0, scale=1) during the grid search.
    # 2. Only evaluate all 9 cases for the top candidates.
    
    def score_nominal(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        return score_case(I_Q, 0.0, 1.0)

    def score_full(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    # Grid Search on Nominal Case
    candidates = []
    # A, sI, eta
    A_vals = np.linspace(0.2, 0.38, 8)
    sI_vals = np.linspace(4.0, 9.0, 8)
    eta_vals = np.linspace(-3.0, 3.0, 8)
    
    for A in A_vals:
        for sI in sI_vals:
            for eta in eta_vals:
                if time.time() > deadline - 60:
                    break
                p = [A, sI, eta, 5.0, 0.0, 0.0, 0.0] # c, shQ, q0 fixed
                I_Q = gen_drag(p)
                s = score_nominal(I_Q)
                candidates.append((s, I_Q, p))
            
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
    
    if not candidates:
        return best
        
    candidates.sort(key=lambda x: x[0])
    
    # Evaluate top 10 candidates with full score
    for s_nom, I_Q, p in candidates[:10]:
        if time.time() > deadline - 30:
            break
        s_full = score_full(I_Q)
        if s_full < best_s:
            best_s = s_full
            best = I_Q

    # ---- Local Optimization on Best Candidate ----
    # Use Nelder-Mead on the 7 DRAG parameters
    # Or better, use the raw 80x2 pulse for the final refinement.
    # Given the time budget, let's try Nelder-Mead on the 7 params first.
    
    def opt_drag(p):
        if time.time() > deadline - 10:
            return 1e6
        I_Q = gen_drag(p)
        if not check_constraints(I_Q):
            return 1e6
        return score_full(I_Q)
        
    # Initialize from the best DRAG candidate found
    # We need to extract the parameters from the best I_Q? 
    # The grid search kept the parameters. 
    # Let's find the best parameter set from the grid search that corresponds to the best I_Q.
    # Actually, we just stored I_Q. We can just use the best I_Q as the starting point for raw optimization.
    
    # Let's do a direct optimization on the raw pulse.
    # To make it converge, we might need to project onto feasible space or use a penalty.
    # L-BFGS-B with bounds is good, but bounds on slew rate are hard.
    # Nelder-Mead is robust to non-smooth constraints if we penalize.
    
    def opt_direct(x):
        if time.time() > deadline - 5:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score_full(I_Q)

    best_x = best.flatten()
    
    # Run Nelder-Mead
    try:
        if time.time() < deadline - 15:
            r = minimize(opt_direct, best_x, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
            if r.fun < best_s:
                I_Q = r.x.reshape(N, 2)
                if check_constraints(I_Q):
                    best_s = r.fun
                    best = I_Q
    except Exception:
        pass

    # Final Check
    if not check_constraints(best):
        best = seed_pulse()
        
    return best
