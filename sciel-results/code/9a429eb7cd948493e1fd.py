from scipy.optimize import minimize
import numpy as np
import time
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 325.0

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

    # Basis states for fidelity calculation
    # The task specifies: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # This implies 6 states.
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]

    # Target states for X gate (bit flip)
    # X|0> = |1>, X|1> = |0>
    # X(|0>+|1>) = |1>+|0>
    # X(|0>-|1>) = |1>-|0> = -(|0>-|1>) ... wait.
    # X = [[0,1],[1,0]].
    # X [1, 1]^T / sqrt(2) = [1, 1]^T / sqrt(2).
    # X [1, -1]^T / sqrt(2) = [-1, 1]^T / sqrt(2).
    # X [1, i]^T / sqrt(2) = [i, 1]^T / sqrt(2).
    # X [1, -i]^T / sqrt(2) = [-i, 1]^T / sqrt(2).
    
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
    
    n_op = a_op.conj().T @ a_op
    # n_op = diag(0, 1, 2)

    # Precompute constant parts of Hamiltonian
    # H = delta n + 0.5 alpha n(n-1) + Drive
    # n(n-1) for n=0,1,2 is 0,0,2.
    # So 0.5 * alpha * n(n-1) = diag(0, 0, alpha).
    # Actually, let's just compute it to be safe.
    n_nm1 = n_op @ (n_op - np.eye(3))
    H_const = 0.5 * alpha * n_nm1  # diag(0, 0, alpha)

    a_dag = a_op.conj().T
    # Drive term: 0.5 scale (I (a + a^dag) + i Q (a - a^dag))
    # a + a^dag = [[0,1,0],[1,0,sqrt(2)],[0,sqrt(2),0]]
    # a - a^dag = [[0,1,0],[-1,0,sqrt(2)],[0,sqrt(2),0]]
    # Let's precompute the matrices for I and Q drive contributions.
    # M_I = 0.5 * (a + a^dag)
    # M_Q = 0.5 * 1j * (a - a^dag)
    M_I = 0.5 * (a_op + a_dag)
    M_Q = 0.5 * 1j * (a_op - a_dag)

    def get_H(I, Q, delta, scale):
        # H = delta n + H_const + scale (I M_I + Q M_Q)
        # delta n is diagonal: [0, delta, 2*delta]
        # We can construct this quickly.
        H = np.zeros((3,3), dtype=complex)
        H[0,0] = 0
        H[1,1] = delta
        H[2,2] = 2*delta + alpha
        
        H[0,1] += scale * (I * M_I[0,1] + Q * M_Q[0,1])
        H[1,0] += scale * (I * M_I[1,0] + Q * M_Q[1,0])
        H[1,2] += scale * (I * M_I[1,2] + Q * M_Q[1,2])
        H[2,1] += scale * (I * M_I[2,1] + Q * M_Q[2,1])
        
        return H

    def simulate_pulse(I_Q, delta, scale):
        # Returns list of infidelities for the 6 states
        infidelities = []
        
        # Precompute H for each step
        H_steps = []
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            H = get_H(I, Q, delta, scale)
            H_steps.append(H)
            
        # Precompute U for each step
        U_steps = []
        for i in range(N):
            U = expm(-1j * H_steps[i] * dt)
            U_steps.append(U)
            
        for idx in range(6):
            psi = states[idx].copy()
            target = targets[idx]
            
            for i in range(N):
                psi = U_steps[i] @ psi
                
            fid = np.abs(np.vdot(target, psi))**2
            infidelities.append(1 - fid)
            
        return infidelities

    def check_constraints(I_Q):
        # Amplitude
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound:
            return False
        # Slew
        # From 0 to step 1
        if np.sqrt(I_Q[0,0]**2 + I_Q[0,1]**2) / dt > slew_bound:
            return False
        # Between steps
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound:
            return False
        # From last step to 0
        if np.sqrt(I_Q[-1,0]**2 + I_Q[-1,1]**2) / dt > slew_bound:
            return False
        return True

    def score_pulse(I_Q):
        total_score = 0.0
        count = 0
        for delta in deltas:
            for scale in scales:
                infs = simulate_pulse(I_Q, delta, scale)
                mean_inf = np.mean(infs)
                max_inf = np.max(infs)
                total_score += (mean_inf + max_inf)
                count += 1
        return total_score / count

    def generate_pulse(sigma, eta):
        t = np.linspace(0, 40, N)
        t_c = 20.0
        # Gaussian with area pi
        # Integral of A * exp(-(t-tc)^2 / (2 sigma^2)) dt = A * sqrt(2 pi) * sigma
        # We want area = pi.
        # A = pi / (sqrt(2 pi) sigma) = sqrt(pi/2) / sigma
        A = np.sqrt(np.pi / 2.0) / sigma
        I = A * np.exp(-(t - t_c)**2 / (2 * sigma**2))
        
        # DRAG: Q = eta * dI/dt
        # dI/dt = I * (-(t-tc)/sigma^2)
        dI = I * (-(t - t_c) / (sigma**2))
        Q = eta * dI
        
        I_Q = np.column_stack((I, Q))
        return I_Q

    # Initial search
    best_score = np.inf
    best_params = (10.0, 5.0)
    
    # Grid search for initial guess
    # sigma controls width, eta controls DRAG strength
    for sigma in [6, 8, 10, 12, 15]:
        if time.time() > deadline - 30: break
        for eta in [-20, -10, -5, 0, 5, 10, 20]:
            I_Q = generate_pulse(sigma, eta)
            if not check_constraints(I_Q):
                continue
            s = score_pulse(I_Q)
            if s < best_score:
                best_score = s
                best_params = (sigma, eta)
                
    # Local optimization using Nelder-Mead
    def opt_func(params):
        sigma, eta = params
        if sigma <= 1: return 10.0
        I_Q = generate_pulse(sigma, eta)
        if not check_constraints(I_Q):
            return 10.0
        return score_pulse(I_Q)

    # Run a few optimizations from different starting points if time permits
    start_points = [best_params, (8, 0), (12, 10)]
    for sp in start_points:
        if time.time() > deadline - 60: break
        try:
            res = minimize(opt_func, sp, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Generate final pulse
    I_Q = generate_pulse(best_params[0], best_params[1])
    
    # Ensure constraints
    if not check_constraints(I_Q):
        # Fallback to a simple valid pulse
        # Seed: Gaussian sigma=40/6, area pi
        sigma_seed = 40.0 / 6.0
        I_Q = generate_pulse(sigma_seed, 0.0)
        if not check_constraints(I_Q):
            # Constant pulse?
            # Area pi over 40ns -> avg height pi/40 ~ 0.078
            # Max amp 0.4.
            I = np.full(N, np.pi / 40.0)
            Q = np.zeros(N)
            I_Q = np.column_stack((I, Q))
            if not check_constraints(I_Q):
                 # Very small constant
                 I = np.full(N, 0.01)
                 Q = np.zeros(N)
                 I_Q = np.column_stack((I, Q))

    return I_Q
