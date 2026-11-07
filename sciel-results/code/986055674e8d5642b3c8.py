import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings('ignore')

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill
    
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
    
    # Basis states for infidelity calculation
    # |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # We need to simulate evolution of these 6 states? 
    # Actually, infidelity to X gate:
    # F = <psi_target| rho_final |psi_target> ? No, fidelity to the gate operation.
    # Standard process fidelity or state fidelity for specific inputs.
    # The prompt says: "infidelity = 1 - mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # So we simulate the evolution of these 6 initial states and compare the final state to the target state X|psi_init>.
    
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_plus_i = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_minus_i = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    target_states = [
        X @ psi_0, # |1>
        X @ psi_1, # |0>
        X @ psi_plus, # |->
        X @ psi_minus, # |+>
        X @ psi_plus_i, # |-i> ? X (|0>+i|1>)/sqrt2 = (|1>-i|0>)/sqrt2
        X @ psi_minus_i # |+i> ? X (|0>-i|1>)/sqrt2 = (|1>+i|0>)/sqrt2
    ]
    # Define X matrix
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    # Recalculate targets properly
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2), # X (|0>+i|1>) = |1> + i|0> = i(|0> - i|1>) ... wait.
        # X |0> = |1>, X |1> = |0>
        # X (|0> + i|1>)/sqrt2 = (|1> + i|0>)/sqrt2
        # X (|0> - i|1>)/sqrt2 = (|1> - i|0>)/sqrt2
    ]
    
    init_states = [psi_0, psi_1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    target_states = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    ]

    # Pre-compute Lindblad terms
    # L1 = a / sqrt(T1)
    # L2 = n * sqrt(2/Tphi)
    a_mat = np.array([[0, 1, 0],
                      [0, 0, np.sqrt(2)],
                      [0, 0, 0]], dtype=complex)
    n_mat = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 2]], dtype=complex)
    
    L1 = a_mat / np.sqrt(T1)
    L2 = n_mat * np.sqrt(2 / Tphi)
    
    # Dissipator part of Liouvillian (vectorized)
    # We will work with density matrices flattened to 9x1 vectors.
    # L_diss(rho) = L1 rho L1^dag - 0.5 {L1^dag L1, rho} + L2 rho L2^dag - 0.5 {L2^dag L2, rho}
    # This is linear in rho. We can precompute the superoperator matrix M_diss (9x9)
    # such that vec(L_diss(rho)) = M_diss @ vec(rho)
    
    def vec(rho):
        return rho.flatten(order='F')
        
    def unvec(v):
        return v.reshape((3, 3), order='F')
        
    # Compute M_diss
    M_diss = np.zeros((9, 9), dtype=complex)
    for k in range(9):
        e_k = np.zeros(9, dtype=complex)
        e_k[k] = 1
        rho_k = unvec(e_k)
        diss_k = L1 @ rho_k @ L1.conj().T - 0.5 * (L1.conj().T @ L1 @ rho_k + rho_k @ L1.conj().T @ L1)
        diss_k += L2 @ rho_k @ L2.conj().T - 0.5 * (L2.conj().T @ L2 @ rho_k + rho_k @ L2.conj().T @ L2)
        M_diss[:, k] = vec(diss_k)
        
    # Hamiltonian part depends on I, Q
    # H = delta * n + 0.5 * alpha * n @ (n - I) + 0.5 * scale * (I_pulse * (a + a^dag) + 1j * Q_pulse * (a - a^dag))
    # Note: I_pulse and Q_pulse are real scalars in the pulse array.
    # Let's call the pulse variables I_p and Q_p to avoid confusion with identity matrix.
    
    I_mat = np.eye(3)
    
    def get_H(I_p, Q_p, delta, scale):
        # n(n-1) = n^2 - n
        n2 = n_mat @ n_mat
        anharmonic = 0.5 * alpha * (n2 - n_mat)
        drive = 0.5 * scale * (I_p * (a_mat + a_mat.conj().T) + 1j * Q_p * (a_mat - a_mat.conj().T))
        detuning = delta * n_mat
        return detuning + anharmonic + drive

    def simulate_single(I_pulse, Q_pulse, delta, scale):
        # I_pulse, Q_pulse are arrays of length 80
        rho = np.zeros((3, 3), dtype=complex)
        # We simulate the evolution of the 6 initial states
        # Actually, it's more efficient to simulate the superoperator evolution?
        # No, just simulate the 6 states separately.
        # State evolution: d/dt psi = -i H psi (ignoring dissipation for state vector? No, Lindblad applies to density matrix).
        # If we start with pure states, we can use the density matrix.
        # Or we can use the Lindblad equation for the density matrix.
        
        # For each initial state, we compute the final density matrix.
        # Since the evolution is linear, we can compute the superoperator U_total = exp(L_total dt) ... 
        # But L_total changes at each step.
        
        # We will simulate the density matrix for each of the 6 initial states.
        # To save time, we can pack the 6 density matrices into a single 9x6 matrix?
        # vec(rho_i) evolves as vec(rho_i_next) = M_step @ vec(rho_i_curr)
        # where M_step = exp(dt * (L_coh(I, Q) + L_diss))
        # L_coh(rho) = -i [H, rho]. This is linear.
        # Superoperator M_coh = -i (Kron(I, H) - Kron(H, I^T)) ?
        # vec(A B C) = (C^T \otimes A) vec(B)
        # vec(H rho) = (I \otimes H) vec(rho)
        # vec(rho H) = (H^T \otimes I) vec(rho)
        # vec([H, rho]) = (I \otimes H - H^T \otimes I) vec(rho)
        
        rho_vecs = np.zeros((9, 6), dtype=complex)
        for i, psi in enumerate(init_states):
            rho_vecs[:, i] = vec(np.outer(psi, psi.conj()))
            
        for i in range(N):
            I_p = I_pulse[i]
            Q_p = Q_pulse[i]
            H = get_H(I_p, Q_p, delta, scale)
            
            # Build M_coh
            M_coh = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
            
            # Total Liouvillian superoperator
            M_total = M_coh + M_diss
            
            # Exponentiate
            U_step = expm(M_total * dt)
            
            # Apply to all 6 states
            rho_vecs = U_step @ rho_vecs
            
        # Compute fidelities
        fidelities = []
        for i, target in enumerate(target_states):
            rho_final = unvec(rho_vecs[:, i])
            # Fidelity for pure target state |t> and mixed state rho:
            # F = <t| rho |t>
            t_vec = target.flatten()
            f = np.real(np.vdot(t_vec, rho_final @ t_vec))
            # Clamp to [0, 1]
            f = np.clip(f, 0.0, 1.0)
            fidelities.append(f)
            
        return np.array(fidelities)

    def score_pulse(I_pulse, Q_pulse):
        infidelities = []
        for delta in deltas:
            for scale in scales:
                fids = simulate_single(I_pulse, Q_pulse, delta, scale)
                inf = 1.0 - np.mean(fids)
                infidelities.append(inf)
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)

    def check_constraints(I_pulse, Q_pulse):
        # Amplitude
        amps = np.sqrt(I_pulse**2 + Q_pulse**2)
        if np.any(amps > amp_bound + 1e-9):
            return False
        # Slew rate
        # ||p[i+1] - p[i]|| / dt <= 0.12
        # p = [I, Q]
        # Check from 0 to step 1
        if np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / dt > slew_bound + 1e-9:
            return False
        # Check between steps
        dI = np.diff(I_pulse)
        dQ = np.diff(Q_pulse)
        dists = np.sqrt(dI**2 + dQ**2)
        if np.any(dists / dt > slew_bound + 1e-9):
            return False
        # Check from last step to 0
        if np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / dt > slew_bound + 1e-9:
            return False
        return True

    def penalty(I_pulse, Q_pulse):
        if check_constraints(I_pulse, Q_pulse):
            return 0.0
        # Soft penalty
        p = 0.0
        amps = np.sqrt(I_pulse**2 + Q_pulse**2)
        p += np.sum(np.maximum(amps - amp_bound, 0)**2)
        
        # Slew
        if np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / dt > slew_bound:
            p += (np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / dt - slew_bound)**2
        dI = np.diff(I_pulse)
        dQ = np.diff(Q_pulse)
        dists = np.sqrt(dI**2 + dQ**2)
        p += np.sum(np.maximum(dists / dt - slew_bound, 0)**2)
        if np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / dt > slew_bound:
            p += (np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / dt - slew_bound)**2
            
        return 1000.0 * p

    def objective(params):
        I_pulse = params[:N]
        Q_pulse = params[N:]
        score = score_pulse(I_pulse, Q_pulse)
        pen = penalty(I_pulse, Q_pulse)
        return score + pen

    # Initial guess: DRAG pulse
    # Gaussian I(t) = A * exp(-t^2 / (2 sigma^2))
    # Q(t) = alpha_drag * dI/dt
    
    def generate_drag_pulse(A, sigma, alpha_drag):
        t = np.arange(N) * dt
        t_center = (N - 1) / 2.0 * dt
        t_shifted = t - t_center
        I = A * np.exp(-t_shifted**2 / (2 * sigma**2))
        # Derivative of Gaussian: -t/sigma^2 * I
        dI_dt = -t_shifted / (sigma**2) * I
        Q = alpha_drag * dI_dt
        # Ensure area is approx pi?
        # Area = A * sigma * sqrt(2*pi)
        # We want Area ~ pi. So A ~ 1/sqrt(2*pi)/sigma?
        # But A is a parameter.
        return I, Q

    # Parametric optimization
    def parametric_objective(params):
        A, sigma, alpha_drag = params
        I, Q = generate_drag_pulse(A, sigma, alpha_drag)
        if not check_constraints(I, Q):
            return 1e6
        return score_pulse(I, Q)

    # Bounds for parametric search
    # A: 0.1 to 0.4 (amp bound)
    # sigma: 2.0 to 10.0
    # alpha_drag: -2.0 to 2.0
    
    bounds = [(0.1, 0.4), (2.0, 10.0), (-2.0, 2.0)]
    
    # Run differential evolution
    # Use small popsize for speed
    de_res = differential_evolution(parametric_objective, bounds, seed=42, maxiter=50, popsize=10, tol=1e-4, polish=False)
    
    best_A, best_sigma, best_alpha = de_res.x
    best_I, best_Q = generate_drag_pulse(best_A, best_sigma, best_alpha)
    best_score = de_res.fun
    
    # Refine with local optimization on full vector
    x0 = np.concatenate([best_I, best_Q])
    
    # Use L-BFGS-B
    # We need to handle constraints. L-BFGS-B supports bounds, but not non-linear constraints easily.
    # We will use the penalty method.
    
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.fun < best_score:
        best_I = res.x[:N]
        best_Q = res.x[N:]
        best_score = res.fun
    
    # Ensure final pulse is valid
    if not check_constraints(best_I, best_Q):
        # Fallback to seed if invalid
        t = np.arange(N) * dt
        t_center = (N - 1) / 2.0 * dt
        t_shifted = t - t_center
        sigma = 40.0 / 6.0
        A = np.sqrt(np.pi / (2 * np.pi)) / sigma
        best_I = A * np.exp(-t_shifted**2 / (2 * sigma**2))
        best_Q = np.zeros(N)
        
    return np.column_stack((best_I, best_Q))
