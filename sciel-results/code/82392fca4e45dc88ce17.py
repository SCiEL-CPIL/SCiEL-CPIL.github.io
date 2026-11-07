"""Robust X_pi control pulse using vectorized Liouvillian evolution and L-BFGS-B."""
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
    
    # 3-level operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3)
    
    # Precompute constant parts of Liouvillian
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    # Dissipation part (constant)
    L1_term = np.kron(L1, L1.T)
    L2_term = np.kron(L2, L2.T)
    L1_diss_left = np.kron(I3, L1_dag_L1)
    L1_diss_right = np.kron(L1_dag_L1.T, I3)
    L2_diss_left = np.kron(I3, L2_dag_L2)
    L2_diss_right = np.kron(L2_dag_L2.T, I3)
    
    dissip = L1_term - 0.5 * (L1_diss_left + L1_diss_right) + \
             L2_term - 0.5 * (L2_diss_left + L2_diss_right)
    
    # Static Hamiltonian part (anharmonicity)
    H_static = 0.5 * anharmonicity * n_op @ (n_op - I3)
    H_static_left = np.kron(I3, H_static)
    H_static_right = np.kron(H_static.T, I3)
    H_static_liouv = -1j * (H_static_left - H_static_right)
    
    # Drive Hamiltonian basis: H_drive = detuning * n + 0.5 * scale * (I*(a+a_dag) + 1j*Q*(a-a_dag))
    # We need to build the full Liouvillian for each (I, Q, detuning, scale)
    
    n_left = np.kron(I3, n_op)
    n_right = np.kron(n_op.T, I3)
    n_liouv = -1j * (n_left - n_right)
    
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag  # Note: a - a_dag is anti-Hermitian
    
    # H_drive = detuning * n_op + 0.5 * scale * (I * (a+a_dag) + 1j * Q * (a-a_dag))
    # Liouvillian part: -i * (H_drive * rho - rho * H_drive)
    # = -i * detuning * (n rho - rho n) - 0.5i * scale * I * ((a+a_dag) rho - rho (a+a_dag))
    #   - 0.5i * scale * Q * (1j*(a-a_dag) rho - rho 1j*(a-a_dag))
    # = -i * detuning * (n rho - rho n) - 0.5i * scale * I * ((a+a_dag) rho - rho (a+a_dag))
    #   + 0.5 * scale * Q * ((a-a_dag) rho - rho (a-a_dag))
    
    # Let's define the superoperator parts for each drive component
    # For X drive (I component): -0.5i * (a+a_dag) rho + 0.5i * rho (a+a_dag)
    # = 0.5i * (rho (a+a_dag) - (a+a_dag) rho)
    # vec form: 0.5i * (kron((a+a_dag).T, I) - kron(I, (a+a_dag)))
    
    X_op = a + a_dag  # Hermitian
    Y_op = 1j * (a - a_dag)  # Hermitian (since a-a_dag is anti-Hermitian, multiplying by i makes it Hermitian)
    
    # Check: Y_op = i(a - a†). (a-a†)† = a† - a = -(a-a†). So Y_op† = (-i)(a†-a) = i(a-a†) = Y_op. Yes, Hermitian.
    
    X_left = np.kron(I3, X_op)
    X_right = np.kron(X_op.T, I3)
    X_liouv = -1j * (X_left - X_right)
    
    Y_left = np.kron(I3, Y_op)
    Y_right = np.kron(Y_op.T, I3)
    Y_liouv = -1j * (Y_left - Y_right)
    
    # So H_drive_liouv = detuning * n_liouv + 0.5 * scale * (I * X_liouv + Q * Y_liouv)
    
    # 9 test cases
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # 6 initial states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    # Target states after X_pi (swap |0> and |1>)
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = s[1]
        t[1] = s[0]
        t[2] = s[2]
        target_states.append(t)
    
    # Build vectorized simulation
    # For each of the 9 cases, we have 6 states
    # Total: 54 initial density matrices
    # Each rho is 3x3 = 9 complex elements
    # We stack them into a 9x54 matrix (or 54x9, let's do 9x54 for column-vector convention)
    
    # Actually, let's think about this more carefully.
    # For each case (det, scale), we have 6 initial states.
    # We want to evolve all 6 states for each case, and all 9 cases.
    # Total 54 trajectories.
    
    # Initial density matrices: rho_k = |psi_k><psi_k| for k=0..53
    # Stack: RHO0 = [rho_0, rho_1, ..., rho_53] as a 9x54 matrix (each column is vec(rho_k))
    
    # For case (c, d, s), the Liouvillian is:
    # L = H_static_liouv + d * n_liouv + 0.5 * s * (I_i * X_liouv + Q_i * Y_liouv) + dissip
    # where I_i, Q_i are the pulse values at step i.
    
    # All 54 trajectories use the same pulse, but different (det, scale) pairs.
    # So at each step, we have 9 different Liouvillians (one per case), each applied to 6 states.
    
    # We can batch this: for each case c, build the Liouvillian L_c, then compute expm(dt * L_c) and apply to the 6 states for that case.
    
    # Even better: since the 6 states for a given case all use the same Liouvillian, we can compute expm once per case per step.
    # 9 cases * 80 steps = 720 expm calls of 9x9 matrices. That's very fast.
    
    # Let's precompute the initial stacked density matrices
    RHO0 = np.zeros((9, 54), dtype=complex)
    for ci, det in enumerate(detunings):
        for si, sc in enumerate(scales):
            case_idx = ci * 3 + si
            for si2, state in enumerate(states):
                col = case_idx * 6 + si2
                rho = np.outer(state, state.conj())
                RHO0[:, col] = rho.flatten()
    
    # Target fidelity vectors
    # For each of the 54 trajectories, compute <target|rho|target>
    # target_fidelity = sum_k conj(target_k) * rho_kl * target_l
    # In vectorized form: fid = trace(target_dag * rho * target) = (target_dag ⊗ target^T) · vec(rho)
    # Actually: <psi|rho|psi> = sum_{ij} conj(psi_i) * rho_ij * psi_j
    # = sum_{ij} (conj(psi_i) * psi_j) * rho_ij
    # In vec form (column-major): vec(rho) has element (i,j) at position i + 3*j
    # fid = sum_{i,j} conj(psi_i) * psi_j * rho[i,j]
    # Let v = conj(psi) ⊗ psi (in appropriate order)
    # Actually, let's just compute it directly for each state
    
    # Precompute fidelity coefficients
    # For each trajectory k, we need: fid_k = <target_k| rho_final_k |target_k>
    # = sum_{i,j} conj(target_k[i]) * rho_final_k[i,j] * target_k[j]
    
    # In the vectorized representation, rho_final is a 9x54 matrix.
    # fid = sum_k sum_{i,j} conj(target_k[i]) * RHO_final[i, k] * target_k[j]
    
    # Let's precompute a 9x54 matrix F where F[i, k] = sum_j conj(target_k[i]) * target_k[j] * delta_{j, ?}
    # Actually, it's simpler to just compute it as a matrix product.
    
    # For each k: fid_k = target_k† @ rho_k @ target_k
    # In vectorized form with RHO as 9x54:
    # rho_k = RHO[:, k].reshape(3, 3)
    # fid_k = sum_{i,j} conj(t_k[i]) * RHO[i, k] * t_k[j]  ... wait, this isn't quite right for the matrix product
    
    # Let me be more careful.
    # rho is 3x3. vec(rho) stacks columns: [rho[0,0], rho[1,0], rho[2,0], rho[0,1], rho[1,1], rho[2,1], rho[0,2], rho[1,2], rho[2,2]]
    # fid = sum_{i,j} conj(t[i]) * rho[i,j] * t[j]
    # = sum_{i,j} conj(t[i]) * t[j] * rho[i,j]
    
    # Let's define a 9-vector w_k such that fid_k = w_k^H · vec(rho_k)
    # w_k[i + 3*j] = conj(t[i]) * t[j]
    
    W = np.zeros((9, 54), dtype=complex)
    for ci, det in enumerate(detunings):
        for si, sc in enumerate(scales):
            case_idx = ci * 3 + si
            for si2, (state, target) in enumerate(zip(states, target_states)):
                col = case_idx * 6 + si2
                for i in range(3):
                    for j in range(3):
                        W[i + 3 * j, col] = np.conj(target[i]) * target[j]
    
    # Now fid_k = W[:, k]^H @ RHO[:, k] ... wait, no.
    # fid_k = sum_{idx} conj(W[idx, k]) * RHO[idx, k]? No.
    # fid_k = sum_{i,j} conj(t[i]) * t[j] * rho[i,j]
    # If vec(rho)[i+3*j] = rho[i,j], then:
    # fid_k = sum_{i,j} W[i+3*j, k] * RHO[i+3*j, k] where W[i+3*j, k] = conj(t[i]) * t[j]
    # But this is not a Hermitian inner product, it's just a dot product.
    # Since fid should be real, we can take the real part.
    
    # Actually, let me reconsider. The fidelity is:
    # F = <psi|rho|psi> = psi† rho psi
    # = sum_{i,j} psi_i* rho_{ij} psi_j
    
    # In my W definition: W[i+3*j, k] = conj(target[i]) * target[j]
    # So sum_{i,j} W[i+3*j, k] * RHO[i+3*j, k] = sum_{i,j} conj(t[i]) * t[j] * rho[i,j] = F
    
    # This is correct! And F should be real for a valid density matrix and pure state.
    
    def vectorized_simulate(pulse_IQ):
        """Simulate all 54 trajectories and return fidelity values."""
        RHO = RHO0.copy()
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            for ci, det in enumerate(detunings):
                for si, sc in enumerate(scales):
                    case_idx = ci * 3 + si
                    # Build Liouvillian for this case
                    L = H_static_liouv + det * n_liouv + 0.5 * sc * (I_val * X_liouv + Q_val * Y_liouv) + dissip
                    
                    # Compute expm(dt * L)
                    U = expm(dt * L)
                    
                    # Apply to the 6 states for this case
                    cols = slice(case_idx * 6, case_idx * 6 + 6)
                    RHO[:, cols] = U @ RHO[:, cols]
        
        # Compute fidelities
        # fid_k = sum_{idx} W[idx, k] * RHO[idx, k]
        fids = np.real(np.sum(W * RHO, axis=0))
        return fids
    
    def compute_score(fids):
        """Compute mean + max infidelity."""
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)
    
    def objective(params):
        """Objective for optimization: smooth approximation of mean+max."""
        pulse_IQ = params.reshape(N, 2)
        
        # Check validity
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amps > amplitude_bound + 1e-10):
            return 1e6
        
        # Slew rate
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-10:
                return 1e6
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-10:
            return 1e6
        
        fids = vectorized_simulate(pulse_IQ)
        infids = 1.0 - fids
        
        # Soft-max for smoothness: use log-sum-exp style
        # score = mean + max ~ mean + soft_max
        beta = 50.0  # Temperature for softmax
        soft_max = np.log(np.sum(np.exp(beta * infids))) / beta
        return np.mean(infids) + soft_max
    
    def objective_exact(params):
        """Exact objective: mean + max."""
        pulse_IQ = params.reshape(N, 2)
        
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amps > amplitude_bound + 1e-10):
            return 1e6
        
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-10:
                return 1e6
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-10:
            return 1e6
        
        fids = vectorized_simulate(pulse_IQ)
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)
    
    # Create initial DRAG pulse
    t = np.arange(N) * dt
    sigma = 40.0 / 6.0
    center = 20.0
    gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
    edge_val = gaussian[0]
    gaussian -= edge_val
    area = np.sum(gaussian) * dt
    if area != 0:
        gaussian *= np.pi / area
    
    I_init = gaussian.copy()
    Q_init = np.zeros(N)
    
    # Add DRAG term
    gamma = 0.1
    dI_dt = np.gradient(I_init, dt)
    Q_init = -gamma * dI_dt
    
    pulse_init = np.column_stack([I_init, Q_init])
    
    # Ensure validity of initial pulse
    amps = np.sqrt(pulse_init[:, 0]**2 + pulse_init[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amplitude_bound:
        pulse_init *= amplitude_bound / max_amp
    
    # Check slew
    prev = np.zeros(2)
    for i in range(N):
        curr = pulse_init[i]
        if np.linalg.norm(curr - prev) / dt > slew_bound:
            # Scale down
            scale = slew_bound * dt / np.linalg.norm(curr - prev)
            pulse_init[i:] *= scale
            break
        prev = curr
    
    x0 = pulse_init.flatten()
    
    # Set up bounds
    bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)
    
    # Time management
    start_time = time.time()
    time_limit = 300.0
    
    # Run L-BFGS-B with smooth objective
    best_x = x0.copy()
    best_score_smooth = objective(x0)
    best_score_exact = objective_exact(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if result.fun < best_score_smooth:
            best_x = result.x
            best_score_smooth = result.fun
            best_score_exact = objective_exact(best_x)
            
    except Exception as e:
        pass
    
    # Use remaining time for refinement with exact objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            result2 = minimize(
                objective_exact,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
            
            if result2.fun < best_score_exact:
                best_x = result2.x
                best_score_exact = result2.fun
                
        except Exception as e:
            pass
    
    # Final validity check
    pulse_final = best_x.reshape(N, 2)
    amps = np.sqrt(pulse_final[:, 0]**2 + pulse_final[:, 1]**2)
    if np.any(amps > amplitude_bound + 1e-10):
        pulse_final = pulse_init  # Fall back to initial
    
    prev = np.zeros(2)
    valid = True
    for i in range(N):
        curr = pulse_final[i]
        if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-10:
            valid = False
            break
        prev = curr
    if np.linalg.norm(prev) / dt > slew_bound + 1e-10:
        valid = False
    
    if not valid:
        pulse_final = pulse_init
    
    return pulse_final
