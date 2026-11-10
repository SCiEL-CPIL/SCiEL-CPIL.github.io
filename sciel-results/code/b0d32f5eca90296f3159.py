"""Pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    # Extract hardware parameters
    anharmonicity = -2 * math.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (-2 * math.pi * 0.002, 2 * math.pi * 0.002)
    scale_bounds = (0.97, 1.03)
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    # Build operators for 3-level transmon
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = math.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    I3 = np.eye(3, dtype=complex)
    
    # Test states
    psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_p = np.array([1.0, 1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    psi_m = np.array([1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    psi_ip = np.array([1.0, 1.0j, 0.0], dtype=complex) / math.sqrt(2.0)
    psi_im = np.array([1.0, -1.0j, 0.0], dtype=complex) / math.sqrt(2.0)
    test_states = [psi0, psi1, psi_p, psi_m, psi_ip, psi_im]
    
    detunings = [-2 * math.pi * 0.002, 0.0, 2 * math.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    dt = 0.5
    N = 80
    t = np.arange(N) * dt + dt / 2
    sigma = 40.0 / 6.0
    
    env = np.exp(-t**2 / (2 * sigma**2))
    gauss_area = sigma * math.sqrt(2 * math.pi)
    env = env * (math.pi / gauss_area)
    env[0] = 0.0
    env[-1] = 0.0
    
    dI = np.zeros(N)
    dI[1:-1] = (env[2:] - env[:-2]) / (2 * dt)
    dI[0] = (env[1] - env[0]) / dt
    dI[-1] = (env[-1] - env[-2]) / dt
    
    beta = 0.0
    I = env.copy()
    Q = beta * dI
    
    def check_constraints(I, Q):
        amps = np.sqrt(I**2 + Q**2)
        if np.max(amps) > amp_bound:
            return False
        if math.sqrt(I[0]**2 + Q[0]**2) / dt > slew_bound:
            return False
        for i in range(N - 1):
            dI_step = I[i+1] - I[i]
            dQ_step = Q[i+1] - Q[i]
            if math.sqrt(dI_step**2 + dQ_step**2) / dt > slew_bound:
                return False
        if math.sqrt(I[-1]**2 + Q[-1]**2) / dt > slew_bound:
            return False
        return True
    
    if not check_constraints(I, Q):
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if check_constraints(I * mid, Q * mid):
                lo = mid
            else:
                hi = mid
        I = I * lo
        Q = Q * lo
    
    def simulate_pulse(I, Q, detuning, scale):
        L0 = a / math.sqrt(T1)
        L1 = n * math.sqrt(2.0 / Tphi)
        
        def lindblad_super(L):
            LdL = L.conj().T @ L
            return np.kron(L, L.conj()) - 0.5 * (np.kron(LdL, I3) + np.kron(I3, LdL.conj().T))
        
        L0_super = lindblad_super(L0)
        L1_super = lindblad_super(L1)
        L_total = L0_super + L1_super
        
        U = I3.copy()
        
        for i in range(N):
            H = detuning * n + 0.5 * anharmonicity * n @ (n - I3)
            H += 0.5 * scale * (I[i] * (a + adag) + 1j * Q[i] * (a - adag))
            
            H_super = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
            Liouvillian = H_super + L_total
            
            try:
                from scipy.linalg import expm
                U_step = expm(Liouvillian * dt)
            except:
                U_step = I3 + Liouvillian * dt
            
            U = U_step @ U
        
        return U
    
    def compute_infidelity(I, Q):
        fidelities = []
        for det in detunings:
            for scl in scales:
                U = simulate_pulse(I, Q, det, scl)
                fid_sum = 0.0
                for psi in test_states:
                    psi_vec = psi.flatten()
                    psi_out = U @ psi_vec.reshape(3, 1)
                    psi_out = psi_out.flatten()
                    fid = abs(np.vdot(psi_vec, psi_out))**2
                    fid_sum += fid
                mean_fid = fid_sum / len(test_states)
                infid = 1.0 - mean_fid
                fidelities.append(infid)
        
        mean_infid = np.mean(fidelities)
        max_infid = np.max(fidelities)
        return mean_infid + max_infid
    
    current_I = I.copy()
    current_Q = Q.copy()
    current_score = compute_infidelity(current_I, current_Q)
    
    start_time = time.time()
    deadline = start_time + 300.0
    
    best_I = current_I.copy()
    best_Q = current_Q.copy()
    best_score = current_score
    
    def make_pulse(params):
        beta_val = params[0]
        sigma_scale = params[1]
        amp_scale = params[2]
        phase = params[3]
        
        sigma_new = sigma * sigma_scale
        env_new = np.exp(-t**2 / (2 * sigma_new**2))
        gauss_area_new = sigma_new * math.sqrt(2 * math.pi)
        env_new = env_new * (math.pi / gauss_area_new)
        env_new[0] = 0.0
        env_new[-1] = 0.0
        
        dI_new = np.zeros(N)
        dI_new[1:-1] = (env_new[2:] - env_new[:-2]) / (2 * dt)
        dI_new[0] = (env_new[1] - env_new[0]) / dt
        dI_new[-1] = (env_new[-1] - env_new[-2]) / dt
        
        I_orig = env_new * amp_scale
        Q_orig = beta_val * dI_new * amp_scale
        
        I_new = I_orig * math.cos(phase) - Q_orig * math.sin(phase)
        Q_new = I_orig * math.sin(phase) + Q_orig * math.cos(phase)
        
        if not check_constraints(I_new, Q_new):
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if check_constraints(I_new * mid, Q_new * mid):
                    lo = mid
                else:
                    hi = mid
            I_new = I_new * lo
            Q_new = Q_new * lo
        
        return I_new, Q_new
    
    np.random.seed(42)
    num_trials = 200
    step_sizes = [0.1, 0.05, 0.01]
    
    for trial in range(num_trials):
        if time.time() > deadline - 30.0:
            break
        
        params = np.array([0.0, 1.0, 1.0, 0.0])
        scale = step_sizes[min(trial // 50, 2)]
        params += np.random.randn(4) * scale
        
        I_new, Q_new = make_pulse(params)
        score_new = compute_infidelity(I_new, Q_new)
        
        if score_new < best_score:
            best_score = score_new
            best_I = I_new.copy()
            best_Q = Q_new.copy()
    
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
