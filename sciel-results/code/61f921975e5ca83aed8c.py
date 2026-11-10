"""Pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill
    
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_min, scale_max = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0
    
    # Build 3-level operators
    a_op = np.zeros((3, 3), dtype=complex)
    a_op[0, 1] = np.sqrt(1.0)
    a_op[1, 2] = np.sqrt(2.0)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    def build_hamiltonian(detuning, scale, I, Q):
        H = detuning * n_op + 0.5 * anharmonicity * (n_op @ n_op - n_op)
        H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H
    
    def lindblad_superop(H, L1, L2):
        dim = 3
        I3 = np.eye(dim)
        H_comm = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
        
        def lindblad_term(L):
            L_dag = L.conj().T
            LdL = L_dag @ L
            term = np.kron(L.conj(), L) + np.kron(L, L_dag) - 0.5 * (np.kron(LdL, I3) + np.kron(I3, LdL))
            return term
        
        return H_comm + lindblad_term(L1) + lindblad_term(L2)
    
    # Pre-compute superoperators for the 9 detuning/scale combinations
    def precompute_superops(I_arr, Q_arr):
        """Pre-compute all step superoperators for all 9 corners."""
        detunings = [-detuning_bound, 0.0, detuning_bound]
        scales = [scale_min, 1.0, scale_max]
        
        all_sops = {}
        for det in detunings:
            for sc in scales:
                sops = []
                for i in range(n_steps):
                    H = build_hamiltonian(det, sc, I_arr[i], Q_arr[i])
                    L = lindblad_superop(H, L1, L2)
                    sops.append(sla.expm(L * dt))
                all_sops[(det, sc)] = sops
        return all_sops
    
    def simulate_with_sops(all_sops, I_arr, Q_arr):
        """Simulate using pre-computed superoperators and return mean + max infidelity."""
        dim = 3
        initial_states = [
            np.array([1.0, 0, 0], dtype=complex),
            np.array([0, 1.0, 0], dtype=complex),
            np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex),
            np.array([1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex),
            np.array([1/np.sqrt(2), 1j/np.sqrt(2), 0], dtype=complex),
            np.array([1/np.sqrt(2), -1j/np.sqrt(2), 0], dtype=complex),
        ]
        
        X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
        
        infs = []
        for key, sops in all_sops.items():
            fids = []
            for psi in initial_states:
                rho_v = np.outer(psi, psi.conj()).flatten()
                for op in sops:
                    rho_v = op @ rho_v
                rho_final = rho_v.reshape(dim, dim)
                psi_t = X_mat @ psi
                fid = np.real(np.vdot(psi_t, rho_final @ psi_t))
                fids.append(fid)
            infs.append(1.0 - np.mean(fids))
        
        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)
    
    def is_valid(I_arr, Q_arr):
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        if np.any(amps > amp_bound + 1e-12):
            return False
        if np.linalg.norm([I_arr[0], Q_arr[0]]) / dt > slew_bound + 1e-12:
            return False
        for i in range(n_steps - 1):
            diff = np.array([I_arr[i+1] - I_arr[i], Q_arr[i+1] - Q_arr[i]])
            if np.linalg.norm(diff) / dt > slew_bound + 1e-12:
                return False
        if np.linalg.norm([I_arr[-1], Q_arr[-1]]) / dt > slew_bound + 1e-12:
            return False
        return True
    
    def make_drag_pulse(sigma_t, alpha):
        """Create a DRAG-style Gaussian pulse."""
        t = np.arange(n_steps) * dt
        center = T / 2
        t_centered = t - center
        gauss = np.exp(-t_centered**2 / (2 * sigma_t**2))
        
        area = np.sum(gauss) * dt
        I = gauss * (np.pi / area)
        
        dI = np.gradient(I, dt)
        Q = -alpha * dI
        
        amps = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I *= scale
            Q *= scale
        
        if not is_valid(I, Q):
            for _ in range(30):
                I *= 0.95
                Q *= 0.95
                if is_valid(I, Q):
                    break
        
        return I, Q
    
    # Generate seed pulse
    sigma_t = 40.0 / 6.0
    alpha_seed = 1.0 / (4.0 * abs(anharmonicity))
    I0, Q0 = make_drag_pulse(sigma_t, alpha_seed)
    
    if not is_valid(I0, Q0):
        I0 *= 0.5
        Q0 *= 0.5
    
    # Compute seed score
    sops0 = precompute_superops(I0, Q0)
    seed_score = simulate_with_sops(sops0, I0, Q0)
    best_I = I0.copy()
    best_Q = Q0.copy()
    best_score = seed_score
    
    # Grid search over sigma and alpha
    sigmas = np.linspace(0.4, 2.5, 15) * (40.0 / 6.0)
    alphas = np.linspace(0.0, 2.5, 16) / (4.0 * abs(anharmonicity))
    
    for sig in sigmas:
        if time.time() > deadline - 30:
            break
        for alpha in alphas:
            I_var, Q_var = make_drag_pulse(sig, alpha)
            if not is_valid(I_var, Q_var):
                continue
            sops = precompute_superops(I_var, Q_var)
            score = simulate_with_sops(sops, I_var, Q_var)
            if score < best_score:
                best_score = score
                best_I = I_var.copy()
                best_Q = Q_var.copy()
    
    # Local refinement around best using random perturbations
    if time.time() < deadline - 20:
        n_trials = 200
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            I_pert = best_I.copy()
            Q_pert = best_Q.copy()
            
            noise_scale = 0.02 * (1 - trial / n_trials)
            I_pert += noise_scale * np.random.randn(n_steps)
            Q_pert += noise_scale * np.random.randn(n_steps)
            
            amps = np.sqrt(I_pert**2 + Q_pert**2)
            max_amp = np.max(amps)
            if max_amp > amp_bound:
                scale = amp_bound / max_amp
                I_pert *= scale
                Q_pert *= scale
            
            if not is_valid(I_pert, Q_pert):
                continue
            
            sops = precompute_superops(I_pert, Q_pert)
            score = simulate_with_sops(sops, I_pert, Q_pert)
            if score < best_score:
                best_score = score
                best_I = I_pert.copy()
                best_Q = Q_pert.copy()
    
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
