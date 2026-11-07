"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    """
    Generate a DRAG-style pulse for X_pi gate with robustness against
    detuning and drive scale variations.
    
    Returns (80, 2) array of I and Q amplitudes (rad/ns).
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_low, scale_high = 0.97, 1.03
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    dt = 0.5  # ns
    n_steps = 80
    total_time = n_steps * dt  # 40 ns
    
    # Start with seed DRAG Gaussian
    sigma = 40.0 / 6.0  # ~6.67 ns
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2  # centers of steps
    
    # Gaussian envelope with area pi
    gauss = np.exp(-t**2 / (2 * sigma**2))
    area = np.sum(gauss) * dt
    gauss = gauss * (np.pi / area)
    
    # DRAG correction: Q channel is proportional to derivative of I
    # For a pi pulse, DRAG beta typically ~ anharmonicity/omega_d
    # Here we use a simpler approach: Q = beta * dI/dt
    beta_drag = -anharmonicity / (2 * np.pi)  # ~0.25
    
    # Initial I channel
    I_seed = gauss.copy()
    
    # Compute Q as derivative-based DRAG
    dI = np.gradient(I_seed, dt)
    Q_seed = beta_drag * dI
    
    # Combine and normalize to meet constraints
    p = np.column_stack([I_seed, Q_seed])
    
    # Apply amplitude bound
    amp = np.sqrt(I_seed**2 + Q_seed**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        p *= scale
    
    # Apply slew rate bound
    # Check from zero to first step
    p_prev = np.zeros(2)
    for i in range(n_steps):
        dp = np.linalg.norm(p[i] - p_prev) / dt
        if dp > slew_bound:
            # Scale down the step
            allowed = slew_bound * dt
            if np.linalg.norm(p[i]) > 0:
                p[i] = p[i] * (allowed / np.linalg.norm(p[i]))
        p_prev = p[i].copy()
    
    # Check from last step to zero
    dp_end = np.linalg.norm(p[-1]) / dt
    if dp_end > slew_bound:
        allowed = slew_bound * dt
        p[-1] = p[-1] * (allowed / np.linalg.norm(p[-1]))
    
    # Now optimize to minimize infidelity across the 9 test cases
    # Parameters to optimize: we'll optimize the Gaussian width, amplitude, and DRAG beta
    
    def simulate_pulse(pulse_array, detuning, scale):
        """
        Simulate the 3-level transmon dynamics with given pulse.
        Returns fidelity for the 4 basis states.
        """
        # Hamiltonian parameters
        # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
        
        # a matrix in 3-level basis |0>, |1>, |2>
        a_mat = np.zeros((3, 3), dtype=complex)
        a_mat[1, 0] = 1.0  # |1> = a|0>... wait, a|1> = |0>, so a[0,1] = 1
        a_mat[0, 1] = 1.0
        a_mat[1, 2] = np.sqrt(2)
        
        a_dag = a_mat.conj().T
        n_mat = a_dag @ a_mat
        
        # Lindblad operators
        L1 = a_mat / np.sqrt(T1)
        L2 = n_mat * np.sqrt(2.0 / Tphi)
        
        # Initial states
        states = [
            np.array([1, 0, 0], dtype=complex),  # |0>
            np.array([0, 1, 0], dtype=complex),  # |1>
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
            np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
            np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
        ]
        
        # Target X gate (pi pulse on |0>-|1> subspace)
        # X = |0><1| + |1><0|
        X_gate = np.zeros((3, 3), dtype=complex)
        X_gate[0, 1] = 1.0
        X_gate[1, 0] = 1.0
        
        fidelities = []
        
        for psi0 in states:
            # Simulate using Liouvillian
            # rho evolves as drho/dt = -i[H, rho] + sum_k (L_k rho L_k† - 0.5{L_k†L_k, rho})
            
            def liouvillian(t, rho_vec):
                rho = rho_vec.reshape(3, 3)
                
                # Compute Hamiltonian
                I_val = pulse_array[0][int(t/dt)] if t < total_time else 0.0
                Q_val = pulse_array[1][int(t/dt)] if t < total_time else 0.0
                
                H = detuning * n_mat + 0.5 * anharmonicity * n_mat @ (n_mat - np.eye(3))
                H += 0.5 * scale * (I_val * (a_mat + a_dag) + 1j * Q_val * (a_mat - a_dag))
                
                # Compute Liouvillian action
                drho = -1j * (H @ rho - rho @ H)
                for L in [L1, L2]:
                    Ldag = L.conj().T
                    drho += L @ rho @ Ldag - 0.5 * (Ldag @ L @ rho + rho @ Ldag @ L)
                
                return drho.reshape(9)
            
            # Initial density matrix
            rho0 = np.outer(psi0, psi0.conj()).ravel()
            
            # Time points
            t_eval = np.linspace(0, total_time, n_steps + 1)
            
            try:
                sol = solve_ivp(liouvillian, [0, total_time], rho0, t_eval=t_eval,
                              method='DOP853', rtol=1e-10, atol=1e-12)
                if not sol.success:
                    return 0.0  # Failed simulation
                
                rho_final = sol.y[:, -1].reshape(3, 3)
                
                # Compute fidelity: F = <psi0| X† rho X |psi0>
                # Or trace(rho_final * X * rho_initial * X†) for pure states
                # For pure state input, fidelity = |<psi_final|X|psi0>|^2
                # But we have density matrix, so use Uhlmann fidelity or simple overlap
                
                # Simpler: compute state vector from rho_final (assuming pure state)
                # Extract largest eigenvector
                eigvals, eigvecs = np.linalg.eigh(rho_final)
                idx = np.argmax(eigvals)
                psi_final = eigvecs[:, idx]
                
                # Apply X gate to initial state
                psi_target = X_gate @ psi0
                
                # Fidelity = |<psi_target|psi_final>|^2
                fid = np.abs(np.vdot(psi_target, psi_final))**2
                fidelities.append(fid)
                
            except Exception:
                return 0.0
        
        return np.mean(fidelities)
    
    def objective(params):
        """
        Optimize pulse parameters.
        params = [sigma, amp_scale, drag_beta]
        """
        sigma, amp_scale, drag_beta = params
        
        # Reconstruct pulse
        t_local = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
        gauss = np.exp(-t_local**2 / (2 * sigma**2))
        area = np.sum(gauss) * dt
        gauss = gauss * (np.pi / area)
        
        I_local = amp_scale * gauss
        dI = np.gradient(I_local, dt)
        Q_local = drag_beta * dI
        
        p_local = np.column_stack([I_local, Q_local])
        
        # Apply amplitude bound
        amp = np.sqrt(I_local**2 + Q_local**2)
        if np.max(amp) > amp_bound:
            scale = amp_bound / np.max(amp)
            p_local *= scale
        
        # Apply slew rate bound
        p_prev = np.zeros(2)
        for i in range(n_steps):
            dp = np.linalg.norm(p_local[i] - p_prev) / dt
            if dp > slew_bound:
                allowed = slew_bound * dt
                if np.linalg.norm(p_local[i]) > 0:
                    p_local[i] = p_local[i] * (allowed / np.linalg.norm(p_local[i]))
            p_prev = p_local[i].copy()
        
        dp_end = np.linalg.norm(p_local[-1]) / dt
        if dp_end > slew_bound:
            allowed = slew_bound * dt
            p_local[-1] = p_local[-1] * (allowed / np.linalg.norm(p_local[-1]))
        
        # Compute infidelity across 9 cases
        total_infidelity = 0.0
        for detuning in [-detuning_bound, 0, detuning_bound]:
            for scale in [scale_low, 1.0, scale_high]:
                fid = simulate_pulse(p_local, detuning, scale)
                total_infidelity += (1 - fid)
        
        return total_infidelity / 9.0
    
    # Optimize with Nelder-Mead
    start_params = [sigma, 1.0, beta_drag]
    
    start_time = time.time()
    deadline = 330.0 - 30.0  # 300s limit, leave 30s margin
    
    best_params = start_params.copy()
    best_score = objective(start_params)
    
    # Quick optimization with limited iterations
    result = minimize(objective, start_params, method='Nelder-Mead',
                     options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-5})
    
    if time.time() < deadline and result.fun < best_score:
        best_params = result.x
        best_score = result.fun
    
    # Reconstruct best pulse
    sigma_best, amp_scale_best, drag_beta_best = best_params
    
    t_local = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    gauss = np.exp(-t_local**2 / (2 * sigma_best**2))
    area = np.sum(gauss) * dt
    gauss = gauss * (np.pi / area)
    
    I_best = amp_scale_best * gauss
    dI = np.gradient(I_best, dt)
    Q_best = drag_beta_best * dI
    
    p_best = np.column_stack([I_best, Q_best])
    
    # Apply amplitude bound
    amp = np.sqrt(I_best**2 + Q_best**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        p_best *= scale
    
    # Apply slew rate bound
    p_prev = np.zeros(2)
    for i in range(n_steps):
        dp = np.linalg.norm(p_best[i] - p_prev) / dt
        if dp > slew_bound:
            allowed = slew_bound * dt
            if np.linalg.norm(p_best[i]) > 0:
                p_best[i] = p_best[i] * (allowed / np.linalg.norm(p_best[i]))
        p_prev = p_best[i].copy()
    
    dp_end = np.linalg.norm(p_best[-1]) / dt
    if dp_end > slew_bound:
        allowed = slew_bound * dt
        p_best[-1] = p_best[-1] * (allowed / np.linalg.norm(p_best[-1]))
    
    return p_best
# EVOLVE-BLOCK-END
