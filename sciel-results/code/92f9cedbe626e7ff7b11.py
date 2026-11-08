"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using spectral shaping with a notch
    at the 1->2 transition to suppress leakage, then refine via projected
    gradient descent on spectral parameters.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    start_time = time.time()
    deadline = 280.0  # seconds, safety margin from 330s limit
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5  # ns
    n_steps = 80
    total_time = n_steps * dt  # 40 ns
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    # Time points (midpoint sampling)
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Pre-compute matrices
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    eye3 = np.eye(3, dtype=complex)
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    # Anharmonicity part of Hamiltonian
    H_anh = 0.5 * anharmonicity * n_op @ (n_op - eye3)
    
    # Test states
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0] / np.sqrt(2),
        [1, -1, 0] / np.sqrt(2),
        [1, 1j, 0] / np.sqrt(2),
        [1, -1j, 0] / np.sqrt(2)
    ], dtype=complex)
    
    # X gate
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    def simulate_unitary(pulse_data, delta, scale):
        """
        Simulate unitary evolution for given pulse, detuning, and scale.
        Returns the total unitary matrix.
        """
        U_total = np.eye(3, dtype=complex)
        
        for i in range(n_steps):
            I_i = pulse_data[i, 0]
            Q_i = pulse_data[i, 1]
            
            H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
            H_t = delta * n_op + H_anh + H_int
            
            # Exponentiate using eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
        
        return U_total
    
    def compute_fidelity_with_dissipation(U, delta, scale):
        """
        Compute mean fidelity over the 6 test states, including a first-order
        dissipation correction.
        """
        fidelities = []
        gamma_1 = total_time / T1
        gamma_phi = total_time / Tphi
        
        for psi in states:
            target = X @ psi
            out = U @ psi
            
            # Pure state fidelity
            fid_pure = np.abs(np.vdot(target, out))**2
            
            # First-order dissipation correction
            # T1: |1> -> |0> decay
            # Tphi: dephasing
            # Simple correction: reduce coherence by gamma_phi, add population decay
            amp_0 = np.abs(out[0])**2
            amp_1 = np.abs(out[1])**2
            amp_2 = np.abs(out[2])**2
            
            # Leakage to |2> is bad
            leakage = amp_2
            
            # T1 decay reduces |1> population
            t1_loss = gamma_1 * amp_1
            
            # Tphi dephasing reduces off-diagonal terms
            # Approximate as reducing fidelity by gamma_phi * (1 - |amp_0 - amp_1|^2)
            dephasing_loss = gamma_phi * (1 - (amp_0 - amp_1)**2) * 0.5
            
            fid = fid_pure - t1_loss - dephasing_loss - 2 * leakage
            fid = max(fid, 0.0)
            fidelities.append(fid)
        
        return np.mean(fidelities)
    
    def compute_score(pulse_data):
        """
        Compute the scoring metric: mean + max infidelity over 9 cases.
        """
        deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        infidelities = []
        for d in deltas:
            for s in scales:
                U = simulate_unitary(pulse_data, d, s)
                fid = compute_fidelity_with_dissipation(U, d, s)
                infidelities.append(1 - fid)
        
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)
    
    def enforce_constraints(p):
        """Enforce amplitude and slew rate constraints."""
        p = p.copy()
        
        # Amplitude
        amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            p *= amp_bound / max_amp
        
        # Slew rate
        max_slew = 0
        for i in range(n_steps):
            if i == 0:
                dp = np.linalg.norm(p[0]) / dt
            else:
                dp = np.linalg.norm(p[i] - p[i-1]) / dt
            if dp > max_slew:
                max_slew = dp
        
        dp_last = np.linalg.norm(p[-1]) / dt
        if dp_last > max_slew:
            max_slew = dp_last
            
        if max_slew > slew_bound:
            p *= slew_bound / max_slew
            
        return p
    
    def spectral_to_time(params):
        """
        Convert spectral parameters to time-domain pulse.
        params: [gauss_sigma, notch_depth, notch_width, area_scale, drag_alpha]
        
        The spectral envelope is a Gaussian centered at omega_01 with a notch
        at omega_12 to suppress leakage.
        """
        gauss_sigma, notch_depth, notch_width, area_scale, drag_alpha = params
        
        # Frequency axis
        omega_max = 2 * np.pi / dt  # Nyquist
        n_freq = 1024
        omega = np.linspace(-omega_max, omega_max, n_freq, endpoint=False)
        
        # Transition frequencies (in rad/ns)
        # omega_01 = 0 (reference frame)
        # omega_12 = anharmonicity = -2*pi*0.25
        omega_01 = 0.0
        omega_12 = anharmonicity
        
        # Gaussian spectral envelope
        spec_gauss = np.exp(-(omega - omega_01)**2 / (2 * gauss_sigma**2))
        
        # Notch at omega_12 to suppress 1->2 transition
        spec_notch = 1.0 - notch_depth * np.exp(-(omega - omega_12)**2 / (2 * notch_width**2))
        
        # Combined spectral envelope
        spec = spec_gauss * spec_notch
        
        # Inverse Fourier transform to get time-domain pulse
        # Use IFFT
        pulse_complex = np.fft.ifft(spec) * n_freq
        I_raw = np.real(pulse_complex)
        
        # Resample to our time grid
        # The IFFT gives values at times k * (omega_max * 2 / n_freq) = k * (2*pi/(n_freq*dt/2))
        # We need to map to our t_centered grid
        t_ifft = np.linspace(-total_time/2, total_time/2, n_freq, endpoint=False)
        
        # Interpolate to our grid
        I_interp = np.interp(t_centered, t_ifft, I_raw, period=total_time)
        
        # Normalize area
        area = np.sum(I_interp) * dt
        if area > 1e-10:
            I_interp = I_interp * (np.pi * area_scale / area)
        
        # DRAG: Q proportional to dI/dt
        dI = np.gradient(I_interp, dt)
        Q_interp = drag_alpha * dI
        
        p = np.column_stack([I_interp, Q_interp])
        return enforce_constraints(p)
    
    # Optimization using Nelder-Mead on spectral parameters
    from scipy.optimize import minimize
    
    def objective(params):
        p = spectral_to_time(params)
        return compute_score(p)
    
    # Initial guess: reasonable spectral parameters
    x0 = np.array([5.0, 0.5, 1.0, 1.0, 0.3])  # sigma, notch_depth, notch_width, area_scale, drag_alpha
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6,
                              'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        best_pulse = spectral_to_time(best_params)
        return best_pulse
    
    # Try a few more local searches
    best_params_copy = best_params.copy()
    
    for restart in range(8):
        if time.time() - start_time > deadline - 30:
            break
        
        # Perturb parameters
        perturbation = np.random.normal(0, 0.3, size=5)
        new_params = best_params_copy + perturbation
        
        # Ensure positive values where needed
        new_params[0] = max(new_params[0], 1.0)  # sigma
        new_params[1] = max(0.0, min(1.0, new_params[1]))  # notch_depth
        new_params[2] = max(new_params[2], 0.5)  # notch_width
        
        new_score = objective(new_params)
        
        if new_score < best_score:
            best_score = new_score
            best_params_copy = new_params
    
    best_pulse = spectral_to_time(best_params_copy)
    
    # Final validity check
    amp = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
    if np.max(amp) > amp_bound:
        best_pulse *= amp_bound / np.max(amp)
    
    max_slew = 0
    for i in range(n_steps):
        if i == 0:
            dp = np.linalg.norm(best_pulse[0]) / dt
        else:
            dp = np.linalg.norm(best_pulse[i] - best_pulse[i-1]) / dt
        if dp > max_slew:
            max_slew = dp
    dp_last = np.linalg.norm(best_pulse[-1]) / dt
    if dp_last > max_slew:
        max_slew = dp_last
        
    if max_slew > slew_bound:
        best_pulse *= slew_bound / max_slew
        
    return best_pulse
# EVOLVE-BLOCK-END
