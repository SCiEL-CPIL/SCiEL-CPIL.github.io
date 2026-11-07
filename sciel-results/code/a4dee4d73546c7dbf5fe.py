"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using a DRAG-style Gaussian.
    Optimizes sigma, amplitude, DRAG coefficient, and Q offset.
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
    
    # Time points (centers of steps)
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Pre-compute matrices
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_mat = adag @ a
    eye3 = np.eye(3)
    
    H0_base = 0.5 * anharmonicity * n_mat @ (n_mat - eye3)
    
    # Test states
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0] / np.sqrt(2),
        [1, -1, 0] / np.sqrt(2),
        [1, 1j, 0] / np.sqrt(2),
        [1, -1j, 0] / np.sqrt(2)
    ], dtype=complex)
    
    X_gate = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)
    
    def simulate_unitary(pulse_arr, delta, scale):
        """
        Simulate unitary evolution for a given pulse, detuning, and scale.
        Returns the 3x3 unitary matrix.
        """
        U_total = np.eye(3, dtype=complex)
        
        # Pre-allocate to avoid overhead? eigh is the bottleneck.
        for i in range(n_steps):
            I_i = pulse_arr[i, 0]
            Q_i = pulse_arr[i, 1]
            
            # H_int = 0.5 * scale * (I (a + adag) + i Q (a - adag))
            # a + adag = [[0, 1, 0], [1, 0, sqrt(2)], [0, sqrt(2), 0]]
            # a - adag = [[0, 1, 0], [-1, 0, sqrt(2)], [0, -sqrt(2), 0]]
            # 1j * (a - adag) = [[0, 1j, 0], [-1j, 0, 1j*sqrt(2)], [0, -1j*sqrt(2), 0]]
            
            # Optimized H construction
            H_t = np.array([
                [delta, 0.5 * scale * (I_i + 1j * Q_i), 0],
                [0.5 * scale * (I_i - 1j * Q_i), delta + anharmonicity, 0.5 * scale * np.sqrt(2) * (I_i + 1j * Q_i)],
                [0, 0.5 * scale * np.sqrt(2) * (I_i - 1j * Q_i), 2 * delta + 3 * anharmonicity]
            ], dtype=complex)
            
            # Note: The Hamiltonian in the prompt is:
            # H = detuning n + 0.5 anharmonicity n (n - 1) + 0.5 scale (I (a + a^dagger) + i Q (a - a^dagger))
            # n = diag(0, 1, 2)
            # n(n-1) = diag(0, 0, 2)
            # H_diag = [0, delta, 2*delta] + [0, 0, anharmonicity]
            # Wait, 0.5 * anharmonicity * n(n-1) for n=2 is 0.5 * alpha * 2 = alpha.
            # So H_22 = 2*delta + alpha.
            # H_11 = delta.
            # H_00 = 0.
            
            # My matrix above:
            # H[2,2] = 2*delta + 3*anharmonicity? 
            # Let's re-verify the anharmonicity term.
            # H = delta n + 0.5 alpha n(n-1)
            # n=0: 0
            # n=1: delta
            # n=2: 2 delta + 0.5 alpha (2)(1) = 2 delta + alpha.
            # So H[2,2] should be 2*delta + anharmonicity.
            
            # Let's fix the H construction to be safe and use the general formula if optimized version is tricky.
            # But the general formula uses matrix multiplications which are slower.
            # Let's stick to the general formula for correctness first, then optimize if needed.
            
            H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
            H_t = delta * n_mat + H0_base + H_int
            
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
        
        return U_total
    
    def compute_fidelity(U):
        """
        Compute mean fidelity over the 6 test states.
        """
        fidelities = []
        for psi in states:
            target = X_gate @ psi
            out = U @ psi
            fid = np.abs(np.vdot(target, out))**2
            fidelities.append(fid)
        
        return np.mean(fidelities)
    
    def compute_score(pulse_arr):
        """
        Compute score: mean + max infidelity over 9 cases.
        """
        deltas = [-detuning_bound, 0, detuning_bound]
        scales = [scale_low, 1.0, scale_high]
        
        infidelities = []
        for d in deltas:
            for s in scales:
                U = simulate_unitary(pulse_arr, d, s)
                fid = compute_fidelity(U)
                infidelities.append(1 - fid)
        
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)
    
    def enforce_constraints(p):
        """
        Enforce amplitude and slew rate constraints.
        """
        # Amplitude bound
        amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            p = p * (amp_bound / max_amp)
        
        # Slew rate bound
        max_slew = 0
        p_prev = np.zeros(2)
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > max_slew:
                max_slew = dp
            p_prev = p[i].copy()
        
        dp_last = np.linalg.norm(p[-1]) / dt
        if dp_last > max_slew:
            max_slew = dp_last
            
        if max_slew > slew_bound:
            p = p * (slew_bound / max_slew)
        
        return p
    
    def parametric_pulse(params):
        """
        Generate pulse from parameters: [sigma, amp, eta, eps_q]
        I(t) = amp * exp(-t^2 / 2 sigma^2)
        Q(t) = eta * amp * d/dt(exp(-t^2 / 2 sigma^2)) + eps_q * exp(-t^2 / 2 sigma^2)
        """
        sigma, amp, eta, eps_q = params
        
        # Ensure sigma is positive
        sigma = max(1e-3, abs(sigma))
        
        gauss = np.exp(-t_centered**2 / (2 * sigma**2))
        
        # Normalize area to pi? 
        # The prompt says "area pi" for the seed.
        # For an X_pi gate, the area of the I component should be pi.
        area = np.sum(gauss) * dt
        if area > 0:
            gauss = gauss * (np.pi / area)
        
        I = amp * gauss
        
        # Derivative of Gaussian
        # d/dt exp(-t^2/2s^2) = -t/s^2 * exp(-t^2/2s^2)
        d_gauss = -t_centered / (sigma**2) * gauss
        
        Q = eta * amp * d_gauss + eps_q * gauss
        
        p = np.column_stack([I, Q])
        
        return enforce_constraints(p)
    
    def objective(params):
        p = parametric_pulse(params)
        return compute_score(p)
    
    # Initial guess
    # Seed: sigma ~ 6.67, amp ~ 1.0 (normalized), eta ~ 0.3, eps_q ~ 0
    x0 = np.array([6.0, 1.0, 0.3, 0.0])
    
    # Track best result and deadline
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Optimization
    # Use Nelder-Mead for robustness without gradients
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8, 'disp': False})
        
        if time.time() < deadline and res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # If time remains, try a second starting point with different eta
    if time.time() < deadline - 100.0:
        x0_2 = np.array([6.0, 1.0, 0.5, 0.0])
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-8, 'disp': False})
            
            if time.time() < deadline and res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        except Exception:
            pass
        
    # Generate final pulse
    best_pulse = parametric_pulse(best_params)
    
    # Final validity check
    amp = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
    if np.max(amp) > amp_bound:
        best_pulse *= amp_bound / np.max(amp)
    
    max_slew = 0
    p_prev = np.zeros(2)
    for i in range(n_steps):
        dp = np.linalg.norm(best_pulse[i] - p_prev) / dt
        if dp > max_slew:
            max_slew = dp
        p_prev = best_pulse[i].copy()
    
    dp_last = np.linalg.norm(best_pulse[-1]) / dt
    if dp_last > max_slew:
        max_slew = dp_last
        
    if max_slew > slew_bound:
        best_pulse *= slew_bound / max_slew
        
    return best_pulse
# EVOLVE-BLOCK-END
