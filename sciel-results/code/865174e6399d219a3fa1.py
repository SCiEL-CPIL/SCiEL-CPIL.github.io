"""New program: Parametric DRAG pulse optimization for 3-level transmon X-pi gate."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only 'X_pi' is supported")
    
    # Parse hardware model
    try:
        anharmonicity = hardware_model.get('anharmonicity', -2 * np.pi * 0.25)
        amplitude_bound = hardware_model.get('amplitude_bound', 0.4)
        slew_bound = hardware_model.get('slew_bound', 0.12)
        detuning_bounds = hardware_model.get('detuning_bounds', (-2 * np.pi * 0.002, 2 * np.pi * 0.002))
        drive_scale_bounds = hardware_model.get('drive_scale_bounds', (0.97, 1.03))
        T1 = hardware_model.get('T1', 25000)
        Tphi = hardware_model.get('Tphi', 40000)
    except AttributeError:
        anharmonicity = getattr(hardware_model, 'anharmonicity', -2 * np.pi * 0.25)
        amplitude_bound = getattr(hardware_model, 'amplitude_bound', 0.4)
        slew_bound = getattr(hardware_model, 'slew_bound', 0.12)
        detuning_bounds = getattr(hardware_model, 'detuning_bounds', (-2 * np.pi * 0.002, 2 * np.pi * 0.002))
        drive_scale_bounds = getattr(hardware_model, 'drive_scale_bounds', (0.97, 1.03))
        T1 = getattr(hardware_model, 'T1', 25000)
        Tphi = getattr(hardware_model, 'Tphi', 40000)
    
    dt = 0.5
    N_steps = 80
    t_total = N_steps * dt
    t_centers = (np.arange(N_steps) + 0.5) * dt
    
    # Precompute operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    I_op = np.eye(3, dtype=complex)
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    # Precompute static parts of Liouvillian for speed
    # H_det depends on detuning and anharmonicity
    # H_drive depends on I, Q, scale
    
    def get_liouvillian(detuning, scale, I_val, Q_val):
        H_det = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - I_op)
        H_drive = 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
        H = H_det + H_drive
        
        S = -1j * (np.kron(I_op, H) - np.kron(H.T, I_op))
        
        # Dissipation terms
        for L in [L1, L2]:
            Ldag = L.conj().T
            term1 = np.kron(Ldag.conj().T, L) 
            term2 = 0.5 * np.kron(I_op, Ldag @ L)
            term3 = 0.5 * np.kron((Ldag @ L).T, I_op)
            S += term1 - term2 - term3
            
        return S

    def simulate_pulse(P, detuning, scale):
        """
        P: (80, 2) array
        Returns mean fidelity over 4 input states.
        """
        I_vals = P[:, 0]
        Q_vals = P[:, 1]
        
        inputs = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
            (np.array([1, 0, 0], dtype=complex) + 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2)
        ]
        targets = [
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 0, 0], dtype=complex),
            (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
            (np.array([1, 0, 0], dtype=complex) - 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2)
        ]
        
        fidelities = []
        
        for k in range(4):
            rho_in = np.outer(inputs[k], inputs[k].conj())
            rho_vec = rho_in.flatten()
            
            for i in range(N_steps):
                S = get_liouvillian(detuning, scale, I_vals[i], Q_vals[i])
                U = expm(S * dt)
                rho_vec = U @ rho_vec
                
            rho_out = rho_vec.reshape(3, 3)
            F = np.real(np.vdot(targets[k], rho_out @ targets[k]))
            fidelities.append(F)
            
        return np.mean(fidelities)

    def generate_pulse(params):
        """
        params: [A, sigma, K1, K2, t_c]
        A: Amplitude
        sigma: Width
        K1: DRAG coefficient (1st derivative)
        K2: DRAG coefficient (2nd derivative)
        t_c: Center time
        """
        A = params[0]
        sigma = params[1]
        K1 = params[2]
        K2 = params[3]
        t_c = params[4]
        
        # Ensure sigma is positive
        sigma = abs(sigma) + 1e-6
        
        t = t_centers
        dt_ = t - t_c
        
        # Gaussian envelope
        exp_term = np.exp(-dt_**2 / (2 * sigma**2))
        I = A * exp_term
        
        # Derivatives
        # dI/dt = I * (-dt_ / sigma^2)
        dI_dt = -I * dt_ / (sigma**2)
        
        # d2I/dt2 = d/dt (-I * dt_ / sigma^2) = -(dI_dt * dt_ / sigma^2 + I / sigma^2)
        # dI_dt = -I * dt_ / sigma^2
        # d2I_dt2 = -(-I * dt_ / sigma^2 * dt_ / sigma^2 + I / sigma^2)
        #          = I * dt_^2 / sigma^4 - I / sigma^2
        #          = I * (dt_^2 / sigma^4 - 1 / sigma^2)
        d2I_dt2 = I * (dt_**2 / (sigma**4) - 1 / (sigma**2))
        
        Q = K1 * dI_dt + K2 * d2I_dt2
        
        I, Q = enforce_constraints(I, Q, amplitude_bound, slew_bound, dt)
        
        P = np.column_stack((I, Q))
        return P

    def objective(params):
        P = generate_pulse(params)
        infs = []
        detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        for d in detunings:
            for s in scales:
                fid = simulate_pulse(P, d, s)
                inf = 1.0 - fid
                infs.append(inf)
                
        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)

    # Initial guess
    # Area of Gaussian is A * sigma * sqrt(2*pi) = pi
    # A = pi / (sigma * sqrt(2*pi))
    sigma_init = 40.0 / 6.0
    A_init = np.pi / (sigma_init * np.sqrt(2 * np.pi))
    # DRAG coefficient estimate
    K1_init = -anharmonicity / (2 * sigma_init**2)
    K2_init = 0.0
    t_c_init = t_total / 2.0
    
    x0 = np.array([A_init, sigma_init, K1_init, K2_init, t_c_init])
    
    # Optimize
    # Use Powell for robustness without gradients
    res = minimize(objective, x0, method='Powell', 
                   options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10, 'maxfev': 5000})
    
    P_opt = generate_pulse(res.x)
    
    # Final constraint check and correction
    P_opt = enforce_constraints(P_opt[:, 0], P_opt[:, 1], amplitude_bound, slew_bound, dt)
    P_opt = np.column_stack(P_opt)
    
    return P_opt

def enforce_constraints(I, Q, amp_bound, slew_bound, dt):
    """
    Scale I and Q to satisfy amplitude and slew rate constraints.
    Returns tuple (I, Q)
    """
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        I = I * scale
        Q = Q * scale
    
    # Slew rate constraint
    # Check internal steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slew_internal = np.sqrt(dI**2 + dQ**2) / dt
    
    # Check start
    slew_start = np.sqrt(I[0]**2 + Q[0]**2) / dt
    # Check end
    slew_end = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
    
    max_slew = np.max(np.concatenate((slew_internal, [slew_start, slew_end])))
    
    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        I = I * scale
        Q = Q * scale
        
        # Re-check amplitude after slew scaling
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            scale2 = amp_bound / max_amp
            I = I * scale2
            Q = Q * scale2
            
    return I, Q
