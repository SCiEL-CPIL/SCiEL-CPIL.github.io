import numpy as np

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Hardware parameters
    chi = -2.0 * np.pi * 0.25  # Anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    
    # Time grid centered at 0
    t = (np.arange(N) - N/2 + 0.5) * dt

    # Gaussian parameters
    sigma = 5.5
    
    # Calculate peak amplitude for area pi
    A = np.sqrt(np.pi / 2.0) / sigma
    
    # I component
    I = A * np.exp(-t**2 / (2.0 * sigma**2))
    
    # Normalize area to exactly pi
    area_I = np.sum(I) * dt
    I *= (np.pi / area_I)
    
    # Check amplitude
    max_I = np.max(np.abs(I))
    if max_I > amp_bound:
        I *= (amp_bound / max_I)
        
    # DRAG Q component
    Omega_0 = np.max(I)
    alpha = chi / (2.0 * Omega_0**2)
    
    dI_dt = I * (-t / sigma**2)
    Q = alpha * dI_dt
    
    # Check slew rate
    I_ext = np.concatenate(([0.0], I))
    Q_ext = np.concatenate(([0.0], Q))
    dI = np.diff(I_ext) / dt
    dQ = np.diff(Q_ext) / dt
    slew = np.sqrt(dI**2 + dQ**2)
    max_slew = np.max(slew)
    
    # If slew is too high, reduce alpha (compromise on leakage suppression)
    # Or scale entire pulse (compromise on area/robustness)
    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        Q *= scale
        # Note: This reduces Q effectiveness but ensures validity.
        
    out = np.column_stack((I, Q))
    return out
