import numpy as np
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Hardware parameters
    dt = 0.5  # ns
    n_steps = 80
    total_time = 40.0  # ns
    
    # Default hardware parameters
    amp_bound = 0.4
    slew_bound = 0.12
    
    # Extract from hardware_model if available
    if hasattr(hardware_model, '__getitem__'):
        try:
            amp_bound = hardware_model.get('amplitude_bound', 0.4)
            slew_bound = hardware_model.get('slew_bound', 0.12)
        except:
            pass

    # Start with DRAG-style Gaussian seed
    t = np.linspace(0, total_time, n_steps + 1)[:-1]  # 80 time points
    center = total_time / 2.0
    sigma = 40.0 / 6.0  # ~6.67 ns
    
    # Gaussian pulse shape
    gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    # Normalize area to pi (for X-pi gate)
    area = np.sum(gauss) * dt
    scale_factor = np.pi / area
    I_seed = scale_factor * gauss
    Q_seed = np.zeros(n_steps)
    
    # Apply amplitude bound
    amp = np.sqrt(I_seed**2 + Q_seed**2)
    mask = amp > amp_bound
    if np.any(mask):
        I_seed[mask] *= amp_bound / amp[mask]
        Q_seed[mask] *= amp_bound / amp[mask]
    
    # Apply slew rate constraint
    # Check from zero into step 1
    amp_0 = np.sqrt(I_seed[0]**2 + Q_seed[0]**2)
    if amp_0 / dt > slew_bound:
        factor = slew_bound * dt / amp_0
        I_seed[0] *= factor
        Q_seed[0] *= factor
    
    # Check between steps
    for i in range(n_steps - 1):
        dI = I_seed[i+1] - I_seed[i]
        dQ = Q_seed[i+1] - Q_seed[i]
        dist = np.sqrt(dI**2 + dQ**2)
        if dist / dt > slew_bound:
            # Scale down the next step to satisfy constraint
            factor = slew_bound * dt / dist
            I_seed[i+1] = I_seed[i] + dI * factor
            Q_seed[i+1] = Q_seed[i] + dQ * factor
    
    # Check from last step to zero
    amp_last = np.sqrt(I_seed[-1]**2 + Q_seed[-1]**2)
    if amp_last / dt > slew_bound:
        factor = slew_bound * dt / amp_last
        I_seed[-1] *= factor
        Q_seed[-1] *= factor
    
    # Build the pulse array
    pulse_array = np.column_stack([I_seed, Q_seed])
    
    return pulse_array
