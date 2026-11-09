"""Improved program for robust X-pi gate control pulse."""
# EVOLVE-BLOCK-START
import numpy as np

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    Uses a DRAG-style pulse with optimized parameters for robustness.
    """
    # Time parameters
    dt = 0.5  # ns
    n_steps = 80
    t = np.arange(n_steps) * dt  # 0 to 39.5 ns
    
    # Hardware parameters from the contract
    # Anharmonicity: -2*pi*0.25 GHz -> in rad/ns: -2*pi*0.25e-3 rad/ns
    # But we work in rad/ns units for amplitudes
    
    # DRAG pulse parameters
    # Gaussian width: sigma = 40/6 ns ≈ 6.667 ns (as per seed)
    sigma = 40.0 / 6.0  # ns
    
    # Center of the pulse
    t_center = 20.0  # ns
    
    # Base Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Normalize area to pi (as per seed: area = pi)
    # Integral of gauss * dt should be pi
    # For a Gaussian: integral = sigma * sqrt(2*pi) * peak
    # So peak = pi / (sigma * sqrt(2*pi)) = sqrt(pi) / sqrt(2) / sigma * sqrt(pi)
    # Actually: ∫ exp(-0.5*((t-tc)/sigma)^2) dt = sigma * sqrt(2*pi)
    # So peak = pi / (sigma * sqrt(2*pi))
    peak = np.pi / (sigma * np.sqrt(2 * np.pi))
    
    # DRAG: add derivative of Gaussian (scaled) to Q channel to suppress |1><->|2> leakage
    # d/dt gauss = -((t - t_center) / sigma^2) * gauss
    drag_beta = 0.3  # DRAG coefficient, optimized for typical transmon
    
    # I channel: Gaussian
    I = peak * gauss
    
    # Q channel: DRAG term (derivative of Gaussian, scaled)
    # The derivative: d/dt[exp(-0.5*((t-tc)/sigma)^2)] = -((t-tc)/sigma^2) * exp(...)
    dgauss = -((t - t_center) / sigma**2) * gauss
    Q = drag_beta * peak * dgauss * sigma  # Scale to have similar magnitude
    
    # Build the pulse array
    p = np.column_stack([I, Q])
    
    # Ensure constraints are met
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > 0.4:
        scale = 0.4 / max_amp
        I *= scale
        Q *= scale
        p = np.column_stack([I, Q])
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Check and clip if necessary
    for i in range(n_steps - 1):
        diff = np.sqrt((I[i+1] - I[i])**2 + (Q[i+1] - Q[i])**2)
        if diff / dt > 0.12:
            # Reduce the step
            ratio = 0.12 * dt / diff
            I[i+1] = I[i] + (I[i+1] - I[i]) * ratio
            Q[i+1] = Q[i] + (Q[i+1] - Q[i]) * ratio
    
    # Also check from zero into step 1 and from last step to zero
    # From zero to step 0
    diff0 = np.sqrt(I[0]**2 + Q[0]**2)
    if diff0 / dt > 0.12:
        scale0 = 0.12 * dt / diff0
        I[0] *= scale0
        Q[0] *= scale0
    
    # From last step to zero
    diff_end = np.sqrt(I[-1]**2 + Q[-1]**2)
    if diff_end / dt > 0.12:
        scale_end = 0.12 * dt / diff_end
        I[-1] *= scale_end
        Q[-1] *= scale_end
    
    p = np.column_stack([I, Q])
    
    return p

# EVOLVE-BLOCK-END
