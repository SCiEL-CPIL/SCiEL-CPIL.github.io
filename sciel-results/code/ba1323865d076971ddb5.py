"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 330.0  # seconds
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return _DEADLINE - _elapsed()

def pulse(target_gate, hardware_model):
    """Return (80, 2) array of I, Q amplitudes for X_pi gate."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    # Extract hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (2 * np.pi * 0.002, -2 * np.pi * 0.002)  # max, min
    scale_bounds = (1.03, 0.97)  # max, min
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    n_steps = 80
    
    # DRAG-style Gaussian pulse
    # sigma such that area = pi (for X-pi gate)
    # Gaussian: A * exp(-(t - t0)^2 / (2*sigma^2))
    # Area = A * sigma * sqrt(2*pi) = pi => A = sqrt(2*pi) / sigma
    
    # Use sigma = 40/6 ns as suggested
    sigma = 40.0 / 6.0
    t = np.linspace(0, 40, n_steps + 1)  # 81 points, but we need 80 steps
    # Actually, 80 steps of 0.5 ns = 40 ns total
    # The pulse is defined at step centers or edges?
    # "80 constant steps of dt = 0.5 ns" suggests piecewise constant
    # So we have 80 values, each held for 0.5 ns
    # Let's define the pulse at the center of each step
    t_centers = np.arange(n_steps) * dt + dt/2  # 0.25, 0.75, ..., 39.75
    
    # Gaussian centered at t=20 ns (middle of 40 ns)
    t0 = 20.0
    # For X-pi gate with DRAG:
    # I(t) = A_I * exp(-(t-t0)^2 / (2*sigma^2))
    # Q(t) = A_Q * d/dt[I(t)] (DRAG correction)
    
    # First, let's just do a simple Gaussian for I, Q=0
    # Area of I pulse should be pi (for X-pi rotation)
    # A_I * sigma * sqrt(2*pi) = pi
    A_I = np.sqrt(2 * np.pi) / sigma
    
    I = A_I * np.exp(-(t_centers - t0)**2 / (2 * sigma**2))
    Q = np.zeros(n_steps)
    
    # Check amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        I *= scale
        Q *= scale
    
    # Check slew constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and last step to zero
    def check_slew(I, Q, dt, slew_bound):
        # From zero to step 1
        slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        if slew0 > slew_bound:
            return False
        # Between consecutive steps
        for i in range(n_steps - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            slew = np.sqrt(dI**2 + dQ**2) / dt
            if slew > slew_bound:
                return False
        # From last step to zero
        slewn = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        if slewn > slew_bound:
            return False
        return True
    
    if not check_slew(I, Q, dt, slew_bound):
        # Scale down to satisfy slew
        max_slew = 0
        slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        max_slew = max(max_slew, slew0)
        for i in range(n_steps - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            slew = np.sqrt(dI**2 + dQ**2) / dt
            max_slew = max(max_slew, slew)
        slewn = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        max_slew = max(max_slew, slewn)
        
        if max_slew > 0:
            scale = slew_bound / max_slew
            I *= scale
            Q *= scale
    
    # Final check
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > amp_bound + 1e-12:
        scale = amp_bound / np.max(amp)
        I *= scale
        Q *= scale
    
    return np.column_stack([I, Q])

# EVOLVE-BLOCK-END
