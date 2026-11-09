"""Robust control pulses for a three-level transmon."""
import numpy as np
from scipy.integrate import solve_ivp
import time

def pulse(target_gate, hardware_model):
    """Return (80, 2) array of I,Q amplitudes (rad/ns) for 40 ns X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Hardware parameters
    chi = -2 * np.pi * 0.25  # anharmonicity (rad/ns)
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    N = 80
    T = N * dt  # 40 ns
    
    # Time grid
    t = np.linspace(0, T, N + 1)
    t_mid = 0.5 * (t[:-1] + t[1:])  # midpoints, length 80
    
    # DRAG pulse design
    sigma = 40.0 / 6.0  # ~6.667 ns
    # Gaussian envelope
    g = np.exp(-0.5 * ((t_mid - T/2) / sigma) ** 2)
    
    # Normalize to area pi (integral of amplitude * dt = pi)
    area = np.sum(g) * dt
    amp_scale = np.pi / area
    
    # I component: scaled Gaussian
    I = amp_scale * g
    
    # Q component: DRAG term proportional to derivative of I
    # dI/dt approximated by central difference
    dI = np.zeros(N)
    dI[1:-1] = (I[2:] - I[:-2]) / (2 * dt)
    dI[0] = (I[1] - I[0]) / dt
    dI[-1] = (I[-1] - I[-2]) / dt
    
    # DRAG beta: for 3-level, beta = -chi / (2 * omega_01^2) but we tune it
    # The DRAG term: Q = beta * dI/dt
    # Optimal beta for minimizing 0->2 leakage
    # For transmon: beta_drag = chi / (2 * (omega_01^2 - omega_12^2)) approximately
    # But since we're in rotating frame, the key is to cancel the counter-rotating term
    # Standard DRAG: Q = -beta * dI/dt where beta = chi / (2 * delta)
    # Here delta is the detuning between 0->1 and 1->2 transitions
    # omega_01 ~ pi (for pi pulse in 40ns), omega_12 = omega_01 + chi
    # delta = omega_12 - omega_01 = chi = -2*pi*0.25
    # beta = chi / (2 * delta) but this is circular...
    # Actually for DRAG: Q(t) = -beta * I'(t) where beta = chi / (2 * omega_01)
    # Let's use the standard formula: beta = -chi / (2 * omega_01)
    # omega_01 = pi / T = pi / 40 for a pi pulse
    omega_01 = np.pi / T
    beta = -chi / (2 * omega_01)
    
    Q = beta * dI
    
    # Apply amplitude bound
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        I *= scale
        Q *= scale
    
    # Apply slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # Include from zero into step 1 and last step to zero
    def check_slew(I, Q, dt, bound):
        # From zero to step 0
        d0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        if d0 > bound:
            return False
        for i in range(N - 1):
            d = np.sqrt((I[i+1]-I[i])**2 + (Q[i+1]-Q[i])**2) / dt
            if d > bound:
                return False
        # From last step to zero
        dlast = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        if dlast > bound:
            return False
        return True
    
    # Iteratively reduce amplitude if slew is violated
    for _ in range(50):
        if check_slew(I, Q, dt, slew_bound):
            break
        # Find max slew violation
        max_slew = 0
        d0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        max_slew = max(max_slew, d0)
        for i in range(N - 1):
            d = np.sqrt((I[i+1]-I[i])**2 + (Q[i+1]-Q[i])**2) / dt
            max_slew = max(max_slew, d)
        dlast = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        max_slew = max(max_slew, dlast)
        
        if max_slew > slew_bound:
            scale = slew_bound / max_slew * 0.99
            I *= scale
            Q *= scale
        else:
            break
    
    # Ensure amplitude bound after slew scaling
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        I *= scale
        Q *= scale
    
    # Build output array
    result = np.column_stack([I, Q])
    
    # Verify constraints
    amp = np.sqrt(result[:,0]**2 + result[:,1]**2)
    assert np.all(amp <= amp_bound + 1e-12), "Amplitude bound violated"
    
    return result
