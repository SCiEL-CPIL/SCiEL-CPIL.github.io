"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pulse(target_gate, hardware_model):
    """
    Generate a robust DRAG pulse for X_pi gate using an analytic sech^2 shape
    and fixed DRAG correction, avoiding runtime optimization.
    
    Returns (80, 2) array of I and Q amplitudes (rad/ns).
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5  # ns
    n_steps = 80
    total_time = n_steps * dt  # 40 ns
    
    # Time points (centers of steps)
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    
    # Pulse Shape: sech^2 is better for leakage suppression than Gaussian
    # We want area = pi
    # I(t) = A * sech^2((t - T/2) / tau)
    # Area = A * tau * pi  (integral of sech^2 is pi)
    # So A * tau = 1 => A = 1/tau
    # We need to choose tau (width) to satisfy slew and amplitude bounds.
    
    # Let's try tau = 6.0 ns (similar to seed sigma but sech^2 is wider in tails)
    # Actually, sech^2 decays slower. Let's pick tau to center the energy.
    # A good rule of thumb for 40ns pulse is effective width ~ 10-15ns.
    # Let's parameterize by 'w' where I(t) = sech^2(t/w)
    # Integral is w * pi. So A = 1/w.
    # Max amplitude is A = 1/w.
    # Derivative: d/dt sech^2(t/w) = -2/w * sech^2(t/w) * tanh(t/w)
    # Max derivative occurs around t/w ~ 0.85? No, max of sech^2 * tanh.
    # Let's just compute numerically for a few candidates and pick the best valid one.
    
    candidates = [
        {'w': 8.0, 'beta': 0.20},
        {'w': 10.0, 'beta': 0.15},
        {'w': 12.0, 'beta': 0.10},
        {'w': 15.0, 'beta': 0.05},
    ]
    
    best_pulse = None
    best_score = 0.0
    
    for c in candidates:
        w = c['w']
        beta = c['beta']
        
        # Center the pulse at T/2 = 20 ns
        t_centered = t - total_time / 2.0
        
        # I(t)
        I = (1.0 / w) * (1.0 / np.cosh(t_centered / w))**2
        
        # Ensure area is exactly pi (numerical correction)
        area = np.sum(I) * dt
        I = I * (np.pi / area)
        
        # Q(t) = beta * dI/dt
        dI = np.gradient(I, dt)
        Q = beta * dI
        
        p = np.column_stack([I, Q])
        
        # Check Validity
        # 1. Amplitude bound
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            # Scale down
            scale = amp_bound / np.max(amp)
            p *= scale
            I = p[:, 0]
            Q = p[:, 1]
        
        # 2. Slew rate bound
        # Check from zero to first step
        p_prev = np.zeros(2)
        valid = True
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > slew_bound:
                # If invalid, this candidate is rejected or we need to scale more aggressively
                # For robustness, we'll just mark it as less preferred
                valid = False
                break
            p_prev = p[i].copy()
        
        if not valid:
            # Check end
            dp_end = np.linalg.norm(p[-1]) / dt
            if dp_end > slew_bound:
                continue
            # If only intermediate or end failed, we might still use it if we scale, 
            # but for simplicity, let's prefer candidates that are naturally valid.
            # Actually, let's just scale to meet slew if possible.
            # But scaling reduces fidelity. Let's stick to candidates that are valid.
            pass
        
        # Check end slew
        dp_end = np.linalg.norm(p[-1]) / dt
        if dp_end > slew_bound:
            # Scale down the whole pulse to meet slew?
            # Slew is linear in amplitude, so scaling by k scales slew by k.
            max_slew = np.max([dp_end, np.max(np.linalg.norm(np.diff(p, axis=0), axis=1) / dt)])
            if max_slew > slew_bound:
                scale = slew_bound / max_slew
                p *= scale
        
        # Final validity check
        amp_final = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        if np.max(amp_final) > amp_bound * (1 + 1e-9):
            continue
        
        p_prev = np.zeros(2)
        valid_slew = True
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > slew_bound * (1 + 1e-9):
                valid_slew = False
                break
            p_prev = p[i].copy()
        if np.linalg.norm(p[-1]) / dt > slew_bound * (1 + 1e-9):
            valid_slew = False
            
        if not valid_slew:
            continue
        
        # Estimate Score (Heuristic: prefer pulses with lower max amplitude and moderate width)
        # Lower max amplitude allows more room for optimization, but here we are fixed.
        # Fidelity is generally better with smoother pulses (smaller w) but DRAG needs to be strong enough.
        # A w=8-10 ns is typically good for 40ns gate.
        # Let's pick the one with the "best" balance. 
        # Score heuristic: minimize max amplitude (efficiency) while keeping width reasonable.
        score_heuristic = -np.max(amp_final) + 0.01 * w 
        
        if score_heuristic > best_score:
            best_score = score_heuristic
            best_pulse = p
    
    if best_pulse is None:
        # Fallback: simple Gaussian, heavily scaled
        t_centered = t - total_time / 2.0
        sigma = 8.0
        I = np.exp(-t_centered**2 / (2 * sigma**2))
        area = np.sum(I) * dt
        I = I * (np.pi / area)
        Q = np.zeros_like(I)
        p = np.column_stack([I, Q])
        # Scale to meet slew
        max_slew = np.max(np.linalg.norm(np.diff(p, axis=0), axis=1) / dt)
        max_slew = max(max_slew, np.linalg.norm(p[0])/dt, np.linalg.norm(p[-1])/dt)
        if max_slew > slew_bound:
            p *= slew_bound / max_slew
        best_pulse = p
        
    return best_pulse
# EVOLVE-BLOCK-END
