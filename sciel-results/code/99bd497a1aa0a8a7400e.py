import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier series basis to model periodic
    phase-space dynamics of the oscillator.
    
    Basis terms:
    1. 1 (bias)
    2. sin(x)
    3. cos(x)
    4. sin(v)
    5. cos(v)
    6. sin(x+v)
    7. cos(x+v)
    8. sin(x-v)
    9. cos(x-v)
    10. sin(2x)
    
    We use sin(2x) instead of cos(2x) to have a mix; the linear combination
    of sin and cos at same frequency is equivalent to a phase-shifted cosine,
    so having both sin and cos at each frequency gives full harmonic representation.
    With 10 params, we can represent:
    - DC offset
    - First harmonic in x (sin, cos)
    - First harmonic in v (sin, cos)
    - First harmonic in x+v (sin, cos)
    - First harmonic in x-v (sin, cos)
    That's 9 terms, leaving 1 for sin(2x) or we could use a different combination.
    
    Actually, let's be more strategic. The oscillator dynamics are likely:
    dx/dt = v
    dv/dt = f(x, v)
    
    The target y is likely dv/dt or some derivative. For a nonlinear oscillator,
    the dynamics are periodic in phase space. A Fourier basis in (x, v) coordinates
    should capture this well.
    
    Let's use:
    1. 1
    2. sin(x)
    3. cos(x)
    4. sin(v)
    5. cos(v)
    6. sin(x + v)
    7. cos(x + v)
    8. sin(x - v)
    9. cos(x - v)
    10. sin(2*x)  -- or we could use a scaled version
    
    Actually, since we have limited params, let's think about what's most important.
    The restoring force likely depends on x, and damping on v. Cross terms capture
    the coupling. Let's keep the above.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to improve numerical stability for Fourier terms
    # x range: [-0.5728, 0.5275], v range: [-0.1754, 0.2821]
    # For Fourier terms, we want the arguments to be in a reasonable range
    # Let's scale x and v to have a period that matches the expected dynamics
    # If the oscillator has a natural frequency, scaling by ~2*pi might help
    # But since we don't know the frequency, let's just use the raw values
    # or scale to have arguments in roughly [-pi, pi]
    
    # x is in [-0.57, 0.53], so x itself is already in a small range
    # v is in [-0.18, 0.28], also small
    # Let's scale to make the Fourier arguments span roughly [-pi, pi]
    # x_scale: to map [-0.57, 0.53] to ~[-pi, pi], scale by pi/0.55 ~ 5.7
    # v_scale: to map [-0.18, 0.28] to ~[-pi, pi], scale by pi/0.23 ~ 13.6
    
    # Actually, let's not over-scale. The inputs are already small.
    # Let's just use them directly, or with mild scaling.
    # For BFGS to work well, the basis functions should have similar magnitudes.
    # sin(x) for x in [-0.57, 0.53] ranges from about -0.54 to 0.51
    # cos(x) ranges from about 0.84 to 0.99
    # These are all O(1), which is good.
    
    X = x
    V = v
    
    # Compute Fourier basis terms
    b0 = np.ones_like(X)
    b1 = np.sin(X)
    b2 = np.cos(X)
    b3 = np.sin(V)
    b4 = np.cos(V)
    b5 = np.sin(X + V)
    b6 = np.cos(X + V)
    b7 = np.sin(X - V)
    b8 = np.cos(X - V)
    b9 = np.sin(2.0 * X)
    
    # Combine linearly
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    # Check time
    if time.time() - start_time > timeout:
        pass  # Safety check, but this function is fast
    
    return result
