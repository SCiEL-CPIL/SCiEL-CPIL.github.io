import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed structure:
    a = -stiffness(x) * x - damping(v) * v + forcing_term
    with tanh-based nonlinearities for bounded response.
    """
    # Scale inputs for numerical stability
    x_scale = 0.55
    v_scale = 0.23
    
    x_n = x / x_scale
    v_n = v / v_scale
    
    # Physics-inspired structure with 10 parameters:
    # p0: base stiffness
    # p1: stiffness nonlinearity strength (Duffing-like)
    # p2: stiffness saturation scale
    # p3: base damping
    # p4: damping nonlinearity (quadratic drag)
    # p5: velocity-dependent stiffness modulation
    # p6: coupling term x*v
    # p7: bias/offset
    # p8: additional velocity nonlinearity
    # p9: additional position nonlinearity
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Stiffness term: -x * (p0 + p1 * tanh(p2 * x^2))
    # This gives linear restoring force with softening/hardening cubic nonlinearity
    stiffness = p0 + p1 * np.tanh(p2 * x_n**2)
    force_x = -x_n * stiffness
    
    # Damping term: -v * (p3 + p4 * v^2)
    # Linear damping plus quadratic drag
    damping = p3 + p4 * v_n**2
    force_v = -v_n * damping
    
    # Velocity-modulated stiffness: additional force from v affecting position response
    force_vx = -p5 * x_n * np.tanh(v_n)
    
    # Coupling term: x*v interaction
    force_coupling = p6 * x_n * v_n
    
    # Additional nonlinearities for flexibility
    # p8: velocity cubed term (bounded by tanh)
    force_v_cubed = p8 * np.tanh(v_n**3)
    
    # p9: position-velocity cross nonlinearity
    force_cross = p9 * np.tanh(x_n * v_n)
    
    # Combine all terms and scale back
    acceleration = (force_x + force_v + force_vx + force_coupling + 
                    force_v_cubed + force_cross + p7)
    
    # Scale output to appropriate range
    # Training target range is approximately [-0.105, 0.111], so scale by ~0.1
    return acceleration * 0.1
