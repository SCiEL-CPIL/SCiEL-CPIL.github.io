import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid physical-structure + polynomial correction model for a nonlinear oscillator.
    
    Structure:
    a = -[p0 + p1 * tanh(p2 * x^2)] * v       # State-dependent damping
        - [p3 + p4 * tanh(p5 * x^2)] * x      # State-dependent stiffness
        + p6 * x^3                            # Duffing cubic stiffness
        + p7 * v * x                          # Velocity-position coupling
        + p8 * v^3                            # Cubic drag
        + p9 * sin(x)                         # Periodic correction
    
    Inputs are normalized to improve numerical conditioning for BFGS.
    The output is scaled back to the original acceleration units.
    """
    # Normalization constants based on training ranges
    # x: [-0.5728, 0.5275] -> mean ~ -0.02265, scale ~ 0.55015 (half-range)
    # v: [-0.1754, 0.2821] -> mean ~ 0.05335, scale ~ 0.22875 (half-range)
    
    x_mean = -0.02265
    x_scale = 0.55015
    v_mean = 0.05335
    v_scale = 0.22875
    
    x_n = (x - x_mean) / x_scale
    v_n = (v - v_mean) / v_scale
    
    p = params
    
    # Extract parameters
    # p0: base damping
    # p1: damping nonlinearity strength
    # p2: damping nonlinearity sharpness
    # p3: base stiffness
    # p4: stiffness nonlinearity strength
    # p5: stiffness nonlinearity sharpness
    # p6: cubic stiffness (Duffing term)
    # p7: velocity-position coupling
    # p8: cubic velocity (drag)
    # p9: sinusoidal forcing/correction
    
    x2 = x_n * x_n
    v2 = v_n * v_n
    
    # Damping term c(x) * v
    # c(x) = p0 + p1 * tanh(p2 * x^2)
    c_x = p[0] + p[1] * np.tanh(p[2] * x2)
    damp_term = c_x * v_n
    
    # Stiffness term k(x) * x
    # k(x) = p3 + p4 * tanh(p5 * x^2)
    k_x = p[3] + p[4] * np.tanh(p[5] * x2)
    stiff_term = k_x * x_n
    
    # Nonlinear corrections
    # Duffing cubic
    duffing_term = p[6] * x_n * x2
    # Velocity-position coupling
    coup_term = p[7] * v_n * x_n
    # Cubic drag
    drag_term = p[8] * v_n * v2
    # Sinusoidal
    sin_term = p[9] * np.sin(x_n)
    
    # Combine
    # Equation of motion: a = -damp - stiff + corrections
    a_n = -damp_term - stiff_term + duffing_term + coup_term + drag_term + sin_term
    
    # Scale back to original acceleration units
    # a = a_n * x_scale
    a = a_n * x_scale
    
    return a
