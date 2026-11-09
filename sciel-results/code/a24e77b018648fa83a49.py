import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structurally-motivated nonlinear oscillator model (Duffing-like).
    Computes acceleration as a function of position x and velocity v.
    Uses 10 parameters to define damping, stiffness, and nonlinearities.
    
    Structure:
    a = -c(x) * v - k(x) * x + F(x, v)
    
    Where:
    c(x) = p0 + p1 * tanh(p2 * x^2)          (state-dependent damping)
    k(x) = p3 + p4 * tanh(p5 * x^2)          (state-dependent stiffness)
    F(x, v) = p6 * x^3 + p7 * v * x + p8 * v^3 + p9 * sin(x)  (nonlinear forcing/corrections)
    
    We normalize inputs to improve conditioning for the optimizer.
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
    
    # Un-normalize for the physical terms? 
    # It is often better to let the optimizer learn the scale in the parameters if we normalize inputs.
    # However, the target y is in the original units.
    # Let's stick to using normalized inputs in the structure, but the output will be in normalized units?
    # No, the output must be in the same units as the target y.
    # The target y is acceleration.
    # If we use normalized x, v, the resulting y will be in "normalized acceleration" units.
    # We need to scale back.
    # acceleration = d2x/dt2.
    # x_n = (x - x_mean) / x_scale => x = x_n * x_scale + x_mean
    # v_n = (v - v_mean) / v_scale => v = v_n * v_scale + v_mean
    # a = d2x/dt2 = x_scale * d2x_n/dt2
    # So if we compute a_n from x_n, v_n, then a = a_n * x_scale.
    
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
    c_x = p[0] + p[1] * np.tanh(p[2] * x2)
    damp_term = c_x * v_n
    
    # Stiffness term k(x) * x
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
