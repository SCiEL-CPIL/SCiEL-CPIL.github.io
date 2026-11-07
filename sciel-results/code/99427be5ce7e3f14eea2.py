"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Physics-Informed Symbolic Regression with Input Normalization.
#
# Rationale:
# 1. The task is a "nonlinear oscillator". Standard generic bases (polynomials, 
#    generic tanh) often struggle with the specific symmetries and scales of 
#    oscillator dynamics (e.g., restoring forces vs damping vs driving).
# 2. The previous approach (score ~3500) used a mix of polynomials and tanh on 
#    raw inputs. The input 't' is in [30, 50], while 'x' and 'v' are in ~[-0.3, 0.4].
#    This scale mismatch causes BFGS to struggle (ill-conditioned Hessian).
# 3. Knowledge from run indicates: "Normalizing inputs to unit scale improves 
#    BFGS conditioning...". We explicitly normalize t, x, and v to roughly 
#    [-1, 1] or [0, 1] before applying the basis.
# 4. We use a basis that respects oscillator physics:
#    - Restoring force: x (linear) + x^3 (Duffing nonlinearity)
#    - Damping/Driving: v (linear damping) + v^3 (Van der Pol saturation)
#    - Time dependence: sin(t), cos(t) for periodic driving/forcing.
#    - Interaction: x*v (energy exchange/damping modulation).
#    - Constant: bias.
#
# This reduces the search space to physically plausible structures, making it 
# easier for BFGS to find the global minimum within the 120s budget.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for nonlinear oscillator acceleration.
    
    Inputs:
    t: time, range [30, 50]
    x: position, range ~[-0.2, 0.2]
    v: velocity, range ~[-0.3, 0.4]
    params: array of 10 coefficients.
    
    Output:
    Acceleration y.
    """
    p = params
    
    # 1. Normalize inputs to improve numerical conditioning for BFGS.
    # t: [30, 50] -> center 40, width 20. Map to [-1, 1].
    t_norm = (t - 40.0) / 10.0
    
    # x: ~[-0.2, 0.2]. Map to [-1, 1] approx.
    # Using a fixed scale of 0.2 is safe based on training range [-0.2056, 0.1749].
    x_norm = x / 0.2
    
    # v: ~[-0.3, 0.4]. Map to [-1, 1] approx.
    # Using a fixed scale of 0.35 is safe based on training range [-0.3391, 0.3624].
    v_norm = v / 0.35
    
    # 2. Construct the basis functions.
    # We use 10 specific terms that capture common nonlinear oscillator dynamics.
    
    # Term 1: Bias
    f0 = 1.0
    
    # Term 2: Linear restoring force (Spring)
    f1 = x_norm
    
    # Term 3: Cubic restoring force (Duffing hard/soft spring)
    f2 = x_norm ** 3
    
    # Term 4: Linear damping (Viscous)
    f3 = v_norm
    
    # Term 5: Cubic damping (Van der Pol negative/positive damping saturation)
    f4 = v_norm ** 3
    
    # Term 6: Interaction term (State-dependent damping)
    f5 = x_norm * v_norm
    
    # Term 7: Quadratic position (Asymmetric restoring force)
    f6 = x_norm ** 2
    
    # Term 8: Quadratic velocity (Air resistance style, though less common in simple ODEs, 
    #         it helps with general fit if the oscillator is damped non-linearly)
    f7 = v_norm ** 2
    
    # Term 9: Sinusoidal drive (Frequency ~1 rad in normalized t? 
    #         Original t is [30,50]. If freq is 1, sin(t) oscillates many times.
    #         Let's use normalized t to allow the fitter to pick the effective frequency 
    #         via the coefficient magnitude, but the argument is fixed.
    #         Actually, standard drive is A*sin(omega*t). 
    #         If we fix omega=1 in normalized space, it's ~10 cycles over the range.
    #         This is a reasonable guess for "oscillator" data.)
    f8 = np.sin(t_norm)
    
    # Term 10: Cosine drive (Phase shift)
    f9 = np.cos(t_norm)
    
    # 3. Linear combination
    # y = p0*f0 + p1*f1 + ... + p9*f9
    
    y = (p[0] * f0 + 
         p[1] * f1 + 
         p[2] * f2 + 
         p[3] * f3 + 
         p[4] * f4 + 
         p[5] * f5 + 
         p[6] * f6 + 
         p[7] * f7 + 
         p[8] * f8 + 
         p[9] * f9)
         
    return y

# EVOLVE-BLOCK-END
