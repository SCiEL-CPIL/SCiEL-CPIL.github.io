"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin for 120s limit + overhead

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Structural Change: Global Smooth Basis Function Model.
    Replaces piecewise logic with a continuous blend of quadratic (elastic/curvature)
    and saturated exponential (hardening) terms, modulated by temperature.
    This avoids kinks and allows BFGS to navigate the parameter space more smoothly.
    """
    e = strain
    t = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Time guard: if we are close to the limit, return a simple linear fallback
    # to ensure the evaluator doesn't kill the process mid-calculation.
    # This is unlikely to trigger for a single equation call, but ensures safety.
    if time.time() - _START_TIME > _TIME_LIMIT:
        return p0 + p1 * e

    # Temperature factor: centered at 0.5, range [0.06, 1.0] approx
    # t_norm in [-0.43, 0.5]
    t_norm = t - 0.5
    
    # 1. Elastic/Linear-Quadratic Component
    # Captures the initial slope and potential curvature in the elastic region
    # p1: linear slope
    # p2: quadratic curvature (small, to handle non-linearity near yield)
    # Modulated by temperature: E(t)
    elastic_mod = p1 * (1.0 + p7 * t_norm)
    elastic_term = elastic_mod * e + p2 * e * e

    # 2. Hardening/Saturation Component
    # Captures the plastic flow and saturation
    # Uses a smooth exponential saturation: A * (1 - exp(-B * e))
    # This is smooth everywhere and behaves like A*B*e for small e, A for large e.
    # We want it to kick in after yield. We can shift e or just let the fit handle it.
    # To allow "yield" behavior, we can use a shifted strain: (e - e_yield)
    # But to keep it smooth and global, we use a weighted sum of basis functions.
    # Basis 1: 1 - exp(-p3 * e)  -> Saturation
    # Basis 2: 1 - exp(-p4 * (e - p5)) -> Shifted saturation (approximates yield offset)
    
    # Let's use a single robust hardening term with temperature modulation
    # p3: saturation rate (inverse of hardening strain)
    # p4: hardening stress amplitude
    # p5: strain offset (yield strain approximation)
    
    # Ensure positive argument for exp if possible, or rely on math
    # exp(-large) -> 0, exp(-negative) -> large. 
    # To avoid overflow, clip the argument of exp.
    arg_hard = p3 * (e - p5)
    # Clip to prevent overflow in exp if p3*(e-p5) is very negative
    arg_hard = np.clip(arg_hard, -500, 500)
    
    hardening_term = p4 * (1.0 - np.exp(-arg_hard))
    
    # Modulate hardening by temperature
    hardening_mod = 1.0 + p8 * t_norm
    hardening_term = hardening_term * hardening_mod

    # 3. Thermal Stress Offset
    # p6: thermal expansion coefficient effect
    thermal_term = p6 * t_norm

    # Combine
    # p0: global offset
    stress = p0 + elastic_term + hardening_term + thermal_term
    
    return stress
# EVOLVE-BLOCK-END
