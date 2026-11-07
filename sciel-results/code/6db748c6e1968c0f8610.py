# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear model with 10 parameters:
    f(x) = a0 + a1*x + a2*x^2 + a3*x^3
           + a4 * sin(b1*x + c1)
           + a5 * cos(b2*x + c2)
           + a6 * tanh(d1*x)
           + a7 * exp(-g1*x^2)
           + a8 * x * exp(-g2*x^2)
           + a9 * sin(b3*x) * cos(b4*x)
    
    Parameters:
    p[0]=a0, p[1]=a1, p[2]=a2, p[3]=a3
    p[4]=a4, p[5]=a5, p[6]=a6, p[7]=a7, p[8]=a8, p[9]=a9
    
    Note: Some parameters (b1,c1,b2,c2,d1,g1,g2,b3,b4) are fixed to reasonable
    default values to keep the model within 10 parameters while still being
    expressive. The key insight is that the linear-in-params structure allows
    BFGS to converge reliably, while the nonlinear basis captures diverse shapes.
    
    Actually, let's use a different approach: use 10 params but make the model
    more flexible by having some params control frequencies/amplitudes in a
    way that BFGS can navigate.
    
    Better design: Use a model where all 10 params appear in a smooth,
    differentiable way that BFGS can optimize from all-ones start.
    
    f(x) = p0 + p1*x + p2*x^2 + p3*x^3
         + p4*sin(p5*x) + p6*cos(p7*x)
         + p8*tanh(p9*x)
    
    This has 10 params, all appearing smoothly. BFGS starting from all ones
    should be able to find reasonable values.
    """
    p = params
    x = np.asarray(x, dtype=float)
    
    # Polynomial part
    poly = p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x
    
    # Fourier part with parametric frequency
    # Use sin(p5*x) and cos(p7*x) where p5, p7 are frequencies
    fourier = p[4]*np.sin(p[5]*x) + p[6]*np.cos(p[7]*x)
    
    # Tanh saturation with parametric scale
    tanh_term = p[8]*np.tanh(p[9]*x)
    
    y = poly + fourier + tanh_term
    
    return y
# EVOLVE-BLOCK-END
