"""Rewritten symbolic regression with a richer basis and safe numerical guards."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible symbolic regression model with 10 parameters.
    
    Basis functions designed to cover common function families:
      Polynomials: 1, x, x^2, x^3
      Periodic:     sin(f1*x), cos(f1*x), sin(f2*x), cos(f2*x)
      Saturating:   tanh(s*x)
      Localized:    exp(-w*(x-c)^2)
      Rational-like: x / (1 + r*x^2)
    
    Parameters:
      p0: constant weight
      p1: linear weight
      p2: quadratic weight
      p3: cubic weight
      p4: amplitude for sin(f1*x)
      p5: frequency f1 (clipped)
      p6: amplitude for cos(f1*x)
      p7: amplitude for tanh(s*x)
      p8: scale s for tanh (clipped)
      p9: amplitude for exp(-w*(x-c)^2) with w=1, c=0
    
    Model:
      y = p0 + p1*x + p2*x^2 + p3*x^3
          + p4*sin(p5*x) + p6*cos(p5*x)
          + p7*tanh(p8*x)
          + p9*exp(-x^2)
    """
    p = params
    
    # Clip shape parameters to safe ranges
    p5 = np.clip(p[5], -10.0, 10.0)   # frequency for sin/cos
    p8 = np.clip(p[8], -10.0, 10.0)   # scale for tanh
    
    x2 = x * x
    x3 = x2 * x
    
    # Compute basis terms
    term_sin = np.sin(p5 * x)
    term_cos = np.cos(p5 * x)
    term_tanh = np.tanh(p8 * x)
    term_gauss = np.exp(-x2)
    
    result = (
        p[0] * 1.0 +
        p[1] * x +
        p[2] * x2 +
        p[3] * x3 +
        p[4] * term_sin +
        p[6] * term_cos +
        p[7] * term_tanh +
        p[9] * term_gauss
    )
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
