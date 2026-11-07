"""Hybrid basis: sigmoids + polynomial for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def _stable_sigmoid(z):
    """Numerically stable sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: low-order polynomial + shifted/scaled sigmoids.
    
    y = a0 + a1*x + a2*x^2 + a3*x^3 
      + w0 * sigmoid(s*(x - (offset - 1.5*s)))
      + w1 * sigmoid(s*(x - (offset - 0.5*s)))
      + w2 * sigmoid(s*(x - (offset + 0.5*s)))
      + w3 * sigmoid(s*(x - (offset + 1.5*s)))
    
    params (length 10):
      [0]=a0, [1]=a1, [2]=a2, [3]=a3,
      [4]=w0, [5]=w1, [6]=w2, [7]=w3,
      [8]=s (shared scale), [9]=offset (center of sigmoid group)
    
    At params=all_ones: 
      poly = 1 + x + x^2 + x^3
      sigmoids centered at x - {−0.5, 0.5, 1.5, 2.5} with scale 1
    """
    x = np.asarray(x, dtype=float)
    
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    
    w0 = params[4]
    w1 = params[5]
    w2 = params[6]
    w3 = params[7]
    
    s = params[8]
    offset = params[9]
    
    # Polynomial part
    x2 = x * x
    x3 = x2 * x
    poly = a0 + a1 * x + a2 * x2 + a3 * x3
    
    # Sigmoid parts with shared scale and offset
    z0 = s * (x - (offset - 1.5 * s))
    z1 = s * (x - (offset - 0.5 * s))
    z2 = s * (x - (offset + 0.5 * s))
    z3 = s * (x - (offset + 1.5 * s))
    
    sig0 = _stable_sigmoid(z0)
    sig1 = _stable_sigmoid(z1)
    sig2 = _stable_sigmoid(z2)
    sig3 = _stable_sigmoid(z3)
    
    result = poly + w0 * sig0 + w1 * sig1 + w2 * sig2 + w3 * sig3
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
