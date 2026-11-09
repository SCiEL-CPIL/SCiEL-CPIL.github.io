"""Parametric limit-cycle oscillator with smooth bounded nonlinearities."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Structure: a = -k(x) * x - c(v) * v + g(x,v) + F(t)
    where:
      k(x) = p0 + p1 * tanh(p2 * x) + p3 * tanh(p2 * x)^2
      c(v) = p4 + p5 * tanh(p6 * v) + p7 * tanh(p6 * v)^2
      g(x,v) = p8 * tanh(x) * tanh(v)
      F(t) = p9 * sin(p2 * t + p6)
    
    params: array of length 10
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinear stiffness:
    tx = np.tanh(p2 * x)
    kx = p0 + p1 * tx + p3 * tx * tx
    
    # Bounded nonlinear damping:
    tv = np.tanh(p6 * v)
    cv = p4 + p5 * tv + p7 * tv * tv
    
    # Bounded cross-coupling: g(x,v) = p8 * tanh(x) * tanh(v)
    cross = p8 * np.tanh(x) * np.tanh(v)
    
    # Driving force: F(t) = p9 * sin(p2 * t + p6)
    F = p9 * np.sin(p2 * t + p6)
    
    # Acceleration: a = -k(x)*x - c(v)*v + cross + F(t)
    a = -kx * x - cv * v + cross + F
    
    return a
# EVOLVE-BLOCK-END
