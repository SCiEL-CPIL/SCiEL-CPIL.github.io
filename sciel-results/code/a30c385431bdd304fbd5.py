"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian-Harmonic-Polynomial hybrid model for a nonlinear oscillator.
    
    Physical rationale:
    - A Gaussian term exp(-p1*(x^2+v^2)) models energy-dependent restoring/damping
      (softening or hardening near origin vs. far field).
    - Harmonic terms cos(p3*t) and sin(p3*t) with a fitted frequency p3 capture
      periodic driving or natural oscillation.
    - Polynomial terms (linear, interaction, quadratic) capture local nonlinearities.
    
    Structure:
    y = p0 * exp(-p1 * (x^2 + v^2))
        + p2 * cos(p3 * t) + p4 * sin(p3 * t)
        + p5 * x + p6 * v + p7 * x * v
        + p8 * x^2 + p9 * v^2
    
    Params:
    p0: Amplitude of the Gaussian energy term
    p1: Decay rate (inverse width squared) of the Gaussian
    p2: Amplitude of cosine drive/oscillation
    p3: Angular frequency of the harmonic terms
    p4: Amplitude of sine drive/oscillation
    p5: Linear coefficient in x
    p6: Linear coefficient in v
    p7: Interaction coefficient x*v
    p8: Quadratic coefficient in x
    p9: Quadratic coefficient in v
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Gaussian energy-dependent term
    r2 = x ** 2 + v ** 2
    gaussian_term = p0 * np.exp(-p1 * r2)
    
    # Harmonic drive/oscillation with fitted frequency
    # Use np.cos and np.sin which handle array inputs naturally
    drive = p2 * np.cos(p3 * t) + p4 * np.sin(p3 * t)
    
    # Polynomial corrections
    poly = (p5 * x + p6 * v + p7 * x * v 
            + p8 * (x ** 2) + p9 * (v ** 2))
    
    return gaussian_term + drive + poly
# EVOLVE-BLOCK-END
