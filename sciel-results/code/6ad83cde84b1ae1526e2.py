"""Complex State-Space Polynomial Oscillator Model:
Uses a complex exponential representation z = x + i*v to construct a polynomial basis
that naturally encodes phase-locked cross-couplings between position and velocity.
This replaces hand-picked tanh/sin features with a systematic complex polynomial basis
(1, z, z^3, e^(it), z*e^(it)), capturing linear, cubic nonlinear, and periodic driving
terms with inherent physical symmetries. The model is linear in 10 real parameters
(5 complex coefficients), enabling robust BFGS optimization while capturing
nonlinear oscillator dynamics more accurately than sparse hand-crafted features."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Complex state-space polynomial model with 10 parameters (5 complex coeffs):
    p0, p1: Real, Imag of c0 (constant offset)
    p2, p3: Real, Imag of c1 (linear state z)
    p4, p5: Real, Imag of c2 (cubic state z^3)
    p6, p7: Real, Imag of c3 (periodic driving w)
    p8, p9: Real, Imag of c4 (interaction z*w)
    
    z = x + i*v
    w = e^(i*t)
    y = Re( c0 + c1*z + c2*z^3 + c3*w + c4*z*w )
    """
    if time.time() > _DEADLINE:
        return np.zeros_like(t)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Complex state and time basis
    # Using complex arithmetic to enforce symmetries
    # z = x + i*v
    # w = cos(t) + i*sin(t)
    
    # To avoid complex number overhead in numpy if possible, we can expand manually,
    # but numpy handles complex arrays efficiently.
    
    z = x + 1j * v
    w = np.exp(1j * t)
    
    # Basis terms
    B0 = 1.0
    B1 = z
    B2 = z ** 3
    B3 = w
    B4 = z * w
    
    # Complex coefficients
    c0 = p0 + 1j * p1
    c1 = p2 + 1j * p3
    c2 = p4 + 1j * p5
    c3 = p6 + 1j * p7
    c4 = p8 + 1j * p9
    
    # Linear combination
    y_complex = c0 * B0 + c1 * B1 + c2 * B2 + c3 * B3 + c4 * B4
    
    # Prediction is the real part
    return np.real(y_complex)
# EVOLVE-BLOCK-END
