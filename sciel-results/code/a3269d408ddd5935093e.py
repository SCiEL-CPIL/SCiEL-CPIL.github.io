"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid polynomial + trigonometric basis (10 terms).
    
    The hidden function may contain non-polynomial (e.g., sinusoidal) structure
    that a pure polynomial basis approximates poorly. By mixing low-order
    polynomial terms with sine terms at multiple frequencies, we give the
    BFGS fitter a richer representational space while keeping the model
    linear-in-parameters (so BFGS converges reliably).
    
    Basis (10 terms):
      p[0]:  1                          (constant)
      p[1]:  u                          (linear x, normalized)
      p[2]:  v                          (linear y, normalized)
      p[3]:  w                          (linear z, normalized)
      p[4]:  u * v                      (cross x*y)
      p[5]:  sin(2*pi*u)                (sine in x)
      p[6]:  sin(2*pi*v)                (sine in y)
      p[7]:  sin(2*pi*w)                (sine in z)
      p[8]:  sin(4*pi*u)                (2nd harmonic x)
      p[9]:  sin(4*pi*v)                (2nd harmonic y)
    
    Inputs normalized from [0.5, 2.0] to [-1, 1] for well-conditioned basis.
    """
    elapsed = time.time() - _start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return np.ones_like(x) * params[0]
    
    p = params
    
    # Normalize inputs to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Polynomial terms
    b0 = np.ones_like(u)
    b1 = u
    b2 = v
    b3 = w
    b4 = u * v
    
    # Trigonometric terms (frequencies chosen to span the normalized range)
    # sin(2*pi*t) has period 1 in t, so on [-1,1] it completes 2 full cycles
    # sin(4*pi*t) has period 0.5, completing 4 full cycles on [-1,1]
    two_pi = 2.0 * np.pi
    four_pi = 4.0 * np.pi
    
    b5 = np.sin(two_pi * u)
    b6 = np.sin(two_pi * v)
    b7 = np.sin(two_pi * w)
    b8 = np.sin(four_pi * u)
    b9 = np.sin(four_pi * v)
    
    return (p[0] * b0 +
            p[1] * b1 +
            p[2] * b2 +
            p[3] * b3 +
            p[4] * b4 +
            p[5] * b5 +
            p[6] * b6 +
            p[7] * b7 +
            p[8] * b8 +
            p[9] * b9)
# EVOLVE-BLOCK-END
