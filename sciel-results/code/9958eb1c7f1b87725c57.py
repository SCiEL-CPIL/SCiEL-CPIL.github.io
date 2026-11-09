"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with fixed orthogonal frequencies.
    
    We use a constant term, sine and cosine terms up to the 5th harmonic.
    The frequencies are chosen to be orthogonal over the domain [-2, 2].
    For a domain of length 4 (from -2 to 2), the fundamental frequency for
    orthogonality is pi / 2 (since period = 4, so 2*pi/T = pi/2).
    
    Basis functions:
      1 (constant)
      sin(k * pi/2 * x), cos(k * pi/2 * x) for k = 1..5
    
    This gives 1 + 10 = 11 basis functions. However, params has length 10.
    We'll use:
      params[0] * 1
      params[1] * sin(1 * pi/2 * x)
      params[2] * cos(1 * pi/2 * x)
      params[3] * sin(2 * pi/2 * x)
      params[4] * cos(2 * pi/2 * x)
      params[5] * sin(3 * pi/2 * x)
      params[6] * cos(3 * pi/2 * x)
      params[7] * sin(4 * pi/2 * x)
      params[8] * cos(4 * pi/2 * x)
      params[9] * sin(5 * pi/2 * x)
    
    Wait, that's 10 terms but I dropped cos(5). Let me reconsider.
    
    Actually, with 10 params, I can do:
      1 constant
      4 sine terms (k=1..4)
      4 cosine terms (k=1..4)
      1 more... maybe sin(5) or cos(5)
    
    Let me use:
      params[0]: constant
      params[1]: sin(pi/2 * x)
      params[2]: cos(pi/2 * x)
      params[3]: sin(pi * x)
      params[4]: cos(pi * x)
      params[5]: sin(3*pi/2 * x)
      params[6]: cos(3*pi/2 * x)
      params[7]: sin(2*pi * x)
      params[8]: cos(2*pi * x)
      params[9]: sin(5*pi/2 * x)
    """
    # Frequencies: k * pi/2 for k = 1, 2, 3, 4, 5
    # This gives orthogonal basis on [-2, 2]
    
    # Precompute the basis
    # x is 1D array of shape (n,)
    
    # Constant term
    out = params[0] * np.ones_like(x, dtype=float)
    
    # Sine and cosine terms for k = 1 to 4 (8 terms), plus one more sine for k=5
    freqs = [np.pi / 2 * k for k in range(1, 5)]
    
    for i, f in enumerate(freqs):
        out += params[1 + 2*i] * np.sin(f * x)
        out += params[2 + 2*i] * np.cos(f * x)
    
    # 5th harmonic sine (params[9])
    f5 = 5 * np.pi / 2
    out += params[9] * np.sin(f5 * x)
    
    return out
# EVOLVE-BLOCK-END
