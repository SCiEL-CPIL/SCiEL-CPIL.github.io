"""Symbolic regression with nonlinear parameterization for richer function space."""
import numpy as np
import time

# EVOLVE-BLOCK-START


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using 10 parameters with nonlinear parameterization.
    
    Parameters control the shape of components:
    - p[0]: amplitude of sin(freq1*x + phase1)
    - p[1]: frequency of first sinusoid
    - p[2]: phase of first sinusoid
    - p[3]: amplitude of Gaussian
    - p[4]: center of Gaussian
    - p[5]: width of Gaussian (inverse)
    - p[6]: amplitude of tanh transition
    - p[7]: center of tanh transition
    - p[8]: steepness of tanh transition
    - p[9]: constant offset
    
    This nonlinear parameterization allows the model to fit functions with
    variable frequencies, locations, and widths - a much richer space than
    a fixed linear basis.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters; pad or truncate if needed
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Clip parameters to avoid numerical issues
    # Frequencies and widths should be positive; use softplus-like clipping
    freq1 = np.tanh(p[1]) * 5.0 + 1.0  # frequency in [1-5, 1+5] = [-4, 6], but let's ensure positive
    freq1 = np.maximum(np.abs(freq1), 0.1)  # ensure positive frequency
    
    width = np.tanh(p[5]) * 3.0 + 2.0  # width in [-1, 5], ensure positive
    width = np.maximum(np.abs(width), 0.1)  # ensure positive width
    
    steepness = np.tanh(p[8]) * 5.0 + 1.0  # steepness in [-4, 6]
    steepness = np.maximum(np.abs(steepness), 0.1)  # ensure positive steepness
    
    # Component 1: Sinusoid with variable frequency and phase
    comp1 = p[0] * np.sin(freq1 * x + p[2])
    
    # Component 2: Gaussian with variable center and width
    comp2 = p[3] * np.exp(-((x - p[4]) ** 2) / (2.0 * width ** 2))
    
    # Component 3: Tanh transition with variable center and steepness
    comp3 = p[6] * np.tanh(steepness * (x - p[7]))
    
    # Component 4: Constant offset
    comp4 = p[9]
    
    result = comp1 + comp2 + comp3 + comp4
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result


# EVOLVE-BLOCK-END
