import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Unpack parameters
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = params
        
        # Polynomial part
        y = a0 + a1 * x + a2 * x**2
        
        # Sinusoidal part with learnable frequency
        y += a3 * np.sin(a4 * x)
        
        # Cosinusoidal part with learnable frequency
        y += a5 * np.cos(a6 * x)
        
        # Hyperbolic tangent part with learnable frequency
        y += a7 * np.tanh(a8 * x)
        
        # Gaussian bump part (fixed width for stability, learnable amplitude)
        # Using exp(-x^2) is safe
        y += a9 * np.exp(-x**2)
        
        return y
