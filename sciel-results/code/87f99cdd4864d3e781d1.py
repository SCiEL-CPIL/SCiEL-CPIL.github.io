"""
Strategy: Chebyshev-Activated Multiplicative Basis.
Instead of summing independent RBFs and polynomials, this approach constructs a 
feature space where global polynomial structure (Chebyshev) is modulated by 
nonlinear activations (tanh, sin). This allows the model to capture complex 
non-periodic nonlinearities and interactions that are hard for pure polynomials 
or local RBFs to represent efficiently.

The 10 parameters correspond to 10 basis functions:
1. T0 * tanh(x)
2. T1 * tanh(x)
3. T2 * tanh(x)
4. T0 * sin(x)
5. T1 * sin(x)
6. T2 * sin(x)
7. T0 * tanh(2x)
8. T1 * tanh(2x)
9. T2 * sin(2x)
10. T3 * tanh(x)

This provides a diverse set of global, smooth, non-linear features.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 10 Chebyshev-activated basis functions.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Predicted y array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params is length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    elif len(p) > 10:
        p = p[:10]

    # Compute Chebyshev polynomials T0 to T3
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    x2 = x * x
    x3 = x2 * x
    
    t0 = np.ones_like(x)
    t1 = x
    t2 = 2.0 * x2 - 1.0
    t3 = 4.0 * x3 - 3.0 * x

    # Compute Activations
    # tanh(x) and tanh(2x) provide smooth non-linear saturation
    # sin(x) and sin(2x) provide periodic non-linear features
    tanh_x = np.tanh(x)
    tanh_2x = np.tanh(2.0 * x)
    sin_x = np.sin(x)
    sin_2x = np.sin(2.0 * x)

    # Construct 10 Basis Functions (Products of Chebyshev and Activations)
    # This creates a richer feature space than simple sums
    
    # Group 1: T0-T2 * tanh(x)
    b0 = t0 * tanh_x
    b1 = t1 * tanh_x
    b2 = t2 * tanh_x
    
    # Group 2: T0-T2 * sin(x)
    b3 = t0 * sin_x
    b4 = t1 * sin_x
    b5 = t2 * sin_x
    
    # Group 3: T0-T1 * tanh(2x)
    b6 = t0 * tanh_2x
    b7 = t1 * tanh_2x
    
    # Group 4: T2 * sin(2x) and T3 * tanh(x)
    b8 = t2 * sin_2x
    b9 = t3 * tanh_x

    # Linear Combination
    # Weights are p[0] to p[9]
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
