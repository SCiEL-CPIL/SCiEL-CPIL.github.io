"""
Strategy: Non-linear parameterized model using a small feedforward neural network structure.
Instead of a linear combination of fixed basis functions (like Chebyshev), we use a sum of 
non-linear activation functions (tanh) where the parameters control the shape (slope, offset) 
and amplitude of each component. This allows the model to fit a wider variety of functional 
forms (sigmoidal, step-like, polynomial-like) that a fixed linear basis might miss or 
approximate poorly, especially if the target function has non-polynomial characteristics 
or if the linear system was ill-conditioned for the specific target.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y using a sum of 3 tanh neurons plus a bias.
    
    params: array of length 10.
    Structure:
    y(x) = w1 * tanh(w2 * x + w3) + 
           w4 * tanh(w5 * x + w6) + 
           w7 * tanh(w8 * x + w9) + 
           w10
    
    where w1..w10 are the 10 parameters.
    """
    x = np.asarray(x, dtype=float)
    
    # Unpack parameters
    # Ensure params is 1D and length 10
    if len(params) != 10:
        # Fallback if shape is weird, though contract says length 10
        p = np.ones(10)
        p[:min(10, len(params))] = params[:min(10, len(params))]
    else:
        p = params
        
    w1, w2, w3 = p[0], p[1], p[2]
    w4, w5, w6 = p[3], p[4], p[5]
    w7, w8, w9 = p[6], p[7], p[8]
    w10 = p[9]
    
    # Compute the non-linear components
    # tanh is bounded in [-1, 1], so w1, w4, w7 control amplitude.
    # w2, w5, w8 control the "steepness" or scale of x.
    # w3, w6, w9 control the horizontal shift.
    
    term1 = w1 * np.tanh(w2 * x + w3)
    term2 = w4 * np.tanh(w5 * x + w6)
    term3 = w7 * np.tanh(w8 * x + w9)
    
    y = term1 + term2 + term3 + w10
    
    return y

# EVOLVE-BLOCK-END
