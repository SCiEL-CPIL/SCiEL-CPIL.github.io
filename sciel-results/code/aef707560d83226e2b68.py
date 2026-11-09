import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Fourier series basis.
    
    Parameters:
    x (np.ndarray): Input array of shape (N,), values in [-3, 3].
    params (np.ndarray): Array of 10 coefficients to be fitted.
    
    Returns:
    np.ndarray: Predictions of shape (N,).
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    if len(x) == 0:
        return np.zeros_like(x)
        
    # Initialize result with constant term
    result = np.full_like(x, params[0])
    
    # Frequencies: 1, 2, 3, 4, 5
    # We have 9 params left.
    # p1, p2: cos(1x), sin(1x)
    # p3, p4: cos(2x), sin(2x)
    # p5, p6: cos(3x), sin(3x)
    # p7, p8: cos(4x), sin(4x)
    # p9: cos(5x)
    
    # We can vectorize the computation
    # Basis:
    # cos(x), sin(x)
    # cos(2x), sin(2x)
    # ...
    
    # Check deadline
    if time.time() > deadline:
        return result
        
    # Compute trig functions
    # To save time, compute for each frequency
    # Note: x is an array
    
    # Frequency 1
    c1 = np.cos(x)
    s1 = np.sin(x)
    result += params[1] * c1 + params[2] * s1
    
    if time.time() > deadline:
        return result
        
    # Frequency 2
    c2 = np.cos(2.0 * x)
    s2 = np.sin(2.0 * x)
    result += params[3] * c2 + params[4] * s2
    
    if time.time() > deadline:
        return result
        
    # Frequency 3
    c3 = np.cos(3.0 * x)
    s3 = np.sin(3.0 * x)
    result += params[5] * c3 + params[6] * s3
    
    if time.time() > deadline:
        return result
        
    # Frequency 4
    c4 = np.cos(4.0 * x)
    s4 = np.sin(4.0 * x)
    result += params[7] * c4 + params[8] * s4
    
    if time.time() > deadline:
        return result
        
    # Frequency 5 (Cosine only, to fit in 10 params)
    c5 = np.cos(5.0 * x)
    result += params[9] * c5
    
    return result
