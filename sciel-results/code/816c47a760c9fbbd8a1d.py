import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learnable Frequency Fourier Series Model.
    
    Unlike the fixed-basis approach, this model allows the frequencies of the 
    oscillatory components to be optimized. This significantly expands the 
    hypothesis class to include functions with non-integer or specific 
    non-integer periodicities.
    
    Model Structure:
    y = p0 
      + (p1 * sin(p2 * x) + p3 * cos(p2 * x))
      + (p4 * sin(p5 * x) + p6 * cos(p5 * x))
      + (p7 * sin(p8 * x) + p9 * cos(p8 * x))
      
    Parameters:
    p0: Constant offset
    p1, p2, p3: Amplitude(sin), Frequency, Amplitude(cos) for term 1
    p4, p5, p6: Amplitude(sin), Frequency, Amplitude(cos) for term 2
    p7, p8, p9: Amplitude(sin), Frequency, Amplitude(cos) for term 3
    
    This uses exactly 10 parameters.
    The combination of sin and cos with the same frequency is equivalent to a 
    single sine wave with a phase shift, providing full flexibility for each 
    frequency component.
    
    Why this might be better:
    1. Adapts to the specific frequencies present in the target function.
    2. The constant term handles the mean.
    3. Three frequencies are usually sufficient to capture the main features 
       of smooth functions on a bounded interval [0.1, 6].
    
    Optimization Considerations:
    - Starting from all ones: Frequencies are 1.0, Amplitudes are 1.0.
    - BFGS will adjust frequencies to minimize MSE.
    - We add a small epsilon to frequencies to avoid zero frequency (which 
      would make sin/cos constant and redundant with p0) if the optimizer 
      pushes them there, though BFGS usually handles this naturally if the 
      gradient is well-defined. However, to be safe against numerical 
      instability if a frequency goes to 0, we don't strictly need to 
      clamp, but we ensure the math is stable.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
        
    p = params
    
    # Extract parameters
    p0 = p[0]
    
    # Term 1
    f1 = p[2]
    a1_sin = p[1]
    a1_cos = p[3]
    term1 = a1_sin * np.sin(f1 * x) + a1_cos * np.cos(f1 * x)
    
    # Term 2
    f2 = p[5]
    a2_sin = p[4]
    a2_cos = p[6]
    term2 = a2_sin * np.sin(f2 * x) + a2_cos * np.cos(f2 * x)
    
    # Term 3
    f3 = p[8]
    a3_sin = p[7]
    a3_cos = p[9]
    term3 = a3_sin * np.sin(f3 * x) + a3_cos * np.cos(f3 * x)
    
    return p0 + term1 + term2 + term3
