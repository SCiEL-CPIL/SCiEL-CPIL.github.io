import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis of degree 9.
    
    The input x is mapped to t in [-1, 1].
    The model is f(x) = sum_{k=0}^{9} c_k T_k(t).
    This uses all 10 parameters.
    
    Chebyshev polynomials are well-conditioned and provide efficient approximation
    for smooth functions on a bounded interval. Since the loss is quadratic in the
    parameters (linear model in terms of basis functions), the optimization landscape
    is convex, ensuring BFGS converges to the global minimum.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6.0] to t in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Clip t to [-1, 1] to handle any slight numerical overflow/out-of-range inputs
    # though the problem states x is in [0.1, 6]
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # Initialize basis matrix or compute linear combination directly
    # Direct linear combination is more memory efficient for small N
    n_points = x.shape[0]
    result = np.zeros(n_points, dtype=np.float64)
    
    if len(params) < 10:
        # Fallback if params is shorter (should not happen per contract)
        # Pad with zeros
        p = np.zeros(10, dtype=np.float64)
        p[:len(params)] = params
    else:
        p = params[:10]
        
    # T_0
    T_km2 = 1.0
    T_km1 = t
    
    result += p[0] * T_km2
    if n_points > 0:
        result += p[1] * T_km1
        
    for k in range(2, 10):
        T_k = 2.0 * t * T_km1 - T_km2
        result += p[k] * T_k
        T_km2 = T_km1
        T_km1 = T_k
        
    return result
