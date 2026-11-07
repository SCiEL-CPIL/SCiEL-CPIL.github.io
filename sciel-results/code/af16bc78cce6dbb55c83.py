import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a dense Chebyshev polynomial basis.
    
    Strategy:
    - Uses Chebyshev polynomials of the first kind T_0(x) through T_9(x).
    - The input domain is [-2, 2]. We map x to t in [-1, 1] via t = x / 2.
    - Chebyshev polynomials are orthogonal on [-1, 1] with respect to the 
      weight 1/sqrt(1-t^2), but even without the weight, they provide much 
      better numerical conditioning than monomials (1, x, x^2, ...) on 
      bounded intervals.
    - Since the model is linear in the parameters (y = sum p_i * T_i(x)), 
      BFGS will converge rapidly to the global minimum of the MSE, avoiding 
      the local minima traps associated with non-linear basis functions 
      (like RBF centers or trigonometric frequencies).
    - This allows the 10 parameters to span a 10-dimensional subspace of 
      continuous functions, capable of fitting any smooth function well 
      within the resolution limit of 10 coefficients.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    # Ensure x is 1D
    x = np.asarray(x).flatten()
    
    # Map x from [-2, 2] to t in [-1, 1]
    # This is crucial for Chebyshev stability
    t = x / 2.0
    
    # Pre-compute Chebyshev polynomials T_0(t) to T_9(t)
    # Recurrence relation: T_0(t) = 1, T_1(t) = t, T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # Initialize array to hold basis functions
    # Shape: (N_points, 10)
    # We will compute them iteratively to avoid storing large intermediate arrays
    # and to keep memory footprint low.
    
    # T_0
    T_curr = np.ones_like(t)
    # T_1
    T_next = t.copy()
    
    # We need to combine them with params.
    # result = p[0]*T_0 + p[1]*T_1 + ... + p[9]*T_9
    
    result = params[0] * T_curr
    if len(params) > 1:
        result = result + params[1] * T_next
    
    # Compute T_2 to T_9
    for n in range(2, 10):
        if n >= len(params):
            break
            
        # T_n = 2*t*T_{n-1} - T_{n-2}
        T_new = 2.0 * t * T_next - T_curr
        
        result = result + params[n] * T_new
        
        # Update for next iteration
        T_curr = T_next
        T_next = T_new
        
        # Check deadline
        if time.time() > deadline:
            break
            
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
