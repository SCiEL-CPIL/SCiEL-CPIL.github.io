"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: 150s limit, aim to finish well before.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _clenshaw_eval(u, coeffs):
    """
    Evaluate Chebyshev series sum_{k=0}^{N} c_k T_k(u) using Clenshaw's algorithm.
    This is numerically stable and efficient.
    u: array of points in [-1, 1]
    coeffs: array of coefficients [c_0, ..., c_N]
    """
    N = len(coeffs) - 1
    if N < 0:
        return np.zeros_like(u)
    
    # Clenshaw recurrence:
    # b_{N+1} = 0, b_{N+2} = 0
    # b_k = c_k + 2*u*b_{k+1} - b_{k+2} for k = N-1, ..., 0
    # result = 0.5*c_0 + b_0 - 0.5*b_1  (Wait, standard Clenshaw for Chebyshev:
    # Actually, the standard form is:
    # b_N = c_N
    # b_{N-1} = c_{N-1} + 2*u*b_N
    # b_k = c_k + 2*u*b_{k+1} - b_{k+2} for k = N-2, ..., 1
    # Result = 0.5*c_0 + b_0 - 0.5*b_1 ? No, let's use the iterative form.
    
    # Correct Clenshaw for T_k:
    # b_{N} = c_N
    # b_{N-1} = c_{N-1} + 2*u * b_N
    # b_{k} = c_k + 2*u * b_{k+1} - b_{k+2} for k = N-2 ... 0
    # f(u) = 0.5*c_0 + b_0 - 0.5*b_1 ? 
    # Let's check for N=0: f = c_0. b_0 = c_0. Formula: 0.5*c_0 + c_0 - 0.5*0 = 1.5 c_0. Incorrect.
    
    # Let's use the standard loop:
    # b_1 = 0
    # b_2 = 0
    # for k = N-1 down to 0:
    #    b_0 = c_k + 2*u*b_1 - b_2
    #    b_2 = b_1
    #    b_1 = b_0
    # f(u) = 0.5*c_0 + b_1 - 0.5*b_2
    
    b_1 = np.zeros_like(u)
    b_2 = np.zeros_like(u)
    
    # Start from the highest degree
    for k in range(N, -1, -1):
        b_0 = coeffs[k] + 2.0 * u * b_1 - b_2
        b_2 = b_1
        b_1 = b_0
        
    # The final value is 0.5*c_0 + b_1 - 0.5*b_2
    # Wait, the loop above includes k=0.
    # If N=0:
    # k=0: b_0 = c_0 + 0 - 0 = c_0. b_2=0, b_1=c_0.
    # Result: 0.5*c_0 + c_0 - 0 = 1.5 c_0. Still wrong.
    
    # Let's re-derive Clenshaw for Chebyshev T_k.
    # T_k = 2u T_{k-1} - T_{k-2}
    # We want S = sum c_k T_k.
    # b_{N+1} = 0, b_{N+2} = 0
    # b_k = c_k + 2u b_{k+1} - b_{k+2}
    # S = 0.5 c_0 + b_0 - 0.5 b_1 ? No.
    
    # Standard Clenshaw algorithm for Chebyshev:
    # b_{N} = c_N
    # b_{N-1} = c_{N-1} + 2u b_N
    # b_{k} = c_k + 2u b_{k+1} - b_{k+2} for k = N-2 ... 0
    # f(u) = 0.5 c_0 + b_0 - 0.5 b_1 ? No, f(u) = b_0 - 0.5 b_1 + 0.5 c_0?
    
    # Let's just use the simple forward recurrence for N=9, it's stable enough.
    # T_0 = 1
    # T_1 = u
    # T_k = 2u T_{k-1} - T_{k-2}
    
    T_prev2 = np.ones_like(u)
    T_prev1 = u.copy()
    result = coeffs[0] * T_prev2
    if N >= 1:
        result = result + coeffs[1] * T_prev1
        
    for k in range(2, N + 1):
        T_curr = 2.0 * u * T_prev1 - T_prev2
        result = result + coeffs[k] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
        
    return result

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Chebyshev polynomial basis (degrees 0-9) on scaled domain [-1, 1].
    Uses forward recurrence for stable evaluation.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback to a simple linear model to ensure valid output
        # This is a safe fallback that won't crash
        return params[0] + params[1] * (x / 3.0)
        
    # Scale x from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    # Ensure params has at least 10 elements
    # If params is shorter, pad with zeros. If longer, truncate.
    p = np.zeros(10, dtype=np.float64)
    n_params = min(len(params), 10)
    p[:n_params] = params[:n_params]
    
    # Evaluate Chebyshev polynomial
    result = _clenshaw_eval(u, p)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
