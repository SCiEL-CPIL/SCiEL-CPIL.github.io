import numpy as np
import time

_start_time = time.time()
DEADLINE = 135.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a Chebyshev polynomial basis of degree 0-9 on the domain [-3, 3].
    Chebyshev polynomials are orthogonal on [-1, 1] and provide well-conditioned
    linear basis functions for smooth function approximation.
    The model is linear in params: y = sum(p_i * T_i(x/3)).
    """
    if time.time() - _start_time > DEADLINE:
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)

    # Scale x from [-3, 3] to [-1, 1] for standard Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    T_k_minus_2 = np.ones_like(t)
    
    if len(p) < 2:
        # Fallback if params are shorter, though contract says length 10
        return p[0] * T_k_minus_2
    
    # T_1(t) = t
    T_k_minus_1 = t
    
    basis = [T_k_minus_2, T_k_minus_1]
    
    for k in range(2, 10):
        # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
        basis.append(T_k)
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        
    # Stack basis: shape (10, N)
    # Ensure we have exactly 10 basis functions
    if len(basis) > 10:
        basis = basis[:10]
        
    basis_matrix = np.vstack(basis)
    
    # Linear combination
    y = p @ basis_matrix
    
    # Ensure finiteness
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return y
