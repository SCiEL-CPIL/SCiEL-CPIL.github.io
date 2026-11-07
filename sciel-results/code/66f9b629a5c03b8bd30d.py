import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic Regression using a Hybrid Chebyshev-Exponential Basis.
    
    Strategy:
    - Combine the structural advantage of Chebyshev polynomials (well-conditioned, 
      near-orthogonal on the bounded domain) with specific non-polynomial features 
      (exponential decay, logarithm, reciprocal) that often appear in natural functions.
    - The model is linear in parameters, ensuring a convex optimization landscape 
      for BFGS.
    - Domain [0.1, 6] is mapped to [-1, 1] for the Chebyshev terms to ensure stability.
    - Basis functions:
      1. Chebyshev T_0(t) to T_5(t) (6 terms)
      2. exp(-x)
      3. log(x)
      4. 1/x
      5. sin(x)
    - This hybrid approach captures both the smooth global trends (via Chebyshev) 
      and specific local behaviors/asymptotes (via exp, log, 1/x, sin) more effectively 
      than a pure polynomial basis, potentially fitting the hidden function better 
      while maintaining numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    n = len(x)
    if n == 0:
        return np.array([])
        
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2x - (max + min)) / (max - min)
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0(t) to T_5(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((6, n), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 6):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # Compute non-polynomial basis functions
    # 6. Exponential decay
    b_exp = np.exp(-x)
    
    # 7. Logarithm
    b_log = np.log(x)
    
    # 8. Reciprocal
    b_recip = 1.0 / x
    
    # 9. Sinusoidal
    b_sin = np.sin(x)
    
    # 10. Cosine (adding this to round out the trigonometric pair, or maybe another Chebyshev? 
    # Let's stick to the plan: 6 Chebyshev + exp + log + 1/x + sin = 10 terms.
    # Wait, 6 + 4 = 10. Correct.
    
    # Stack basis functions into a matrix (N, 10)
    # T is (6, N), so we need to transpose it to (N, 6)
    B_cheb = T.T
    
    # Combine all basis functions
    B = np.column_stack((B_cheb, b_exp, b_log, b_recip, b_sin))
    
    # Compute linear combination
    # B is (N, 10), p is (10,)
    result = B @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
