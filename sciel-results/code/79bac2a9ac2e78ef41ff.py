import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Parametric Cubic Spline with Fixed Knots
    
    The model is a continuous piecewise cubic polynomial defined by 10 parameters.
    
    Why this works:
    1. Structural Flexibility: Unlike RBFs which are globally smooth (C-infinity), 
       splines can have sharp changes in curvature (C2 continuous). This is crucial 
       if the target function contains kinks or rapid local changes that a smooth 
       Gaussian basis would smooth out or fail to capture efficiently.
    2. Exact Representation: If the true function is a polynomial or piecewise 
       polynomial, this model can represent it exactly (or near-exactly) with the 
       right parameters, whereas RBFs would require many terms to approximate a 
       straight line or polynomial.
    3. Linear in Parameters: The model is linear in the 10 parameters (coefficients 
       of the spline segments). This ensures BFGS converges to the global minimum 
       of the training MSE, avoiding local minima issues associated with learning 
       knot positions or non-linear basis parameters.
    4. Fixed Knots: We use 4 fixed interior knots to divide the domain [0.1, 6.0] 
       into 5 segments. This uses 10 parameters (4 coefficients per segment? No, 
       continuity constraints reduce DoF. Let's use a simpler basis).
       
    Revised Basis Approach:
    Instead of raw spline coefficients which require complex continuity constraints 
    to maintain linearity and differentiability, we use a **B-Spline Basis** or 
    **Piecewise Polynomial Basis** that is inherently continuous.
    
    However, implementing a full B-Spline basis with 10 parameters is complex.
    
    Alternative: **Piecewise Linear + Quadratic/Cubic Correction**
    Let's use a basis of 10 "hat" functions (piecewise linear) and 10 "bump" 
    functions? No, that's RBF-like.
    
    Let's stick to a **Global Polynomial + Local Piecewise Linear Corrections**.
    Actually, the most robust "structural" change from RBF is a **Fourier Series** 
    with learnable frequencies? No, that's non-linear.
    
    Let's use a **Chebyshev Polynomial Basis** of degree 9.
    Why? 
    - Chebyshev polynomials are orthogonal on [-1, 1].
    - They provide the best uniform approximation for smooth functions.
    - They are linear in parameters.
    - They handle the entire domain [0.1, 6] well if scaled.
    - Unlike RBFs, they don't have "local" artifacts; they are global but 
      spectrally efficient for smooth functions.
    - If the function is a polynomial, it's exact.
    - If the function is smooth but non-polynomial, high-degree Chebyshev 
      converges faster than Gaussian RBFs for global trends.
    
    Wait, the inspiration mentioned "Learnable-Width GP" scored 31k. 
    The current RBF scored 639k. 
    The "Step 1" inspiration (hybrid sinusoids) scored 832k (worse).
    
    Let's try **Chebyshev Polynomials**. It is a fundamentally different basis 
    (global orthogonal polynomials vs local radial functions).
    
    Parameters: 10 coefficients for T_0(x) to T_9(x).
    Domain scaling: Map [0.1, 6.0] to [-1, 1].
    x_scaled = (2*x - (0.1+6.0)) / (6.0-0.1)
    
    This is linear in params, stable, and different from RBF.
    """
    
    start_time = time.time()
    TIME_LIMIT = 145.0

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # Scale x from [0.1, 6.0] to [-1, 1] for Chebyshev stability
    # x_min = 0.1, x_max = 6.0
    # x_scaled = (2*x - x_min - x_max) / (x_max - x_min)
    x_min = 0.1
    x_max = 6.0
    x_scaled = (2.0 * x - x_min - x_max) / (x_max - x_min)
    
    # Clip to [-1, 1] to avoid numerical overflow in Chebyshev recurrence 
    # if x is slightly out of bounds (though data is uniform in [0.1, 6])
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    
    # Compute Chebyshev Polynomials T_0 to T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    
    n_terms = 10
    T = np.zeros((len(x), n_terms), dtype=np.float64)
    
    if len(x) > 0:
        T[:, 0] = 1.0
        if n_terms > 1:
            T[:, 1] = x_scaled
        for k in range(2, n_terms):
            T[:, k] = 2.0 * x_scaled * T[:, k-1] - T[:, k-2]
            
    if time.time() - start_time > TIME_LIMIT:
        # Fallback: just return the constant term
        return np.full_like(x, p[0])

    # Prediction is linear combination
    y = T @ p
    
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
