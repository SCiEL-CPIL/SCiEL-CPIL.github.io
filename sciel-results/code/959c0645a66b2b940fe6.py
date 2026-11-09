"""
Strategy: Hybrid Chebyshev-Tensor Basis.
Instead of monomial polynomials (which suffer from ill-conditioning at higher degrees) 
or local RBFs (which require many centers to cover the domain), this strategy uses 
Chebyshev polynomials of the first kind for the univariate components. Chebyshev 
polynomials are orthogonal on [-1, 1], providing excellent numerical stability for 
least-squares fitting (which BFGS approximates here). 

The basis consists of:
1. Univariate Chebyshev terms for x: T0(x), T1(x), T2(x), T3(x)
2. Univariate Chebyshev terms for y: T0(y), T1(y), T2(y), T3(y)
   (Note: T0 is constant 1, so we share the bias term to save parameters)
3. Low-frequency interaction terms: sin(πxy), cos(πxy)
   These capture potential radial or multiplicative smooth interactions that 
   pure tensor-product polynomials might approximate poorly with only 10 params.

Total 10 parameters:
p[0]: Bias (T0(x)T0(y) equivalent, but we just use 1)
p[1]: T1(x) = x
p[2]: T2(x) = 2x^2 - 1
p[3]: T3(x) = 4x^3 - 3x
p[4]: T1(y) = y
p[5]: T2(y) = 2y^2 - 1
p[6]: T3(y) = 4y^3 - 3y
p[7]: sin(πxy)
p[8]: cos(πxy)
p[9]: x*y (linear interaction, often strong)

This basis is global, stable, and captures both smooth trends and key interactions.
"""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure we have at least 10 parameters
    if len(params) < 10:
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
    
    # Compute Chebyshev polynomials
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    
    T1_x = x
    T2_x = 2.0 * x2 - 1.0
    T3_x = 4.0 * x3 - 3.0 * x
    
    T1_y = y
    T2_y = 2.0 * y2 - 1.0
    T3_y = 4.0 * y3 - 3.0 * y
    
    # Interaction terms
    xy = x * y
    pi_xy = np.pi * xy
    sin_pi_xy = np.sin(pi_xy)
    cos_pi_xy = np.cos(pi_xy)
    
    # Assign parameters
    # p[0]: Bias
    # p[1]: T1(x)
    # p[2]: T2(x)
    # p[3]: T3(x)
    # p[4]: T1(y)
    # p[5]: T2(y)
    # p[6]: T3(y)
    # p[7]: sin(pi * x * y)
    # p[8]: cos(pi * x * y)
    # p[9]: x * y
    
    pred = (p[0] 
            + p[1] * T1_x 
            + p[2] * T2_x 
            + p[3] * T3_x 
            + p[4] * T1_y 
            + p[5] * T2_y 
            + p[6] * T3_y 
            + p[7] * sin_pi_xy 
            + p[8] * cos_pi_xy 
            + p[9] * xy)
    
    # Safety check for time
    if time.time() - start_time > 10:
         # Fallback to simple linear if something goes wrong (unlikely)
         return p[0] + p[1] * x + p[4] * y + p[9] * xy

    return pred
