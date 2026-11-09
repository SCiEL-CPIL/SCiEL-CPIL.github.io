"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a mixed basis expansion.
    
    The basis combines Chebyshev polynomials (for stable polynomial approximation)
    with trigonometric functions (for periodic components) and a Gaussian bump
    (for localized features).
    
    Basis functions (10 total):
    0: T_0(t) = 1
    1: T_1(t) = t
    2: T_2(t) = 2t^2 - 1
    3: T_3(t) = 4t^3 - 3t
    4: T_4(t) = 8t^4 - 8t^2 + 1
    5: T_5(t) = 16t^5 - 20t^3 + 5t
    6: sin(x)
    7: cos(x)
    8: exp(-x^2)
    9: T_9(t) = 512t^9 - 1024t^7 + 672t^5 - 160t^3 + 15t (high degree Chebyshev)
    
    Wait, let's stick to a simpler, well-conditioned set.
    Chebyshev polynomials are orthogonal on [-1, 1], which makes them ideal for
    least squares fitting. The evaluator uses BFGS on the parameters. Since the
    model is linear in parameters, BFGS should converge quickly to the global
    minimum if the basis is well-conditioned.
    
    Let's use:
    0-5: Chebyshev T_0 to T_5 (covers polynomial behavior up to degree 5)
    6: sin(x)
    7: cos(x)
    8: exp(-x^2)
    9: tanh(x)
    
    This covers:
    - Polynomials (via Chebyshev)
    - Periodic functions (sin, cos)
    - Localized bumps (exp)
    - Sigmoid-like transitions (tanh)
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_5
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    T5 = 2 * t * T4 - T3
    
    # Other basis functions
    S1 = np.sin(x)
    C1 = np.cos(x)
    E1 = np.exp(-x * x)
    H1 = np.tanh(x)
    
    # Linear combination
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * T5 +
         params[6] * S1 +
         params[7] * C1 +
         params[8] * E1 +
         params[9] * H1)
    
    return y
# EVOLVE-BLOCK-END
