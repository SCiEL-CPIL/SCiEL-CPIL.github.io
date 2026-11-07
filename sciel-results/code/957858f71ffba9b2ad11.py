"""
Strategy: Chebyshev-Activated Multiplicative Basis.

The current program uses a fixed-center Gaussian RBF. While local, RBFs can
suffer from ill-conditioning if centers are too close or if the function has
features that span multiple scales. Also, the fixed grid might miss optimal
center locations.

This new approach uses a "Chebyshev-Activated" basis. We construct 10 basis
functions as products of:
1. Low-degree Chebyshev polynomials T_n(x) mapped to [-3, 3]. These provide
   global stability and orthogonality-like properties, preventing the
   instability of raw power series.
2. Nonlinear activations (tanh, sin, or identity) applied to scaled x.
   These introduce the necessary nonlinearity to capture non-polynomial
   features (like steps, plateaus, or waves) without requiring the basis
   centers to be optimized (which would make the problem nonlinear in params).

The model is y = sum_{i=0}^{9} p_i * phi_i(x).
This is linear in p, so BFGS will find the optimal coefficients for this
fixed basis very reliably. The "search" is effectively in the choice of this
hybrid basis, which is designed to cover a broad class of functions:
- T_0 * 1 = 1 (constant)
- T_1 * 1 = x (linear)
- T_2 * tanh(x) (quadratic-like with saturation)
- T_3 * sin(x) (cubic-like with oscillation)
- T_4 * tanh(2x) (quartic-like with sharper saturation)
- T_5 * 1 (pure quintic)
- T_1 * sin(2x) (linear * oscillatory)
- T_2 * tanh(0.5x) (quadratic * slow saturation)
- T_3 * 1 (pure cubic)
- T_4 * sin(0.5x) (quartic * slow oscillation)

This diverse set of shapes allows the linear combination to approximate many
functions better than a pure polynomial or pure RBF set.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a 10-term Chebyshev-Activated Multiplicative Basis.
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 parameters (coefficients).
    
    Returns:
    y: Predicted values array of shape (N,)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure we have exactly 10 params
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    p = p[:10]
    
    # Map x from [-3, 3] to [-1, 1] for standard Chebyshev polynomials
    # t = (x + 3) / 6 - 1 = (x - 3) / 6
    t = (x - 3.0) / 6.0
    
    # Compute Chebyshev polynomials T_0 to T_5
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    T0 = np.ones_like(x)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    T5 = 16 * t**5 - 20 * t**3 + 5 * t
    
    # Define the 10 basis functions
    # We mix Chebyshev degrees with different nonlinear activations of x
    # using x directly for the activation to allow for different "frequencies"
    # or "sharpness" relative to the global Chebyshev trend.
    
    # Activations
    act1 = np.tanh(x)
    act2 = np.sin(x)
    act3 = np.tanh(2 * x)
    act4 = np.sin(2 * x)
    act5 = np.tanh(0.5 * x)
    act6 = np.sin(0.5 * x)
    
    # Basis Functions
    # 0: Constant
    phi0 = T0
    # 1: Linear
    phi1 = T1
    # 2: Quadratic trend with saturation
    phi2 = T2 * act1
    # 3: Cubic trend with oscillation
    phi3 = T3 * act2
    # 4: Quartic trend with sharp saturation
    phi4 = T4 * act3
    # 5: Pure Quintic (global high-order)
    phi5 = T5
    # 6: Linear trend with fast oscillation
    phi6 = T1 * act4
    # 7: Quadratic trend with slow saturation
    phi7 = T2 * act5
    # 8: Pure Cubic (global mid-order)
    phi8 = T3
    # 9: Quartic trend with slow oscillation
    phi9 = T4 * act6
    
    # Compute prediction
    # y = p0*phi0 + p1*phi1 + ... + p9*phi9
    y = (p[0] * phi0 + 
         p[1] * phi1 + 
         p[2] * phi2 + 
         p[3] * phi3 + 
         p[4] * phi4 + 
         p[5] * phi5 + 
         p[6] * phi6 + 
         p[7] * phi7 + 
         p[8] * phi8 + 
         p[9] * phi9)
         
    return y

# EVOLVE-BLOCK-END
