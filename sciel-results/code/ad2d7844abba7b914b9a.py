"""
Symbolic regression with dynamic feature selection.
Instead of a fixed basis, we maintain a large pool of candidate functions.
At each call to equation(x, params), we perform a greedy forward selection
of 10 features from the pool that best explain the current 'target' implied by params.
Wait, params are the weights. The target is hidden.
Actually, the evaluator minimizes MSE(y, equation(x, params)).
If equation(x, params) depends on params in a non-linear way (via selection),
BFGS will try to adjust params to minimize error.
However, standard BFGS assumes smoothness. Greedy selection is discrete.
To make this work with BFGS, we can use a 'soft' selection or simply
perform the greedy selection based on the current residual of a linear fit?
No, we don't have y inside equation.
We only have x and params.
The evaluator computes y_pred = equation(x, params).
Then it computes loss = mean((y_true - y_pred)^2).
It does not pass y_true to equation.
So equation cannot do feature selection based on y_true directly.
It can only use params.

Revised Strategy:
Use the 10 params to control a **mixture of experts** or a **sparse combination** 
via a differentiable approximation of sparsity/selection.
Or, simpler: Use a **Fourier Series** with higher frequencies but regularized 
by the params acting as amplitudes. The current score is ~4367.
The target is likely a smooth function.
Let's try a **Spline-like basis** using B-splines or a **Gaussian Process** 
kernel basis (Nyström approximation).
Actually, the most robust improvement over a fixed polynomial/trig basis 
is to use a **RBF (Radial Basis Function) network** with fixed centers 
but learnable weights (params). 
10 RBFs with centers spread across [-3, 3] can approximate any smooth function 
very well. This is a different design family: Local Basis Functions vs Global Basis Functions.

Let's use 10 Gaussian RBFs with fixed centers at evenly spaced points in [-3, 3] 
and fixed bandwidths. The params will be the weights for these RBFs.
This is still linear in params, so BFGS works perfectly.
Why is this different? The current program uses global polynomials and trig.
RBFs are local. They can capture sharp features or non-smooth parts better 
if the function has localized behavior, or simply provide a more flexible 
interpolation basis than degree-5 Chebyshev + 4 trig terms.
Actually, degree-5 Chebyshev is quite powerful for smooth functions on [-3,3].
But RBFs with 10 centers can effectively act like a higher-order polynomial 
or capture local variations that a global low-degree polynomial misses.

Let's refine: Use **10 Gaussian RBFs** with centers $c_i = -3 + 6*i/9$ for i=0..9.
Bandwidth $\sigma = 1.0$ (or optimized? No, params are only 10).
We have exactly 10 params. So 10 RBFs, 10 weights.
This is a linear model: $y = \sum_{i=1}^{10} w_i \exp(-\frac{(x-c_i)^2}{2\sigma^2})$.
This is structurally different from the current Chebyshev+Trig+Tanh+Exp basis.
It is a local approximation method.
"""

import numpy as np

# Pre-compute centers and bandwidths for the RBF basis
# We have 10 parameters, so we use 10 RBFs.
# Centers evenly spaced in [-3, 3]
_CENTERS = np.linspace(-3.0, 3.0, 10)
# Bandwidth: A smaller bandwidth allows more local flexibility.
# If sigma is too small, the basis vectors become nearly orthogonal (good for conditioning) 
# but might require larger weights. If too large, they overlap heavily (ill-conditioned).
# Let's pick sigma = 0.8 for a good balance on [-3, 3].
_SIGMA = 0.8

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Radial Basis Function (RBF) network.
    The model is a linear combination of 10 Gaussian RBFs with fixed centers
    and fixed bandwidths. The 10 parameters are the weights of the RBFs.
    
    This approach uses local basis functions, which can capture non-global
    features (like bumps, steps, or localized curvature) more effectively 
    than a fixed low-order polynomial/trig basis, while remaining linear 
    in parameters for stable BFGS convergence.
    """
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is at least 1-D for broadcasting, but preserve shape for return
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Compute the RBF basis matrix: shape (N_points, 10)
    # x_flat: (N, 1) or (N,)
    # _CENTERS: (10,)
    # We want (N, 10)
    diff = x_flat[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Gaussian RBF: exp(-diff^2 / (2 * sigma^2))
    basis = np.exp(-(diff ** 2) / (2.0 * _SIGMA ** 2))
    
    # Linear combination: (N, 10) @ (10,) -> (N,)
    result = basis @ p
    
    return result.reshape(orig_shape)
