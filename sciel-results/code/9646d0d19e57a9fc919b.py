"""
Strategy: Adaptive Local Basis Construction (Global KNN-like).

The previous strategy relied on fixed global basis functions (polynomials, trigonometric, fixed-grid RBFs).
These are limited in their ability to capture sharp features or complex non-linearities if the global 
form doesn't match the target.

This new strategy constructs a basis matrix where each column corresponds to a "local" feature derived 
from the distance between the current point and a subset of reference points (training points). 
Specifically, we use a set of 9 reference points (a 3x3 grid or selected quantiles) and define features 
as Gaussian kernels centered at these points, but with a width (sigma) that is learned or fixed to be 
small enough to allow local variation. 

However, since we only have 10 parameters, we cannot use 200+ kernels. 
Instead, we use a "Global-Local" hybrid:
We define 9 basis functions as Gaussian kernels centered at the 9 points of a 3x3 grid 
[-1, 0, 1] x [-1, 0, 1]. 
Wait, the current program already does this in `_basis_rbf`.

Let's rethink. The current RBF uses sigma=1.0. 
A materially different approach is to use **Spline-like or Piecewise Linear** bases, or to use 
**Higher-order interactions** that are not present in the current hybrid.

Actually, a very powerful strategy for symbolic regression with limited parameters is 
**Feature Selection via Lasso-like sparsity** is not possible because we must return a prediction 
using the `params` array directly, and the evaluator fits `params`. We cannot change the fitting process 
to be sparse. The evaluator uses BFGS on the MSE.

So we must provide a basis $A(x,y)$ such that $A(x,y) @ params$ approximates $y$.
The key is the choice of $A$.

Current Bases:
1. Poly3: 1, x, y, x2, y2, xy, x3, y3, x2y, xy2
2. Trig: 1, cos(x), sin(x), cos(y), sin(y), ...
3. RBF: 9 Gaussians + bias
4. Hybrid: Poly2 + Trig1

What is missing?
- Higher order trigonometric terms (harmonics).
- Mixed polynomial-trigonometric terms (e.g., x*cos(x)).
- Sigmoidal or step-like functions (tanh, sigmoid) which are good for capturing discontinuities or saturation.
- Radial basis with different centers (e.g., corners, edges, center).

Let's try a strategy that uses **tanh-based basis functions**. 
Tanh functions are excellent for approximating step functions and saturating behaviors.
Basis: 1, tanh(x), tanh(y), tanh(x+y), tanh(x-y), tanh(x*y), tanh(x^2+y^2), tanh(x^2-y^2), tanh(2x), tanh(2y).

Another strong candidate: **Chebyshev Polynomials**. They are orthogonal and well-conditioned.
Chebyshev T0, T1, T2, T3 for x and y, and products.

Let's implement a strategy that searches over a diverse set of bases, including:
1. Chebyshev (Polynomial)
2. Sine/Cosine (Fourier)
3. Tanh (Neural Network-like activation)
4. RBF (Gaussian)
5. B-spline like (Piecewise linear/cubic) - approximated by smooth bumps.

We will evaluate each basis using `np.linalg.lstsq` to find the optimal parameters and the resulting training error. We select the basis with the lowest training error. Since the basis is linear in parameters, the BFGS optimizer in the evaluator will converge to the same global minimum (or very close) as the least-squares solution, provided the basis is well-conditioned.

To improve upon the current score, we need bases that are more expressive or better conditioned for the specific hidden function. The current RBF with sigma=1.0 might be too smooth or too wide. We can try multiple sigmas for RBF. We can also try tanh with different scales.

New Strategy:
1. Define a list of candidate basis functions.
2. For each basis, compute the training error using least squares.
3. Select the best basis.
4. Return the prediction using the best basis and the provided `params`.

Bases to include:
- Chebyshev Polynomials (degree 3)
- Fourier Series (up to frequency 2)
- Tanh Network (various arguments)
- RBF (multiple sigmas: 0.5, 1.0, 2.0)
- Polynomial (degree 4, sparse)

This is a different design family in the sense that it expands the search space of structural forms significantly beyond the initial 4, specifically targeting different inductive biases (orthogonality, saturation, locality with varying scale).
"""

import numpy as np
import time

def _basis_chebyshev(x, y):
    """
    Chebyshev polynomials up to degree 3.
    T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x
    Basis: 1, T1(x), T1(y), T2(x), T2(y), T1(x)T1(y), T2(x)T1(y), T1(x)T2(y), T3(x), T3(y)
    """
    T1x = x
    T1y = y
    T2x = 2 * x**2 - 1
    T2y = 2 * y**2 - 1
    T3x = 4 * x**3 - 3 * x
    T3y = 4 * y**3 - 3 * y
    
    return np.column_stack([
        np.ones_like(x),
        T1x,
        T1y,
        T2x,
        T2y,
        T1x * T1y,
        T2x * T1y,
        T1x * T2y,
        T3x,
        T3y
    ])

def _basis_fourier(x, y):
    """
    Fourier basis up to frequency 2.
    1, cos(x), sin(x), cos(y), sin(y), cos(2x), sin(2x), cos(2y), sin(2y), cos(x)sin(y)
    """
    return np.column_stack([
        np.ones_like(x),
        np.cos(x),
        np.sin(x),
        np.cos(y),
        np.sin(y),
        np.cos(2 * x),
        np.sin(2 * x),
        np.cos(2 * y),
        np.sin(2 * y),
        np.cos(x) * np.sin(y)
    ])

def _basis_tanh(x, y):
    """
    Tanh basis with various arguments.
    1, tanh(x), tanh(y), tanh(x+y), tanh(x-y), tanh(x*y), tanh(x^2+y^2), tanh(2x), tanh(2y), tanh(x^2-y^2)
    """
    return np.column_stack([
        np.ones_like(x),
        np.tanh(x),
        np.tanh(y),
        np.tanh(x + y),
        np.tanh(x - y),
        np.tanh(x * y),
        np.tanh(x**2 + y**2),
        np.tanh(2 * x),
        np.tanh(2 * y),
        np.tanh(x**2 - y**2)
    ])

def _basis_rbf_sigma05(x, y):
    """
    RBF with sigma=0.5 (sharper peaks).
    """
    centers = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ])
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dist_sq = dx**2 + dy**2
    kernels = np.exp(-0.5 * dist_sq) # sigma^2 = 0.5? No, exp(-dist^2 / (2*sigma^2)). 
    # Let's stick to exp(-gamma * dist^2). gamma = 1/(2*sigma^2).
    # If sigma=0.5, gamma = 1/(2*0.25) = 2.0.
    kernels = np.exp(-2.0 * dist_sq)
    bias = np.ones((len(x), 1))
    return np.hstack([kernels, bias])

def _basis_rbf_sigma2(x, y):
    """
    RBF with sigma=2.0 (wider peaks).
    """
    centers = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ])
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dist_sq = dx**2 + dy**2
    # sigma=2.0 -> gamma = 1/(2*4) = 0.125
    kernels = np.exp(-0.125 * dist_sq)
    bias = np.ones((len(x), 1))
    return np.hstack([kernels, bias])

def _basis_poly4_sparse(x, y):
    """
    Sparse polynomial degree 4.
    1, x, y, x^2, y^2, xy, x^3, y^3, x^4, y^4
    """
    return np.column_stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        y**2,
        x * y,
        x**3,
        y**3,
        x**4,
        y**4
    ])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0
    
    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    
    bases = [
        _basis_chebyshev,
        _basis_fourier,
        _basis_tanh,
        _basis_rbf_sigma05,
        _basis_rbf_sigma2,
        _basis_poly4_sparse
    ]
    
    best_basis_func = _basis_chebyshev
    best_error = np.inf
    
    for basis_func in bases:
        if time.time() > deadline:
            break
            
        try:
            A = basis_func(x_flat, y_flat)
            # Check for numerical stability
            if not np.all(np.isfinite(A)):
                continue
                
            p_hat, residuals, rank, s = np.linalg.lstsq(A, y_flat, rcond=None)
            pred = A @ p_hat
            error = np.mean((pred - y_flat)**2)
            
            if error < best_error:
                best_error = error
                best_basis_func = basis_func
        except Exception:
            continue
    
    A_test = best_basis_func(x_flat, y_flat)
    pred = A_test @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        # Fallback to simple polynomial if something went wrong
        A_fallback = _basis_chebyshev(x_flat, y_flat)
        pred = A_fallback @ params
        
    return pred
