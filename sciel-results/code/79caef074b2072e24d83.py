"""
Strategy: Internal multi-start optimization with a rich basis.
The evaluator calls equation(x, y, params) once.
We interpret 'params' as a seed for internal randomization or initial values.
We construct a basis of Chebyshev polynomials (up to order 4) and RBFs.
We run a lightweight internal optimizer (SGD/Adam hybrid or simple L-BFGS with multiple restarts) 
to fit the weights of this basis to the training data (x, y).
We return the predictions from the best model found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache to store best weights if we want to be fancy, 
# but since equation is called once per evaluation, we just compute.
# However, the evaluator might call equation multiple times during BFGS?
# No, the contract says: "The evaluator fits params... minimizing training MSE."
# "equation must return an array with one prediction per point."
# "A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
# This implies the evaluator calls equation(x, y, current_params) to get predictions, 
# computes MSE, and passes gradients back to BFGS which updates params.
#
# CRITICAL INSIGHT: 
# If the evaluator uses BFGS on `params`, it expects `equation` to be differentiable 
# and smooth with respect to `params`. 
# If I do internal optimization, the output of `equation` becomes a non-smooth, 
# piecewise-constant (or step-like) function of `params` (since the internal optimizer 
# converges to local minima that jump around). This will break BFGS.
#
# Therefore, the strategy MUST be a smooth function of `params`.
# 
# Re-evaluating Strategy:
# The current program is `p0*x + p1*y + p2`. 
# The inspiration 1 used Chebyshev. 
# The inspiration 3 used "Non-parametric-to-parametric distillation".
# 
# If I cannot do internal multi-start (because it breaks differentiability for the outer BFGS),
# I must design a **better smooth basis** that is easier for BFGS to navigate.
#
# New Strategy: 
# Use a **tanh-saturated polynomial basis** or a **Gaussian Process-like kernel expansion** 
# where the parameters are the kernel weights, but mapped through a non-linearity 
# that keeps the gradient landscape smooth.
#
# Actually, the most robust structural change is to use a **Universal Approximator basis** 
# with **pre-computed fixed centers** (RBFs) but **learnable widths** or **learnable amplitudes** 
# that are well-conditioned.
#
# Let's look at the "tanh denominator" lesson: 
# "Reparameterizing denominator coefficients with tanh bounds them... making the loss landscape much smoother."
#
# Let's try a **Neural Network-like basis** (single hidden layer) with fixed, well-distributed 
# random weights (initialized once) and learnable output weights. 
# Wait, if the hidden weights are fixed, the basis functions are fixed. 
# The parameters `params` will be the output weights. 
# This is equivalent to a fixed-basis linear model. 
# The difference is the **choice of basis**.
#
# Current: Linear (x, y, 1).
# Inspiration 1: Chebyshev.
#
# Let's try a **Hybrid Basis**: 
# 1. Low-order polynomials (x, y, xy, x^2, y^2)
# 2. Radial Basis Functions (RBFs) centered at a grid, with fixed widths.
# 3. To ensure smoothness and avoid ill-conditioning, we will use a **regularized** form.
#
# However, the evaluator fits `params` (length 10). 
# If I use 10 RBFs, each `params[i]` is the weight for RBF i.
# The basis functions are fixed.
# The function is linear in `params`. 
# BFGS should handle this well IF the basis is well-conditioned.
#
# Why did the linear model fail? Likely because the target is non-linear.
# Why might Chebyshev fail? Maybe the target has sharp features not captured by low-order poly.
#
# Let's use **10 RBFs** with centers spread across the domain [-1, 1]x[-1, 1].
# To make it "structurally different" from the linear model and potentially better than Chebyshev:
# Use **Gaussian RBFs** with adaptive widths? No, widths must be fixed to keep params length 10.
#
# Let's use a **grid of RBFs** with **varying widths** to capture both global and local features.
#
# Basis Design:
# 4 corners: width 1.0 (wide)
# 4 edges: width 0.5 (medium)
# 1 center: width 0.5
# 1 extra center: width 1.0
# Total 10 parameters.
#
# This creates a smooth, flexible surface that is linear in the weights.
# BFGS will optimize the weights.
#
# To further improve the landscape, we can apply a **tanh transformation** to the parameters 
# if we suspect they might blow up, but for RBF weights, they can be any real number.
# However, the lesson mentioned tanh for denominators. 
#
# Let's stick to the RBF basis but ensure the centers are well-distributed.
#
# Alternative: **Fourier Basis**? 
# sin(kx), cos(kx), sin(ky), cos(ky)... 
# This is also a fixed basis.
#
# Let's go with **RBFs** as they are local and good for arbitrary functions.
#
# Implementation details:
# 1. Define 10 centers in [-1, 1] x [-1, 1].
# 2. Define 10 widths (fixed).
# 3. Compute phi(x, y) = exp(-width_j * ((x-cx_j)^2 + (y-cy_j)^2)) for j=0..9.
# 4. Prediction = sum(params[j] * phi_j).
#
# This is a linear model in params. BFGS will solve it.
# The key is that the basis is rich enough to approximate the hidden function.
#
# To make it even more robust, I can add a small constant term (bias) by including a "flat" RBF 
# with very large width, or just assume the RBFs cover it.
#
# Let's refine the centers and widths to ensure coverage.
# Centers:
# (-1, -1), (1, -1), (-1, 1), (1, 1) -> Corners
# (0, -1), (0, 1), (-1, 0), (1, 0) -> Edges
# (0, 0) -> Center
# (0.5, 0.5) -> Extra? Or maybe (-0.5, -0.5)?
#
# Let's use a 3x3 grid plus one more?
# Grid:
# -1, 0, 1
# -1, 0, 1
# -1, 0, 1
# That's 9 points. Add (0,0) again? Or (0.5, 0.5)?
# Let's add (0, 0) with a different width to allow for a "core" feature.
#
# Widths:
# Corners: 2.0 (wider to cover corners)
# Edges: 1.5
# Center: 1.0
# Extra: 1.0
#
# This creates a multi-scale representation.

def _get_basis(x, y):
    """
    Returns an array of shape (N, 10) containing the basis function evaluations.
    x, y: arrays of shape (N,)
    """
    N = len(x)
    
    # Centers
    centers = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  1.0],
        [-1.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0,  0.0],
        [ 0.5,  0.5]  # 10th center
    ])
    
    # Widths (inverse of variance in Gaussian exp(-w * dist^2))
    # Larger w = narrower peak
    widths = np.array([
        2.0, 2.0, 2.0, 2.0,  # Corners
        1.5, 1.5, 1.5, 1.5,  # Edges
        1.0,                  # Center
        1.5                   # Extra
    ])
    
    # Compute squared distances
    # x: (N,), centers: (10, 2)
    # dist_sq: (N, 10)
    dx = x[:, np.newaxis] - centers[0, :]  # (N, 1)
    dy = y[:, np.newaxis] - centers[1, :]  # (N, 1)
    # Wait, this is wrong. Need to broadcast against all centers.
    
    # Correct broadcasting:
    # x_col: (N, 1), cx: (10,) -> dx: (N, 10)
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    cx = centers[:, 0]  # (10,)
    cy = centers[:, 1]  # (10,)
    
    dx = x_col - cx  # (N, 10)
    dy = y_col - cy  # (N, 10)
    
    dist_sq = dx**2 + dy**2  # (N, 10)
    
    # Apply widths
    # widths: (10,) -> (1, 10) for broadcasting
    w_col = widths[np.newaxis, :]
    
    # Gaussian RBF: exp(-w * dist_sq)
    basis = np.exp(-w_col * dist_sq)
    
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x, y: arrays of shape (N,)
    params: array of shape (10,)
    Returns: array of shape (N,)
    """
    # Ensure inputs are 1D
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    params = np.asarray(params).flatten()
    
    # Safety check for length
    if len(params) < 10:
        # Pad with zeros if needed, though evaluator should provide 10
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
        
    # Compute basis
    B = _get_basis(x, y)  # (N, 10)
    
    # Linear combination
    # B: (N, 10), params: (10,) -> pred: (N,)
    pred = B @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something goes wrong
        # This is a safety net
        return params[0] * x + params[1] * y + params[2]
        
    return pred
# EVOLVE-BLOCK-END
