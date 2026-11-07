"""
Rewritten program: Gaussian Radial Basis Function (RBF) basis for symbolic regression.

Strategy:
Use a basis of Gaussian Radial Basis Functions (RBFs) in 2D.
RBFs are local functions that decay exponentially with distance from a center.
This allows the model to capture non-linear, non-polynomial features that
global polynomials (like Chebyshev) struggle with, especially if the target
function has sharp features or complex interactions.

We define 10 fixed centers and 10 fixed scales.
The centers are chosen to cover the domain [-1, 1] x [-1, 1] reasonably well.
The scales are chosen to provide multi-resolution coverage.

Basis functions:
phi_i(x, y) = exp(-||[x, y] - c_i||^2 / (2 * sigma_i^2))

The 10 parameters are the weights for these 10 RBFs.
Since the centers and scales are fixed, the model is linear in the parameters,
which is ideal for BFGS (though for linear models, least squares would be better,
BFGS should still work if the Hessian is well-behaved).

To improve conditioning and allow the model to fit a constant offset better,
we include a constant basis function (sigma -> infinity, or just 1) as one of the
basis functions, or rely on the sum of RBFs to approximate constants if centers
are spread out. However, a pure RBF sum might struggle with a large constant
offset if the centers are not dense enough.
Let's use 9 RBFs and 1 constant term (1.0) to ensure we can model the mean.

Centers (c_i):
Let's place them on a grid or random-ish fixed points.
To be deterministic and cover the space:
(0,0)
(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)
(0, 0.8), (0, -0.8), (0.8, 0), (-0.8, 0)
Scales (sigma_i):
Mix of small and large scales.
0.2, 0.5, 0.2, 0.5, 0.2, 0.8, 0.8, 0.8, 0.8
And the constant term is effectively sigma = inf, weight w_10.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a Gaussian RBF basis + constant.
    
    The 10 parameters are weights for:
    p0: 1 (constant)
    p1: RBF at (0, 0), sigma=0.3
    p2: RBF at (0.5, 0.5), sigma=0.4
    p3: RBF at (-0.5, 0.5), sigma=0.4
    p4: RBF at (0.5, -0.5), sigma=0.4
    p5: RBF at (-0.5, -0.5), sigma=0.4
    p6: RBF at (0, 0.7), sigma=0.5
    p7: RBF at (0, -0.7), sigma=0.5
    p8: RBF at (0.7, 0), sigma=0.5
    p9: RBF at (-0.7, 0), sigma=0.5
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Precompute squared distances to centers
    # Center 0: (0,0)
    d2_0 = x**2 + y**2
    # Center 1: (0.5, 0.5)
    d2_1 = (x - 0.5)**2 + (y - 0.5)**2
    # Center 2: (-0.5, 0.5)
    d2_2 = (x + 0.5)**2 + (y - 0.5)**2
    # Center 3: (0.5, -0.5)
    d2_3 = (x - 0.5)**2 + (y + 0.5)**2
    # Center 4: (-0.5, -0.5)
    d2_4 = (x + 0.5)**2 + (y + 0.5)**2
    # Center 5: (0, 0.7)
    d2_5 = x**2 + (y - 0.7)**2
    # Center 6: (0, -0.7)
    d2_6 = x**2 + (y + 0.7)**2
    # Center 7: (0.7, 0)
    d2_7 = (x - 0.7)**2 + y**2
    # Center 8: (-0.7, 0)
    d2_8 = (x + 0.7)**2 + y**2
    
    # Scales squared (2 * sigma^2)
    # Using slightly larger sigmas to ensure overlap and smoothness for BFGS
    # sigma_0 = 0.3 -> 2*0.09 = 0.18
    # sigma_1..4 = 0.4 -> 2*0.16 = 0.32
    # sigma_5..8 = 0.5 -> 2*0.25 = 0.50
    
    inv_2s2_0 = 1.0 / 0.18
    inv_2s2_1 = 1.0 / 0.32
    inv_2s2_2 = 1.0 / 0.32
    inv_2s2_3 = 1.0 / 0.32
    inv_2s2_4 = 1.0 / 0.32
    inv_2s2_5 = 1.0 / 0.50
    inv_2s2_6 = 1.0 / 0.50
    inv_2s2_7 = 1.0 / 0.50
    inv_2s2_8 = 1.0 / 0.50
    
    # Compute RBF values
    rbf_0 = np.exp(-d2_0 * inv_2s2_0)
    rbf_1 = np.exp(-d2_1 * inv_2s2_1)
    rbf_2 = np.exp(-d2_2 * inv_2s2_2)
    rbf_3 = np.exp(-d2_3 * inv_2s2_3)
    rbf_4 = np.exp(-d2_4 * inv_2s2_4)
    rbf_5 = np.exp(-d2_5 * inv_2s2_5)
    rbf_6 = np.exp(-d2_6 * inv_2s2_6)
    rbf_7 = np.exp(-d2_7 * inv_2s2_7)
    rbf_8 = np.exp(-d2_8 * inv_2s2_8)
    
    # Constant term
    const = np.ones_like(x)
    
    # Combine with parameters
    # params[0] is weight for constant
    # params[1] is weight for rbf_0
    # ...
    f = (params[0] * const +
         params[1] * rbf_0 +
         params[2] * rbf_1 +
         params[3] * rbf_2 +
         params[4] * rbf_3 +
         params[5] * rbf_4 +
         params[6] * rbf_5 +
         params[7] * rbf_6 +
         params[8] * rbf_7 +
         params[9] * rbf_8)
         
    return f
# EVOLVE-BLOCK-END
