import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Hybrid basis with polynomial terms and fixed RBFs.
    The 10 parameters control a mix of global polynomial trends (degree 2) and local bumps.
    To avoid local minima in BFGS when fitting the non-convex RBF part, we use an internal
    iterative least-squares refinement. Since the basis is fixed, we can solve for the optimal
    linear coefficients via normal equations. We use this to get a very accurate fit, which
    BFGS (starting from 1s) should be able to reach if the landscape is smooth enough, 
    or we can guide the parameters to be close to the solution if we were allowed to 
    modify the interface (which we are not). 
    
    However, the contract says BFGS fits the params. If we just return the linear prediction,
    BFGS will find the exact solution for a linear model in a few steps (quadratic loss).
    The risk is that the RBFs make the loss non-convex, so BFGS might get stuck.
    
    Alternative: Use ONLY a polynomial basis (linear model) to guarantee global convergence of BFGS.
    The current program uses degree 3 polynomial. The inspirations suggest RBFs work better (score 22 vs 1e8).
    Wait, the current program (polynomial degree 3) has score 3.5e8, which is terrible.
    The inspirations have scores 322, 49, 22. This suggests the polynomial basis is failing badly, 
    possibly due to ill-conditioning or the target function being far from a degree 3 polynomial.
    
    Let's look at the inspirations: "Dense Multi-Scale Fixed RBF Basis".
    If I use a linear model (polynomial), BFGS should work perfectly. Why did it fail?
    Maybe the polynomial basis is ill-conditioned for BFGS?
    Normal equations for polynomials on [-1,1] can be ill-conditioned.
    
    Let's try a better-conditioned linear basis: Chebyshev polynomials or orthogonal polynomials?
    Or just a simple RBF basis which is often better conditioned if centered well.
    
    Given the inspirations used RBFs and got scores < 100, while polynomial got 1e8, 
    RBFs are likely the better approach. The issue with RBFs is non-convexity.
    
    To maximize the chance of BFGS succeeding:
    1. Use a fixed RBF basis with well-separated, multi-scale centers.
    2. Ensure the basis functions are not too overlapping to avoid ill-conditioning.
    3. The initial guess is all ones. If the true solution is close to 1s, BFGS works.
    
    Let's implement a robust RBF basis with 10 centers.
    Centers:
    - 1 large center at (0,0)
    - 4 medium centers at (±0.5, ±0.5)
    - 4 small centers at (±0.8, 0), (0, ±0.8)
    - 1 extra? Let's say 5 small ones? No, 1+4+4 = 9. Add one at (0.8, 0.8)?
    
    Actually, a grid of 3x3 = 9 centers + 1 global constant is often good.
    But params length is 10.
    
    Let's define the basis:
    f0 = 1 (global constant)
    f1..f9 = RBFs at grid points:
    Grid: 
    (-0.6, -0.6), (-0.6, 0), (-0.6, 0.6)
    ( 0.0, -0.6), ( 0.0, 0), ( 0.0, 0.6)
    ( 0.6, -0.6), ( 0.6, 0), ( 0.6, 0.6)
    
    Widths:
    The center (0,0) should be wider to cover the middle.
    Corners can be narrower.
    
    To help BFGS, we should ensure the basis matrix is well-conditioned.
    
    Let's write the equation.
    """
    start_time = time.time()
    
    # Parameters
    p = params[:10]
    
    # Define centers
    # Center 0: Global (handled by p[0] as constant)
    # Centers 1-9: Grid
    cx = np.array([
        -0.6, -0.6, -0.6,
         0.0,  0.0,  0.0,
         0.6,  0.6,  0.6
    ])
    cy = np.array([
        -0.6,  0.0,  0.6,
        -0.6,  0.0,  0.6,
        -0.6,  0.0,  0.6
    ])
    
    # Widths (sigma in exp(-dist^2 / (2*sigma^2)))
    # Use a mix of scales
    # Center (0,0) gets a wider width to fill the middle
    # Corners get narrower
    sigmas = np.array([
        0.4, 0.4, 0.4,
        0.4, 0.6, 0.4,
        0.4, 0.4, 0.4
    ])
    
    # Compute predictions
    # x and y are arrays of shape (N,)
    
    # Constant term
    pred = p[0] * np.ones_like(x)
    
    # RBF terms
    for i in range(9):
        dx = x - cx[i]
        dy = y - cy[i]
        dist_sq = dx**2 + dy**2
        sigma = sigmas[i]
        rbf = np.exp(-dist_sq / (2.0 * sigma * sigma))
        pred += p[i+1] * rbf
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to zeros if something explodes
        return np.zeros_like(x)
        
    return pred
