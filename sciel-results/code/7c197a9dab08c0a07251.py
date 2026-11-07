"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Basis Selection for Nonlinear Oscillator Regression.
    
    Instead of a fixed basis, this function:
    1. Generates a pool of ~40 physically plausible features (polynomials, trig, mixed).
    2. Performs a fast greedy forward selection (using QR decomposition/normal equations)
       to find the 10 features that best explain the variance in the target 'y' 
       (approximated by the current 'params' or a dummy target if we were fitting, 
       but here we assume the evaluator provides 'params' as the weights for the 
       selected basis. Wait, the evaluator fits 'params'. 
       
       Correction on Strategy:
       The evaluator calls equation(t, x, v, params) and minimizes MSE.
       'params' are the weights for the 10 terms in our returned expression.
       If we change the basis dynamically, the meaning of params[0..9] changes.
       The evaluator starts params at all ones.
       
       Crucial Constraint: The evaluator fits `params` (length 10) to minimize MSE.
       If we select a subset of features, we must map the 10 `params` to those features.
       However, the selection process depends on the data (t, x, v) and the target (y).
       We do NOT have access to the target `y` inside `equation`. 
       
       Re-evaluating the "Different Strategy":
       Since we don't have `y`, we cannot do supervised feature selection inside `equation`.
       
       Alternative Strategy: **Orthogonalized Basis via Gram-Schmidt on Input Data**
       We can use the inputs (t, x, v) to construct a basis that is less correlated.
       However, the best "different strategy" without `y` is to use a **Fourier Series** 
       or **RBF-like** basis that is naturally more orthogonal or captures periodicity better.
       
       Let's look at the inputs: t is [30, 50]. This is a specific time window.
       The oscillator is likely periodic or quasi-periodic.
       
       New Strategy: **Time-Dependent Basis with Frequency Tuning**
       The current basis uses cos(t) and sin(t). 
       A nonlinear oscillator with parametric drive might have sidebands.
       Let's use a basis that explicitly models the drive frequency and harmonics.
       
       Wait, the previous program scored 3678. The inspiration p24 scored 1.04 (huge improvement).
       The diff for p24 shows it changed the basis to "Physics-informed sparse basis".
       The prompt says p24 is a child of p15 (the current program).
       p24 score: 1.044. Current score: 3678.
       The diff for p24 was truncated but described as:
       "Basis terms: 0: x, 1: x^3..."
       
       My task is to propose a DIFFERENT strategy than the current one (which is a fixed polynomial/trig mix).
       The "Different Strategy" should be distinct from the fixed basis approach.
       
       Strategy: **Spline-like Basis using B-splines or Piecewise Polynomials**
       Nonlinear oscillators can be approximated well by piecewise functions.
       However, implementing splines with 10 params is tricky.
       
       Alternative: **Kernel Regression Basis (RBF)**
       Use 10 Radial Basis Functions centered at specific points in the (x, v) phase space.
       This allows for highly flexible nonlinear mapping.
       Centers can be fixed (e.g., a grid) or derived from input stats.
       
       Let's try **RBF Basis**.
       Features: exp(-gamma * ||[x, v] - c_i||^2)
       With 10 centers.
       This is a very different family from polynomials/trig.
       
       Implementation:
       1. Define 10 centers in the (x, v) domain. 
          x range: [-0.2, 0.17], v range: [-0.34, 0.36].
          We can place centers on a grid covering this range.
       2. Compute distances from each input point to each center.
       3. Return linear combination of RBFs weighted by params.
       
       Why this might work better:
       - RBFs are local. They can model complex nonlinear shapes (like Duffing potential) 
         that global polynomials struggle with unless high-order.
       - Polynomials of order 3+ might oscillate or be ill-conditioned.
       - RBFs are smooth and flexible.
       
       Let's refine the centers.
       x_min, x_max = -0.21, 0.18
       v_min, v_max = -0.34, 0.37
       
       We need 10 centers. A 3x3 grid plus 1 center? Or 5x2?
       Let's use a 3x3 grid (9 centers) + 1 center at origin.
       
       Gamma (width) is critical. If gamma is too large, it's like a constant. 
       If too small, it's spiky.
       We can fix gamma to a reasonable value based on the range.
       Range of x ~ 0.4, Range of v ~ 0.7.
       Distances are in the unit of [x, v].
       Let's normalize x and v before computing RBF to make gamma scale-invariant?
       No, we only have 10 params. We can't fit gamma.
       We must hardcode a good gamma.
       
       Let's try normalizing inputs internally:
       x_norm = (x - 0) / 0.2
       v_norm = (v - 0) / 0.35
       
       Centers in normalized space:
       Grid: -1, 0, 1 for x_norm and -1, 0, 1 for v_norm.
       
       Let's implement this.
    """
    p = params
    
    # Normalize inputs to a standard scale to make RBF widths effective
    # x range approx [-0.2, 0.2] -> scale 0.2
    # v range approx [-0.35, 0.35] -> scale 0.35
    x_norm = x / 0.2
    v_norm = v / 0.35
    
    # Define 10 centers in normalized (x, v) space
    # 9 grid points + 1 origin
    cx = np.array([-1.0, 0.0, 1.0])
    cv = np.array([-1.0, 0.0, 1.0])
    
    # Create meshgrid of centers
    cx_grid, cv_grid = np.meshgrid(cx, cv, indexing='ij')
    centers_x = cx_grid.flatten()
    centers_v = cv_grid.flatten()
    
    # Add origin as the 10th center
    centers_x = np.append(centers_x, 0.0)
    centers_v = np.append(centers_v, 0.0)
    
    # Centers shape: (10, 2)
    centers = np.column_stack((centers_x, centers_v))
    
    # Input points shape: (N, 2)
    points = np.column_stack((x_norm, v_norm))
    
    # Calculate squared distances between each point and each center
    # ||p - c||^2 = ||p||^2 + ||c||^2 - 2 p.c
    # points: (N, 2), centers: (10, 2)
    # dist_sq: (N, 10)
    
    p_sq = np.sum(points**2, axis=1)[:, np.newaxis]  # (N, 1)
    c_sq = np.sum(centers**2, axis=1)[np.newaxis, :]  # (1, 10)
    dot = points @ centers.T                          # (N, 10)
    
    dist_sq = p_sq + c_sq - 2 * dot
    
    # Ensure numerical stability (clamp negative values due to float error)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # RBF: exp(-gamma * dist_sq)
    # Choose gamma. 
    # If gamma is too small, features are similar (collinear).
    # If gamma is too large, features are orthogonal but might not capture global trends.
    # A typical value for normalized data in [-1, 1] is around 1.0 to 2.0.
    gamma = 1.0
    
    rbf_values = np.exp(-gamma * dist_sq)
    
    # Linear combination with params
    # rbf_values: (N, 10)
    # params: (10,)
    y = rbf_values @ p
    
    return y

# EVOLVE-BLOCK-END
