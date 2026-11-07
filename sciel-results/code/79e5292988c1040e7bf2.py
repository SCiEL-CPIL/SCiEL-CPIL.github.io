"""Symbolic regression with 1 input(s) - Combined Chebyshev and RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression combining Chebyshev polynomials and Gaussian RBFs.
    
    The model uses 10 parameters split between:
    - 4 Chebyshev polynomial coefficients (T_0 to T_3) for global smooth trends
    - 4 RBF centers (c_0 to c_3) for local feature capture
    - 2 RBF widths (sigma_0, sigma_1) shared or paired for stability
    
    Actually, to use all 10 params effectively:
    - 4 params for Chebyshev T_0..T_3
    - 4 params for RBF centers c_0..c_3
    - 2 params for RBF widths sigma_0, sigma_1 (applied to pairs of centers)
    
    Wait, the directive says split into 5 centers and 5 widths. But inspiration 3
    used fixed centers/widths with linear params. The current program uses 
    learnable frequencies/scales. Let's follow the directive's suggestion of 
    learnable centers and widths, but we only have 10 params total.
    
    Revised plan:
    - 5 RBF terms: exp(-(x - c_i)^2 / sigma_i^2)
    - But we need to define c_i and sigma_i from params.
    - If we use 5 centers and 5 widths, that's 10 params.
    - However, widths must be positive. We can use exp(params) or abs(params)+eps.
    - Centers can be any real number.
    
    Let's try:
    p0..p4: centers c_0..c_4
    p5..p9: widths sigma_0..sigma_4 (using softplus or abs to ensure positivity)
    
    But wait, if the function is smooth and global, RBFs with learnable centers
    might struggle with BFGS from all-ones. All-ones start means centers at 1.0 
    and widths ~1.0. This is a very specific initialization.
    
    Alternative: Hybrid of fixed-basis linear model (like inspiration 3 which 
    scored 226) and the current structure. The current score is ~12M, which is 
    terrible. Inspiration 3's score of 226 is much better. The knowledge note 
    says Chebyshev improved island 5 from 23 to 191.
    
    Let's combine:
    1. Use a stable global basis: Chebyshev polynomials T_0 to T_3 (4 params).
    2. Use local basis: 3 Gaussian RBFs with fixed centers and widths, but 
       learnable amplitudes (3 params).
    3. Use 3 additional params for non-linear transformations like tanh or 
       higher-order Chebyshev?
    
    Actually, the directive specifically suggests: "Replace the fixed or learnable 
    non-linear basis with a Radial Basis Function (RBF) network where the 10 
    parameters are split into 5 centers (c_i) and 5 widths (sigma_i)".
    
    Let's try that, but add safeguards.
    If BFGS starts at all-ones, centers are all 1.0. Widths are all 1.0.
    This might converge to a local minimum where all RBFs collapse to x=1.
    
    To mitigate this, we can use a hybrid:
    - 5 Chebyshev coefficients (T_0..T_4) -> 5 params
    - 5 Gaussian RBFs with fixed centers/widths, learnable amplitudes -> 5 params
    
    This is linear in params, which BFGS handles well. It combines the global 
    stability of Chebyshev (from knowledge) and local flexibility of RBFs 
    (from inspiration 3).
    
    Centers: linspace(-2, 2, 5)
    Widths: 0.5 (fixed)
    
    Let's implement this.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.full_like(x, np.nan)
    
    # Part 1: Chebyshev Polynomials T_0 to T_4
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2 * x2 - 1
    T3 = 4 * x3 - 3 * x
    T4 = 8 * x4 - 8 * x2 + 1
    
    cheb_part = p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 + p[4] * T4
    
    # Part 2: Gaussian RBFs with fixed centers and widths
    # Centers: -2, -1, 0, 1, 2
    # Width: 0.5
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    sigma = 0.5
    
    # Compute RBF values: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-0.5 * (diff / sigma) ** 2)
    
    rbf_part = rbf @ p[5:10]
    
    y = cheb_part + rbf_part
    
    # Ensure output is finite
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
