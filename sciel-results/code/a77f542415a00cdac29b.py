"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: global quadratic polynomial + single centered RBF.
    
    params[0:10] -> quadratic polynomial coefficients (centered at 1.25)
    params[10]   -> RBF amplitude (if we had 11 params, but we only have 10)
    
    Wait, params has length 10. So we need to fit exactly 10 parameters.
    
    Let's use a full quadratic polynomial in centered variables.
    Center: 1.25 (midpoint of [0.5, 2])
    
    Terms:
    0: 1
    1: cx
    2: cy
    3: cz
    4: cx^2
    5: cy^2
    6: cz^2
    7: cx*cy
    8: cx*cz
    9: cy*cz
    
    This is a 10-term full quadratic. It's linear in params, so BFGS will
    find the global optimum quickly (it's actually a least squares problem).
    
    But wait - the observation shows p40 with this exact structure scored 7.393,
    while the seed (linear) scored 1.0. That's WORSE. Why?
    
    The issue: BFGS starts from all ones. For a linear-in-params model, the
    optimal solution is found by linear regression. But BFGS might converge
    to a local minimum or have numerical issues.
    
    Actually, for a linear model, BFGS should find the global minimum. The
    score of 7.393 suggests something went wrong - perhaps the quadratic terms
    are overfitting or the conditioning is poor.
    
    Let me think differently. The seed (linear) scores 1.0. The quadratic
    scores 7.393. This means the quadratic is worse. Perhaps the target function
    is close to linear, and the quadratic adds noise/overfitting.
    
    Alternative: use a different basis. Maybe include some non-polynomial terms
    that are more flexible but still well-conditioned.
    
    Let's try: centered linear terms + a few selected quadratic terms + one
    interaction. Or better: use a radial basis function centered at the mean.
    
    Actually, let's reconsider. The observation says "Combines local RBF fitting
    with global linear and isotropic quadratic trends" scored 257.8. That's even
    worse.
    
    The key insight: the linear model scores 1.0, which is the baseline. We need
    to do better than 1.0. The quadratic made it worse (7.393).
    
    Maybe the target is not a polynomial at all. Let's try a model with
    different functional forms:
    - sin, cos, exp, log terms
    - But these are nonlinear in params, so BFGS is needed.
    
    Let's try: a[0]*x + a[1]*y + a[2]*z + a[3] + 
               a[4]*sin(x) + a[5]*sin(y) + a[6]*sin(z) + 
               a[7]*cos(x) + a[8]*cos(y) + a[9]*cos(z)
    
    This gives BFGS a nonlinear optimization problem, but with 10 params.
    Starting from all ones, BFGS should be able to find a good solution.
    
    The sine/cosine terms can capture periodic or oscillatory behavior that
    polynomials can't.
    
    Let's go with this.
    """
    p = params
    
    # Linear terms + constant
    base = p[0] * x + p[1] * y + p[2] * z + p[3]
    
    # Trigonometric terms
    trig = (p[4] * np.sin(x) + p[5] * np.sin(y) + p[6] * np.sin(z) +
            p[7] * np.cos(x) + p[8] * np.cos(y) + p[9] * np.cos(z))
    
    return base + trig
# EVOLVE-BLOCK-END
