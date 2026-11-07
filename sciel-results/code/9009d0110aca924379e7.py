"""
Strategy: Internal RBF Fitting with Closed-Form Linear Solution.
Instead of relying on the external BFGS optimizer to fit all 10 parameters 
simultaneously in a non-convex landscape, we use the 10 parameters to define 
a small RBF network (3 Gaussians + 1 linear term, or similar). 
The 'equation' function performs its own internal optimization:
1. It treats the inputs as a 4D space.
2. It performs a coarse grid search/coordinate descent to find good centers and widths 
   for the RBFs.
3. For each candidate set of centers/widths, it solves for the amplitudes 
   using closed-form linear least squares (np.linalg.lstsq), which is fast and robust.
4. It returns the prediction from the best configuration found within the time limit.
This effectively ignores the external BFGS's initial guess of ones and finds the 
optimal structure itself.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize  # Not used for main logic, but available

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using an internal RBF fitting strategy.
    
    The external optimizer is bypassed for the heavy lifting. We use the 
    training data (b, s, temp, pH) and target y (which we don't have directly, 
    but the evaluator minimizes MSE against y) to fit the model.
    
    Wait, the evaluator calls equation(inputs, params) to get predictions. 
    It then computes MSE against the hidden y. 
    It does NOT give us y inside equation(). 
    So we CANNOT do internal fitting against y inside equation() because we don't have y.
    
    CORRECTION: 
    The task says: "The evaluator fits params... minimizing the training MSE."
    This means the evaluator runs:
    1. Generate predictions: pred = equation(X_train, params_current)
    2. Compute loss = MSE(pred, y_train)
    3. Update params using BFGS.
    
    So I DO NOT have access to y_train inside equation(). 
    I CANNOT do internal least squares against y.
    
    Therefore, the "Internal Fitting" strategy is invalid because I don't have the target.
    
    I must return a prediction based on X and params.
    
    Let's rethink.
    The current program scores 40.0 (bad). The seed scores 1.0.
    Inspiration 3 (score 0.98) used "Gaussian Radial Basis Functions with all centers, widths, and amplitudes exposed as the 10 optimizable".
    
    If I expose centers, widths, and amplitudes as params, the external BFGS has to optimize 10 non-linear parameters. This is hard.
    
    However, if I structure the function such that the params are "easy" for BFGS, or if I use a representation that is closer to linear?
    
    Alternative Strategy: **Fourier Basis / Polynomial Expansion**
    If I use a polynomial or Fourier basis, the problem becomes linear in the coefficients. 
    But I have only 10 params. 
    If I fix the basis functions (e.g., specific polynomials or sines) and let the 10 params be the coefficients, the optimization is linear (convex). BFGS will converge quickly to the global optimum (or close to it) because it's a quadratic error surface.
    
    Let's try a **Fixed-Basis Linear Model**.
    Basis:
    - Constant
    - Linear terms for b, s, temp, pH (4)
    - Interactions: b*s, b*temp, s*temp (3)
    - Quadratic: b^2, s^2, temp^2, pH^2 (4)
    Total 1+4+3+4 = 12 terms. I have 10 params.
    
    Let's select 10 meaningful basis functions.
    1. 1 (intercept)
    2. b_norm
    3. s_norm
    4. temp_norm
    5. pH_norm
    6. b_norm * s_norm
    7. temp_norm * pH_norm
    8. b_norm^2
    9. s_norm^2
    10. temp_norm^2
    
    Since the relationship is likely smooth and bounded, a polynomial basis is a strong candidate.
    Crucially, if the model is linear in params, the MSE is a convex quadratic function of params. BFGS will find the minimum very reliably, unlike the non-convex Gaussian model.
    
    To help BFGS start from ones, I should normalize the inputs so that the basis functions are O(1) in magnitude.
    
    Let's refine the basis to be orthogonal-ish or at least well-scaled.
    Inputs:
    b in [1, 5] -> b_n = (b - 3) / 2  (center 3, range [-1, 1])
    s in [1, 5] -> s_n = (s - 3) / 2
    temp in [1, 40] -> t_n = (temp - 20.5) / 19.5  (center 20.5, range ~[-1, 1])
    pH in [2, 12] -> ph_n = (pH - 7) / 5  (center 7, range [-1, 1])
    
    Basis:
    1. 1
    2. b_n
    3. s_n
    4. t_n
    5. ph_n
    6. b_n * s_n
    7. t_n * ph_n
    8. b_n * t_n
    9. s_n * t_n
    10. b_n * ph_n
    
    This captures main effects and key interactions.
    
    Why this is better:
    1. Convex optimization: The external BFGS will not get stuck in local minima.
    2. Scalability: Normalized inputs keep gradients stable.
    3. Flexibility: Interactions allow capturing non-linearities (e.g., temperature effect depends on pH).
    
    Let's implement this.
    """
    
    # Normalize inputs to roughly [-1, 1] to keep basis values small and gradients stable for BFGS
    # Ranges from task: b:[1,5], s:[1,5], temp:[1,40], pH:[2,12]
    
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    t_n = (temp - 20.5) / 19.5
    ph_n = (pH - 7.0) / 5.0
    
    p = params
    
    # Construct the 10 basis functions
    # We want the model to be: sum(p[i] * basis[i])
    # This makes the problem linear in p, ensuring convex MSE landscape.
    
    # 1. Intercept
    term0 = 1.0
    
    # 2-5. Linear main effects
    term1 = b_n
    term2 = s_n
    term3 = t_n
    term4 = ph_n
    
    # 6-10. Selected interactions
    # Interaction between substrate and density (nutrient limitation)
    term5 = b_n * s_n
    # Interaction between temperature and pH (physiological stress)
    term6 = t_n * ph_n
    # Interaction between density and temperature
    term7 = b_n * t_n
    # Interaction between substrate and temperature
    term8 = s_n * t_n
    # Interaction between density and pH
    term9 = b_n * ph_n
    
    # Combine
    # result = p[0]*term0 + ... + p[9]*term9
    
    result = (p[0] * term0 + 
              p[1] * term1 + 
              p[2] * term2 + 
              p[3] * term3 + 
              p[4] * term4 + 
              p[5] * term5 + 
              p[6] * term6 + 
              p[7] * term7 + 
              p[8] * term8 + 
              p[9] * term9)
              
    return result
# EVOLVE-BLOCK-END
