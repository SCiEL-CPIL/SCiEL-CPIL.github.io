"""
Strategy: Linearly Parameterized Feature Expansion.

The current program uses a multiplicative biological model (Monod * Gaussian * Quadratic) 
which creates a highly non-convex loss landscape for the 10 parameters. The external BFGS 
optimizer, starting from all ones, likely converges to a poor local minimum (score ~40).

This new approach restructures the model into a linear combination of basis functions:
y = sum(w_i * phi_i(x))

We have 10 parameters. We can use them to define the basis functions in a way that 
keeps the optimization landscape relatively smooth and allows BFGS to find a good fit.
However, a pure linear model with fixed basis functions isn't expressive enough if the 
basis is too simple. 

Better Strategy: 
Use the 10 parameters to define a small set of adaptive basis functions (like RBFs or 
polynomials with adjustable centers/widths) BUT ensure the final output is a linear 
combination of these basis functions with weights that are easy to optimize.
Actually, the constraint is "up to 10 fitted constants". The evaluator fits `params` (length 10).
If we make the model linear in `params`, BFGS will converge very reliably to the global 
minimum for those linear parameters.

Let's define 10 basis functions phi_1(x), ..., phi_10(x) that are fixed (or depend on 
simple transformations of inputs) and let params be the weights.
But we need enough flexibility. Fixed basis functions might not capture the complex 
interaction of temp/pH/substrate/density well enough with only 10 terms.

Alternative: 
Use a hybrid. 
Params 0-4: Weights for 5 distinct, well-separated basis functions.
Params 5-9: Centers/widths for some of these? No, that makes it non-linear in params again.

Let's look at the inputs: b, s, temp, pH.
Ranges:
b: [1, 5]
s: [1, 5]
temp: [1, 40]
pH: [2, 12]

Target: [-0.002, 1.55]

A robust approach that BFGS handles well is a **Polynomial Regression** with interaction terms, 
but normalized inputs.
If we use a polynomial basis of degree 2 or 3, we can have more than 10 terms. 
But we only have 10 params.

Let's select 10 specific, informative features:
1. Bias (1)
2. b_norm
3. s_norm
4. temp_norm
5. pH_norm
6. b_norm * s_norm
7. temp_norm^2
8. pH_norm^2
9. s_norm * temp_norm
10. s_norm * pH_norm

This is a linear model: y = p[0]*1 + p[1]*b_n + ...
This is convex in p. BFGS will find the exact least-squares solution (or very close) 
regardless of the starting point (all ones). The only risk is if the true function 
cannot be well-approximated by these 10 features. 
Given the biological nature, interactions between substrate and temperature/pH are likely 
crucial. The current model failed likely due to optimization failure, not necessarily 
model insufficiency, but 40.0 is very bad. A linear model on normalized features 
should at least achieve a score near 1.0 (seed level) or better if the seed was 
also struggling with optimization.

To improve expressiveness without breaking linearity in params, we can use 
**Radial Basis Functions (RBFs)** with FIXED centers and widths, and let params be the 
amplitudes. 
If we choose centers that cover the input space well, and widths that are broad enough, 
a linear combination of 10 RBFs can approximate a wide variety of smooth functions.
This keeps the problem linear in params (convex), ensuring BFGS convergence.

Let's define 10 RBFs in the 4D space (b, s, temp, pH).
To avoid curse of dimensionality, we can use 1D RBFs on individual variables or 
pairs, or 4D RBFs with large widths.
4D RBFs with large widths are essentially polynomials.
Let's try a mix:
- 1 Global RBF (centered at mean, large width) -> captures overall level
- 3 RBFs on Temp (low, mid, high)
- 3 RBFs on pH (low, mid, high)
- 2 RBFs on Substrate (low, high)
- 1 RBF on Density (mid)

Actually, simpler: Just use a polynomial basis of degree 2 on normalized inputs, 
selecting the most likely relevant terms.
Normalization:
b_n = (b-1)/4
s_n = (s-1)/4
t_n = (temp-1)/39
p_n = (pH-2)/10

Features:
1. 1
2. b_n
3. s_n
4. t_n
5. p_n
6. t_n^2 (Temperature is often quadratic)
7. p_n^2 (pH is often quadratic)
8. s_n * t_n (Substrate effect depends on temp)
9. s_n * p_n (Substrate effect depends on pH)
10. b_n * s_n (Density/substrate interaction)

This is a linear model. BFGS will solve it perfectly.
"""

import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 polynomial features on normalized inputs.
    This structure makes the loss function convex with respect to `params`,
    ensuring the external BFGS optimizer converges to the global minimum.
    """
    # Normalize inputs to [0, 1] range based on task description
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Ensure inputs are 1D arrays for consistent broadcasting if needed, 
    # though they are likely already 1D.
    b_n = np.asarray(b_n).ravel()
    s_n = np.asarray(s_n).ravel()
    t_n = np.asarray(t_n).ravel()
    p_n = np.asarray(p_n).ravel()
    
    p = np.asarray(params).ravel()
    
    # Define the 10 features
    # f0: Bias
    f0 = np.ones_like(b_n)
    
    # f1-f4: Linear terms
    f1 = b_n
    f2 = s_n
    f3 = t_n
    f4 = p_n
    
    # f5-f6: Quadratic terms for Temperature and pH (biologically significant)
    f5 = t_n * t_n
    f6 = p_n * p_n
    
    # f7-f9: Interaction terms (Substrate with Temp, pH, and Density)
    f7 = s_n * t_n
    f8 = s_n * p_n
    f9 = b_n * s_n
    
    # Combine linearly
    # result = p[0]*f0 + ... + p[9]*f9
    result = (p[0] * f0 + 
              p[1] * f1 + 
              p[2] * f2 + 
              p[3] * f3 + 
              p[4] * f4 + 
              p[5] * f5 + 
              p[6] * f6 + 
              p[7] * f7 + 
              p[8] * f8 + 
              p[9] * f9)
              
    return result

# EVOLVE-BLOCK-END
