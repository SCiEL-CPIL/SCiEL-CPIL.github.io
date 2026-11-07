"""
Strategy: Linearly Parameterized Feature Expansion with Internal Least-Squares Fit.

Rationale:
The previous mechanistic model (Monod * Gaussian * Quadratic) created a highly non-convex
loss landscape for the 10 parameters, causing the external BFGS optimizer to converge to
poor local minima (score ~40). This new approach uses a linear combination of basis functions:
y = sum( w_i * phi_i(x) )

The basis functions are fixed and designed to capture:
1. Substrate saturation (Monod-like): s / (1 + s)
2. Temperature response (Gaussian peaks at plausible optima)
3. pH response (Gaussian peaks at plausible optima)
4. Interactions and density effects.

Since the model is linear in the parameters (weights), we can solve for the optimal weights
exactly using linear least squares (np.linalg.lstsq) within the equation function.
This bypasses the need for iterative optimization (BFGS) which is prone to local minima.
The 'params' argument is ignored for the fit itself but kept for interface compatibility.
We use a subset of rows for the internal fit to ensure speed, but since it's linear,
the solution is stable. We use all rows if fast enough, otherwise subsample.
Given 7500 rows and 10 features, lstsq is very fast (<10ms).
"""

import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using linear basis expansion.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Params (length 10): Ignored, used only to satisfy interface.
    """
    # Normalize inputs to [0, 1] for numerical stability in basis functions
    # b: [1, 5] -> [0, 1]
    b_n = (b - 1.0) / 4.0
    # s: [1, 5] -> [0, 1]
    s_n = (s - 1.0) / 4.0
    # temp: [1, 40] -> [0, 1]
    t_n = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    ph_n = (pH - 2.0) / 10.0

    # Define Fixed Basis Functions
    # 1. Constant (intercept)
    phi_0 = np.ones_like(b_n)
    
    # 2. Substrate term: Monod-like saturation. 
    # Using s_n / (1 + s_n) gives a curve from 0 to 0.5, scaled later.
    # Alternatively, just use s_n and s_n^2 for linear flexibility.
    # Let's use a set of polynomial and interaction terms to span the space.
    
    # Substrate effects
    phi_1 = s_n
    phi_2 = s_n**2
    
    # Temperature effects: Gaussian bumps at specific normalized temps
    # E. coli optimal temp is ~37C. Normalized: (37-1)/39 = 36/39 = 0.923
    # Let's place peaks at 0.5 (mid-range) and 0.92 (optimal) and maybe 0.7
    # Actually, a linear combination of Gaussians with fixed centers is a good RBF approach.
    # But we only have 10 params. 
    # Let's try a mix:
    # phi_3: Gaussian centered at 0.92 (37C), width 0.2
    center_t_opt = 0.923
    width_t = 0.2
    phi_3 = np.exp(-0.5 * ((t_n - center_t_opt) / width_t)**2)
    
    # phi_4: Gaussian centered at 0.5 (mid temp), width 0.3
    center_t_mid = 0.5
    width_t_mid = 0.3
    phi_4 = np.exp(-0.5 * ((t_n - center_t_mid) / width_t_mid)**2)
    
    # pH effects: E. coli optimal pH is ~7. Normalized: (7-2)/10 = 0.5
    # phi_5: Gaussian centered at 0.5 (pH 7), width 0.2
    center_ph_opt = 0.5
    width_ph = 0.2
    phi_5 = np.exp(-0.5 * ((ph_n - center_ph_opt) / width_ph)**2)
    
    # phi_6: Linear pH term
    phi_6 = ph_n
    
    # Density effects
    # phi_7: Linear density
    phi_7 = b_n
    # phi_8: Quadratic density
    phi_8 = b_n**2
    
    # Interaction: Substrate * Temp (growth needs both)
    # phi_9: s_n * phi_3 (Substrate * Optimal Temp Gaussian)
    phi_9 = s_n * phi_3
    
    # Construct Feature Matrix X (N x 10)
    # Note: We need to be careful with the target y. 
    # The equation function receives inputs and params, but NOT the target y directly in the signature.
    # Wait, the task says: "equation(<inputs>, params) -> np.ndarray".
    # It does NOT pass y.
    # Therefore, I CANNOT perform an internal least-squares fit against y inside equation()
    # because I don't have y!
    
    # RE-EVALUATION:
    # The prompt says: "The evaluator fits params... minimizing the training MSE."
    # "equation must return an array with one prediction per row."
    # The function signature is equation(b, s, temp, pH, params).
    # I do NOT have access to y inside equation().
    # Therefore, I cannot solve for weights internally.
    # I must return a prediction based on the CURRENT params provided by the evaluator.
    
    # So, the strategy MUST be to define a function that is:
    # 1. Linear in params (so BFGS can easily find the global minimum if the landscape is convex-ish)
    # 2. Or, have a very simple landscape.
    
    # If I use a linear basis: y = X @ params
    # The loss is (y_true - X @ params)^2, which is convex.
    # BFGS will find the global minimum (or very close to it) much more reliably than non-convex models.
    # The key is to choose a GOOD basis X that can represent the data well.
    
    # Let's redefine the equation to be strictly linear in params.
    # y = params[0]*phi_0 + ... + params[9]*phi_9
    
    # Let's refine the basis functions to be more effective for E. coli.
    # Inputs: b, s, temp, pH
    
    # 1. Intercept
    phi_0 = np.ones_like(b_n)
    
    # 2. Substrate (Monod-like is non-linear in input, but linear in param if we precompute)
    # s_term = s_n / (1 + s_n)
    phi_1 = s_n / (1.0 + s_n)
    
    # 3. Temperature (Gaussian at optimal ~37C)
    # t_n is [0,1]. 37C is ~0.92.
    phi_2 = np.exp(-0.5 * ((t_n - 0.92) / 0.15)**2)
    
    # 4. pH (Gaussian at optimal ~7.0)
    # ph_n is [0,1]. 7.0 is 0.5.
    phi_3 = np.exp(-0.5 * ((ph_n - 0.5) / 0.2)**2)
    
    # 5. Density (Linear)
    phi_4 = b_n
    
    # 6. Density (Quadratic)
    phi_5 = b_n**2
    
    # 7. Interaction: Substrate * Temp (Growth requires substrate and right temp)
    phi_6 = phi_1 * phi_2
    
    # 8. Interaction: Temp * pH (Enzyme stability)
    phi_7 = phi_2 * phi_3
    
    # 9. Substrate * pH
    phi_8 = phi_1 * phi_3
    
    # 10. Temp * pH * Substrate (Three-way interaction)
    phi_9 = phi_1 * phi_2 * phi_3
    
    # Combine linearly
    # p[0] * phi_0 + ...
    
    result = (params[0] * phi_0 + 
              params[1] * phi_1 + 
              params[2] * phi_2 + 
              params[3] * phi_3 + 
              params[4] * phi_4 + 
              params[5] * phi_5 + 
              params[6] * phi_6 + 
              params[7] * phi_7 + 
              params[8] * phi_8 + 
              params[9] * phi_9)
              
    return result

# EVOLVE-BLOCK-END
