import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural change: Instead of a fixed polynomial, we use a hybrid model
    # that combines:
    # 1. A flexible "radial basis" or Gaussian-like response for temperature (often unimodal in biology)
    # 2. Sigmoid-like saturation for nutrients (Monod-like but flexible)
    # 3. Linear terms for other factors
    # 
    # However, the directive says "Replace the multiplicative Gaussian and Monod terms with a single, unconstrained 4th-order multivariate polynomial".
    # The current program IS that polynomial. It scored 1.0171.
    # The parent (which had Monod/Gaussian) scored 26.43 (much worse).
    # So the polynomial is an improvement over the structured model.
    #
    # Why might the polynomial be stuck at 1.0171?
    # - It might be under-fitting complex interactions.
    # - It might be over-fitting noise.
    # - The optimizer (BFGS from all ones) might be getting stuck in a local minimum.
    #
    # The directive says: "Make a STRUCTURAL change: a different design or modelling family... A first try in a new direction may score worse; that is acceptable."
    #
    # Let's try a different family: A Product of Terms (POT) model, but with more flexibility.
    # Or, let's try a different polynomial basis.
    #
    # Current basis:
    # 1, b, s, t, ph, b*s, b*t, b*ph, s*t, s*ph
    #
    # What if we include higher-order terms for the most important variable?
    # Usually, temperature (t) has a strong non-linear effect (bell curve).
    # Nutrients (b, s) have saturation effects.
    #
    # Let's try a model that explicitly models:
    # - Temperature as a Gaussian: exp(-((t - mu)^2) / (2*sigma^2))
    # - Nutrients as Monod: s / (s + K)
    #
    # But the directive specifically suggested moving AWAY from this to a polynomial.
    # The current program IS the polynomial.
    #
    # Perhaps the issue is that the polynomial is too simple (only 2nd order interactions, no cubes/quartics).
    # Let's try a richer polynomial basis.
    #
    # New Basis Idea:
    # 1. Constant
    # 2. b
    # 3. s
    # 4. t
    # 5. ph
    # 6. b^2
    # 7. s^2
    # 8. t^2
    # 9. t^3  (to better capture the bell shape)
    # 10. t^4
    #
    # This focuses on capturing the non-linearity of temperature, which is often the most complex factor in growth rate.
    # It drops the interactions, assuming that the main effect of temperature is the dominant non-linear term.
    
    y = (p[0] 
         + p[1] * b_norm 
         + p[2] * s_norm 
         + p[3] * t_norm 
         + p[4] * ph_norm 
         + p[5] * b_norm**2 
         + p[6] * s_norm**2 
         + p[7] * t_norm**2 
         + p[8] * t_norm**3 
         + p[9] * t_norm**4)
    
    return y
