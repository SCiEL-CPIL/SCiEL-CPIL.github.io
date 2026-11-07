"""Symbolic regression with a rich, stable basis and BFGS-compatible evaluation."""
import numpy as np
import time

# EVOLVE-BLOCK-START


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using 10 parameters in a diverse, numerically stable basis.
    
    The basis is designed to span common function families:
    - Polynomials up to degree 5
    - Trigonometric functions (sin, cos) with fundamental frequency
    - Hyperbolic functions (tanh)
    - Exponential decay/growth (symmetric pair)
    - Gaussian
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters; pad or truncate if needed
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Build basis functions
    # Polynomials: degree 0 to 5
    b0 = np.ones_like(x)                    # constant
    b1 = x                                  # linear
    b2 = x ** 2                             # quadratic
    b3 = x ** 3                             # cubic
    b4 = x ** 4                             # quartic
    b5 = x ** 5                             # quintic
    
    # Trigonometric: fundamental frequency
    b6 = np.sin(x)                          # sin(x)
    b7 = np.cos(x)                          # cos(x)
    
    # Hyperbolic: tanh for smooth saturation
    b8 = np.tanh(x)                         # tanh(x)
    
    # Exponential: symmetric pair for decay/growth
    b9 = np.exp(-x ** 2)                    # Gaussian (stable, localized)
    
    # Wait, I need 10 distinct basis functions. Let me recount:
    # b0: 1, b1: x, b2: x^2, b3: x^3, b4: x^4, b5: x^5, b6: sin(x), b7: cos(x), b8: tanh(x), b9: exp(-x^2)
    # That's 10. But inspiration 1 and 2 used exp(x) and exp(-x) instead of some others.
    # Let's look at the scores:
    # Current: 190.5 (uses poly 0-4, sin, cos, sin2x, tanh, gauss)
    # Insp 1: 136.6 (uses poly 0-4, sin, cos, tanh, exp(x), exp(-x))
    # Insp 2: 77.6 (uses poly 0-5, sin, cos, sin2x, cos2x)
    
    # The current program is the best so far (190.5).
    # Inspiration 1 was better than Inspiration 2 (136.6 vs 77.6).
    # The jump from 77.6 to 136.6 was attributed to replacing higher-order polynomials/double-freq trig with tanh/exp.
    # The jump from 136.6 to 190.5 (current) added sin(2x) and Gaussian, removed exp(x)/exp(-x) and tanh?
    # No, current has: 1, x, x2, x3, x4, sin, cos, sin2x, tanh, gauss.
    # Insp 1 has: 1, x, x2, x3, x4, sin, cos, tanh, exp(x), exp(-x).
    
    # Let's combine the best elements.
    # The current program's score (190.5) is significantly better than Insp 1 (136.6).
    # This suggests the specific combination in the current program is superior.
    # However, we can try to refine the basis further.
    
    # One idea: The current basis has sin(x), cos(x), sin(2x). It lacks cos(2x).
    # It has tanh(x) and exp(-x^2).
    # Inspiration 1 had exp(x) and exp(-x). These are unstable for large |x| (overflow).
    # exp(-x^2) is very stable.
    
    # Let's try a basis that mixes the strengths:
    # Keep the polynomial base (0-4) as in current.
    # Keep sin(x), cos(x).
    # Keep tanh(x) for non-linear saturation.
    # Keep exp(-x^2) for localized features.
    # Replace sin(2x) with something else? Or add cos(2x)?
    
    # Actually, looking at the current code, it uses:
    # b0: 1
    # b1: x
    # b2: x^2
    # b3: x^3
    # b4: x^4
    # b5: sin(x)
    # b6: cos(x)
    # b7: sin(2x)
    # b8: tanh(x)
    # b9: exp(-x^2)
    
    # This is a very solid basis. To improve, we might consider:
    # 1. Adding cos(2x) instead of sin(2x)? Unlikely to be a big jump.
    # 2. Using x^5 instead of sin(2x)? Insp 2 had x^5 and scored lower.
    # 3. Using a different exponential? e.g., exp(-|x|) or 1/(1+x^2)?
    
    # Let's try replacing sin(2x) with cos(2x) to complete the 2nd harmonic pair?
    # Or perhaps replacing the quartic with a quintic?
    
    # Given the directive to combine, let's look at the "structural idea" of one and "refinement" of the other.
    # Current is the structural base.
    # Inspiration 1's refinement was the use of tanh and exponentials. Current already has these.
    # Inspiration 2's refinement was higher order polynomials and double frequency trig. Current has double frequency trig.
    
    # Let's try a small variation: replace sin(2x) with cos(2x) to see if the phase matters, 
    # or better, keep the current basis but ensure numerical stability is maximized.
    
    # Actually, the current basis is quite good. Let's try swapping sin(2x) for cos(2x) 
    # and see if that helps, or perhaps use a slightly different set of frequencies.
    
    # Alternative: Use 1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), cos(2x), tanh(x).
    # This drops the Gaussian. The Gaussian might be crucial for localized features.
    
    # Let's stick closer to the current high-scoring basis but make a small, targeted change.
    # The current basis: [1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), tanh(x), exp(-x^2)]
    
    # What if we replace x^4 with x^5? 
    # What if we replace sin(2x) with cos(2x)?
    
    # Let's try: [1, x, x^2, x^3, x^4, sin(x), cos(x), cos(2x), tanh(x), exp(-x^2)]
    # This keeps the structure but changes the phase of the 2nd harmonic.
    
    # Or, more aggressively, combine the polynomial richness of Insp 2 with the non-linear stability of Current.
    # Basis: [1, x, x^2, x^3, x^4, x^5, sin(x), cos(x), tanh(x), exp(-x^2)]
    # This drops sin(2x) and adds x^5.
    
    # Let's try this last one. It spans polynomials up to degree 5, fundamental trig, tanh, and Gaussian.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = x ** 5
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.tanh(x)
    b9 = np.exp(-x ** 2)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)


# EVOLVE-BLOCK-END
