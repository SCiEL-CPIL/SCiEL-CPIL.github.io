"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Strategy: Shifted Legendre polynomial basis of degree up to 3.
    # Inputs are in [0.5, 2.0]. We normalize to [-1, 1] to use standard Legendre polynomials.
    # This provides an orthogonal basis on the domain, which is well-conditioned for least squares fitting.
    # We use 10 terms:
    # 1. Constant
    # 2. Linear: P1(x), P1(y), P1(z)
    # 3. Quadratic: P2(x), P2(y), P2(z)
    # 4. Cubic: P3(x)  (Only one cubic term to fit in 10 params? Or mix?)
    
    # Let's count:
    # P0 = 1
    # P1(u) = u
    # P2(u) = 0.5*(3*u^2 - 1)
    # P3(u) = 0.5*(5*u^3 - 3*u)
    
    # We have 10 params.
    # Option A: Full tensor product up to degree 1 (linear): 1 + 3 = 4 terms.
    # Option B: Add quadratic main effects: +3 = 7 terms.
    # Option C: Add cubic main effects: +3 = 10 terms.
    # This gives: 1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3 (in Legendre form).
    # This captures smooth univariate functions well but no interactions.
    
    # Option D: 1, x, y, z, xy, xz, yz, x^2, y^2, z^2? 
    # The prompt suggests: "combined with the constant term and one cross-interaction term".
    # But we have 10 params. 
    # Let's stick to the directive: "basis of shifted Legendre polynomials of degree up to 3 ... combined with ... one cross-interaction term".
    # This phrasing is a bit ambiguous. It might mean a specific set of 10 terms.
    # A robust choice for 10 terms that balances main effects and interactions:
    # 1. P0 (1)
    # 2. P1(x)
    # 3. P1(y)
    # 4. P1(z)
    # 5. P2(x)
    # 6. P2(y)
    # 7. P2(z)
    # 8. P1(x)*P1(y)  (Interaction)
    # 9. P1(x)*P1(z)  (Interaction)
    # 10. P1(y)*P1(z) (Interaction)
    
    # This set includes all main effects up to quadratic and all linear-linear interactions.
    # This is a "quadratic with interactions" model, but using Legendre polynomials for conditioning.
    # It is more flexible than pure main effects and likely better for general smooth functions than pure RBFs with only 10 centers (which might be too sparse or too dense).
    
    # Normalize inputs to [-1, 1]
    # Domain: [0.5, 2.0]
    # Center: 1.25
    # Half-width: 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Compute Legendre polynomials
    # P0(t) = 1
    P0 = np.ones_like(u)
    
    # P1(t) = t
    P1_u = u
    P1_v = v
    P1_w = w
    
    # P2(t) = 0.5 * (3*t^2 - 1)
    P2_u = 0.5 * (3 * u**2 - 1)
    P2_v = 0.5 * (3 * v**2 - 1)
    P2_w = 0.5 * (3 * w**2 - 1)
    
    # P3(t) = 0.5 * (5*t^3 - 3*t)
    # We won't use P3 in this specific 10-term model if we use interactions.
    # If we used main effects only, we would have 10 terms:
    # 1, u, v, w, P2_u, P2_v, P2_w, P3_u, P3_v, P3_w
    # Let's compare:
    # Model 1 (Main effects up to cubic):
    #   1, u, v, w, P2_u, P2_v, P2_w, P3_u, P3_v, P3_w
    # Model 2 (Main effects up to quad + interactions):
    #   1, u, v, w, P2_u, P2_v, P2_w, u*v, u*w, v*w
    
    # Model 2 is generally better for functions with interactions.
    # Model 1 is better for separable functions.
    # Given "unknown function", interactions are a common source of error.
    # Also, the prompt specifically mentioned "one cross-interaction term" in the directive text, 
    # but we have 10 params. The directive says "combined with ... one cross-interaction term". 
    # It might imply that the base is 9 terms + 1 interaction?
    # Or it's just a suggestion.
    # Let's go with Model 2 as it is a standard quadratic regression basis (with orthogonalization).
    
    # Construct the design matrix features
    # Feature 1: P0
    # Feature 2: P1_u
    # Feature 3: P1_v
    # Feature 4: P1_w
    # Feature 5: P2_u
    # Feature 6: P2_v
    # Feature 7: P2_w
    # Feature 8: P1_u * P1_v
    # Feature 9: P1_u * P1_w
    # Feature 10: P1_v * P1_w
    
    # Note: P1_u * P1_v is just u*v.
    
    F = np.column_stack((
        P0,
        P1_u,
        P1_v,
        P1_w,
        P2_u,
        P2_v,
        P2_w,
        P1_u * P1_v,
        P1_u * P1_w,
        P1_v * P1_w
    ))
    
    # F shape: (N, 10)
    # params shape: (10,)
    
    result = F @ params
    
    return result
# EVOLVE-BLOCK-END
