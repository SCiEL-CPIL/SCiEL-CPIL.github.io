"""
Symbolic regression using a rational function basis.
The hidden function is approximated by a ratio of two low-degree polynomials:
y = (p0 + p1*x + p2*x^2 + p3*x^3) / (1 + q1*x + q2*x^2)
The denominator is constrained to be strictly positive by mapping parameters through
a softplus-like function (1 + 0.5*(exp(z) + z)) or similar, ensuring BFGS stability
and the ability to capture hyperbolic/exponential trends that linear bases miss.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Start timer for safety margin
_START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s buffer before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 6 coefficients (p0, p1, p2, p3, q1, q2)
    Returns: predicted values
    """
    if x.size == 0:
        return np.array([])
    
    # Check time limit
    if time.time() - _START_TIME > TIME_LIMIT:
        # Return a simple fallback if time is up (should not happen in normal flow)
        return np.zeros_like(x)

    # Extract parameters
    # We have 10 params available, but we only need 6 for this model.
    # The optimizer will likely zero out the unused ones or ignore them if we structure the loss correctly,
    # but since we control the equation, we just use the first 6.
    # Note: The evaluator fits ALL 10 params. To ensure the extra params don't break things,
    # we can either:
    # 1. Ignore params[6:]
    # 2. Use them in a way that vanishes (e.g., multiply by 0)
    # Ignoring them is safest and standard for "over-parameterized" models where the loss
    # gradient w.r.t unused params is zero.
    
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    q1, q2 = params[4], params[5]
    
    # Construct numerator: P(x) = p0 + p1*x + p2*x^2 + p3*x^3
    # We use Horner's method for numerical stability
    num = p0 + x * (p1 + x * (p2 + x * p3))
    
    # Construct denominator: Q(x) + 1
    # We need Q(x) > -1 to keep denominator positive.
    # A simple way to enforce positivity of the whole denominator D = 1 + q1*x + q2*x^2
    # is to constrain q1 and q2 such that the minimum of D is > 0.
    # However, BFGS works on unconstrained parameters. 
    # A robust trick: Let q1 = 2 * tanh(a1) * scale, q2 = 2 * tanh(a2) * scale?
    # Or simpler: Just ensure the denominator is 1 + (bounded terms).
    # If we map q1_raw -> q1 = 2*tanh(q1_raw), then q1 in [-2, 2].
    # If we map q2_raw -> q2 = 2*tanh(q2_raw), then q2 in [-2, 2].
    # Max |q1*x + q2*x^2| for x in [-2,2] is roughly 2*2 + 2*4 = 12.
    # So D could be 1 - 12 = -11. This is unstable.
    
    # Better approach: Scale the denominator terms to be small.
    # Let q1 = 0.5 * tanh(params[4])
    # Let q2 = 0.5 * tanh(params[5])
    # Then |q1*x + q2*x^2| <= 0.5*2 + 0.5*4 = 3.
    # D = 1 + ... can still be negative (1-3 = -2).
    
    # Even safer: Use exponential mapping for the "stability" part or just constrain the range tightly.
    # Let's use a softplus-like bound that keeps the correction term small.
    # q1 = tanh(params[4]) * 0.5
    # q2 = tanh(params[5]) * 0.25
    
    # Actually, the most robust way to ensure D > 0 for x in [-2, 2] with only 2 params is:
    # D = 1 + a * exp(-b * x^2) ? No, that's not a polynomial.
    # The task asks for a rational function (ratio of polynomials).
    # Let's stick to polynomials but with very tight bounds.
    # q1 = 0.1 * tanh(params[4])
    # q2 = 0.05 * tanh(params[5])
    # Max deviation: 0.1*2 + 0.05*4 = 0.2 + 0.2 = 0.4.
    # D is in [0.6, 1.4]. Always positive. Stable.
    
    q1_bounded = 0.1 * np.tanh(params[4])
    q2_bounded = 0.05 * np.tanh(params[5])
    
    den = 1.0 + x * q1_bounded + (x * x) * q2_bounded
    
    # Safety check: if den is too close to 0, clip it. 
    # With the bounds above, den >= 0.6, so it's safe.
    # But just in case of numerical overflow in tanh or something weird:
    den = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6 + 1e-6, den)
    # Actually, since we know den > 0.5, we can just do:
    den = np.maximum(den, 1e-6)
    
    return num / den
# EVOLVE-BLOCK-END
