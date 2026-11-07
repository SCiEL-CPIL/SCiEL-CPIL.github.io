"""Rational function model with quadratic numerator and denominator for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit

    # Ensure x and y are 1D arrays for consistent shape handling
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Rational function with hybrid basis:
    # Numerator: Polynomial (quadratic)
    # Denominator: 1 + Trigonometric terms
    #
    # This form captures polynomial trends in the numerator and
    # bounded, smooth variations in the denominator, which can
    # model periodic or saturating behaviors more stably than
    # a purely polynomial denominator.

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute numerator (Polynomial)
    numerator = (p0 + p1 * x + p2 * y + 
                 p3 * x**2 + p4 * y**2 + p5 * x * y)

    # Compute denominator (1 + Trigonometric)
    # Using sin/cos keeps the denominator bounded and smooth
    denominator = (1.0 + p6 * np.sin(x) + p7 * np.sin(y) + 
                   p8 * np.cos(x) + p9 * np.cos(y))

    # Avoid division by zero or near-zero
    # Use a small epsilon to prevent numerical instability
    eps = 1e-12
    # Clip denominator to avoid values too close to zero
    # We want to preserve the sign but avoid division by ~0
    sign_denom = np.sign(denominator)
    abs_denom = np.abs(denominator)
    # Where |denom| < eps, replace with eps * sign (or eps if sign is 0)
    safe_denom = np.where(abs_denom < eps, eps * np.where(sign_denom == 0, 1.0, sign_denom), denominator)

    result = numerator / safe_denom

    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return a safe constant if numerical issues arise
        return np.zeros_like(x)

    return result
# EVOLVE-BLOCK-END
