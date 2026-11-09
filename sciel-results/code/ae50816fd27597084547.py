"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
DEADLINE = 150.0  # Wall-clock limit
SAFETY_MARGIN = 5.0  # Seconds to spare

# Strategy: Use a basis of shifted Legendre polynomials of degree up to 3
# applied to each normalized input, combined with cross-interaction terms.
#
# Normalized inputs:
# u = (x - 1.25) / 0.75  -> [-1, 1]
# v = (y - 1.25) / 0.75  -> [-1, 1]
# w = (z - 1.25) / 0.75  -> [-1, 1]
#
# Shifted Legendre polynomials on [-1, 1]:
# P0(t) = 1
# P1(t) = t
# P2(t) = (3t^2 - 1) / 2
# P3(t) = (5t^3 - 3t) / 2
#
# We have 10 parameters.
# Let's allocate them as follows to capture curvature and interactions:
#
# Pure terms (using Legendre polynomials):
# 1. P0(u) = 1  (constant)
# 2. P1(u) = u
# 3. P2(u) = (3u^2 - 1)/2
# 4. P1(v) = v
# 5. P2(v) = (3v^2 - 1)/2
# 6. P1(w) = w
# 7. P2(w) = (3w^2 - 1)/2
#
# Interaction terms (using low-order cross terms):
# 8. u * v  (linear interaction)
# 9. u * w  (linear interaction)
# 10. v * w (linear interaction)
#
# This gives us:
# - Constant
# - Linear terms in each variable
# - Quadratic terms in each variable (via P2)
# - Pairwise linear interactions
#
# This basis is well-conditioned due to the orthogonality of Legendre polynomials
# for the pure terms, and the interaction terms capture cross-dependencies.
#
# Alternatively, we could include P3 terms if we drop some interactions,
# but let's start with this balanced approach.

def _legendre_p0(t):
    return np.ones_like(t)

def _legendre_p1(t):
    return t

def _legendre_p2(t):
    return (3.0 * t**2 - 1.0) / 2.0

def _legendre_p3(t):
    return (5.0 * t**3 - 3.0 * t) / 2.0

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Fallback: return constant
        return np.ones_like(x) * params[0]
    
    p = params
    
    # Normalize inputs to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Compute basis functions
    # Pure terms
    b0 = _legendre_p0(u)  # 1
    b1 = _legendre_p1(u)  # u
    b2 = _legendre_p2(u)  # (3u^2 - 1)/2
    b3 = _legendre_p1(v)  # v
    b4 = _legendre_p2(v)  # (3v^2 - 1)/2
    b5 = _legendre_p1(w)  # w
    b6 = _legendre_p2(w)  # (3w^2 - 1)/2
    
    # Interaction terms
    b7 = u * v
    b8 = u * w
    b9 = v * w
    
    # Combine with parameters
    result = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    
    return result

# EVOLVE-BLOCK-END
