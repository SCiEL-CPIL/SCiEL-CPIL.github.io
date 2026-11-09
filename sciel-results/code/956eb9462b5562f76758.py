"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
DEADLINE = 150.0  # Wall-clock limit
SAFETY_MARGIN = 5.0  # Seconds to spare

# Precompute fixed RBF centers on a 2x2x2 grid in [0.5, 2.0]^3
# This gives 8 centers. We need 10 params, so we use 8 RBFs + 1 constant + 1 linear-like term
# Actually, let's use 9 RBFs on a 3x3x1 grid? No, 3D domain.
# Let's use 8 centers on a 2x2x2 grid. That uses 8 params.
# We have 10 params.
# Strategy: 8 RBFs + 2 global polynomial terms (e.g., constant and linear in x+y+z)
# Or: 10 RBFs? A 2x2x2 grid is 8. A 2x2x3 grid is 12.
# Let's try 8 RBFs + 1 constant + 1 "trend" term.
# The trend term can be a low-order polynomial to capture global drift that RBFs might miss if bandwidth is too narrow.
# However, RBFs with a reasonable bandwidth should capture global trends if the sum of weights allows it.
# Let's stick to 8 RBFs + 2 linear terms (e.g., u and v) or just 1 constant + 1 linear.
# Actually, a pure RBF basis with a constant term is often sufficient for smooth functions if the centers cover the domain.
# Let's use 9 RBFs? No, 8 is symmetric.
# Let's use 8 RBFs and 2 additional terms: one constant, one linear combination?
# The prompt suggests: "placing 10 fixed, evenly spaced centers".
# 10 centers in 3D is awkward. 2x2x2=8, 2x2x3=12.
# Maybe a 2x3x2 grid? 12 points.
# Maybe a 1D grid of 10 points projected? No.
# Let's interpret "10 fixed, evenly spaced centers" as 10 distinct centers.
# We can place them on a 2x2x2 grid (8 points) plus 2 points at the center of the domain?
# Or a 2x2x2 grid plus the two mid-points of the longest axes?
# Let's just use 8 RBFs on a 2x2x2 grid, and 2 global linear terms (u and v) to handle any residual linear trend.
# This gives 10 parameters total.

# Centers for 2x2x2 grid in [0.5, 2.0]
# x: 0.5, 2.0 -> mid is 1.25. Let's use 0.5 and 2.0? Or 0.5, 1.25, 2.0?
# "Evenly spaced" suggests a grid.
# If we use 2 points per axis: 0.5, 2.0.
# If we use 3 points per axis: 0.5, 1.25, 2.0. That's 27 points. Too many.
# Let's use 8 points: corners of the cube [0.5, 2.0]^3.
# Centers:
# (0.5, 0.5, 0.5), (0.5, 0.5, 2.0), (0.5, 2.0, 0.5), (0.5, 2.0, 2.0)
# (2.0, 0.5, 0.5), (2.0, 0.5, 2.0), (2.0, 2.0, 0.5), (2.0, 2.0, 2.0)

# Let's define the centers
centers = np.array([
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 2.0],
    [0.5, 2.0, 0.5],
    [0.5, 2.0, 2.0],
    [2.0, 0.5, 0.5],
    [2.0, 0.5, 2.0],
    [2.0, 2.0, 0.5],
    [2.0, 2.0, 2.0]
])

# Bandwidth gamma for Gaussian kernel exp(-gamma * ||x - c||^2)
# Domain size is 1.5 per axis. Diagonal is sqrt(3)*1.5 ~ 2.6.
# A gamma of 2.0 seems reasonable. exp(-2 * dist^2).
# If dist=0.5 (half step), exp(-0.5) ~ 0.6.
# If dist=1.5 (max step), exp(-4.5) ~ 0.01.
# This provides local approximation with some global smoothness.
GAMMA = 2.0

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Fallback: return mean of params or just p[0]
        return np.ones_like(x) * params[0]
    
    p = params
    
    # Normalize inputs to [-1, 1] for the linear terms
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Compute RBFs
    # x, y, z are arrays of shape (N,)
    # centers is (8, 3)
    # We need ||[x,y,z] - c_i||^2 for each i
    
    # Stack inputs
    # X is (N, 3)
    X = np.column_stack((x, y, z))
    
    # Centers is (8, 3)
    # diff is (N, 8, 3)
    # We can do this efficiently:
    # X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # (N, 8)
    
    # RBF values
    rbf_vals = np.exp(-GAMMA * sq_dist)  # (N, 8)
    
    # Combine with params
    # p[0..7] correspond to the 8 RBFs
    # p[8] corresponds to u (linear x)
    # p[9] corresponds to v (linear y)
    # We could also use w, but we only have 2 slots left.
    # Maybe a constant is better than a linear term?
    # RBFs sum to ~1 if weights are balanced, so they can represent a constant.
    # But adding a pure constant term p[8] and a linear term p[9]*u might be safer for conditioning?
    # Let's use p[8] as a constant 1.0 and p[9] as a linear trend (u+v+w)/3?
    # Or just p[8] = 1.0 and p[9] = u.
    # Let's use p[8] = 1.0 (constant) and p[9] = (u+v+w)/3.0 (global linear trend)
    
    b_const = np.ones_like(u)
    b_trend = (u + v + w) / 3.0
    
    result = np.sum(rbf_vals * p[0:8], axis=1) + p[8] * b_const + p[9] * b_trend
    
    return result

# EVOLVE-BLOCK-END
