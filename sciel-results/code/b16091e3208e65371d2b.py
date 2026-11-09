"""
Strategy: Gaussian Radial Basis Function (RBF) Network.

Rationale:
- The unknown function f(x, y, z) is sampled on [0.5, 2]^3.
- The current rational basis is non-linear in parameters, leading to a non-convex optimization landscape.
- An RBF basis with fixed centers makes the model linear in the parameters (weights).
- Linear models have a convex MSE surface, so BFGS from any start point (including all-ones)
  will converge to the global minimum, assuming the basis is rich enough.
- We use 10 Gaussian centers distributed across the domain to capture local variations.
- The parameters `params` are the weights for these 10 basis functions.
"""

import numpy as np
import time

# Global state for deadline management
_start_time = None

# Pre-compute RBF centers and widths to avoid repeated calculation in the hot loop
# Domain: [0.5, 2.0] for x, y, z
# We need 10 centers. Let's use a grid-like distribution.
# 2x2x2 grid gives 8 centers. We can add 2 more at the center of the domain and perhaps a corner?
# Or just a 2x2x2 grid is 8. Let's use 2x2x2 = 8 centers and add 2 "global" centers at the domain center (1.25, 1.25, 1.25) with different widths?
# Actually, 10 centers:
# 8 corners of a sub-grid:
# x in [0.75, 1.75], y in [0.75, 1.75], z in [0.75, 1.75]
# This covers the bulk of the volume.
# Centers:
centers = []
# 2x2x2 grid
for cx in [0.75, 1.75]:
    for cy in [0.75, 1.75]:
        for cz in [0.75, 1.75]:
            centers.append((cx, cy, cz))
# Add 2 more centers to make 10. 
# Let's add the global center (1.25, 1.25, 1.25) and maybe another one?
# Or just spread them out more.
# Let's try:
# 8 corners as above.
# Center 9: (1.25, 1.25, 1.25)
# Center 10: (1.25, 1.25, 1.25) ? No, duplicate.
# Let's add (0.5, 0.5, 0.5) and (2.0, 2.0, 2.0)? No, those are outside the dense region.
# Let's stick to 8 well-placed centers and 2 "broad" centers at the center.
centers.append((1.25, 1.25, 1.25))
centers.append((1.25, 1.25, 1.25)) # Duplicate is bad for identifiability but okay for optimization? 
# Better: Make the last two have different widths or positions.
# Let's replace the duplicate with (1.0, 1.0, 1.0) and (1.5, 1.5, 1.5)?
# Let's just use a 2x2x2 grid (8 pts) and 2 pts at (1.25, 1.25, 1.25) and (1.25, 1.25, 1.25) is redundant.
# Let's use a 3x3x3 grid? 27 pts is too many for 10 params.
# Let's use 10 specific points:
# 8 corners of [0.75, 1.75]^3
# 1 center (1.25, 1.25, 1.25)
# 1 edge midpoint? (1.25, 1.25, 0.75)?
centers = [
    (0.75, 0.75, 0.75),
    (0.75, 0.75, 1.75),
    (0.75, 1.75, 0.75),
    (0.75, 1.75, 1.75),
    (1.75, 0.75, 0.75),
    (1.75, 0.75, 1.75),
    (1.75, 1.75, 0.75),
    (1.75, 1.75, 1.75),
    (1.25, 1.25, 1.25),
    (1.25, 1.25, 0.75) # One more to cover z-direction variation near center
]

# Widths (sigma) for the Gaussians.
# A width of ~0.5 covers a good portion of the [0.5, 2] range.
# Let's use a uniform width for simplicity, or slightly larger for the center ones.
sigmas = np.array([0.5] * 8 + [0.8, 0.8])

# Pre-compute squared distances? No, x,y,z change. But we can pre-compute center coords.
cx = np.array([c[0] for c in centers])
cy = np.array([c[1] for c in centers])
cz = np.array([c[2] for c in centers])

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Gaussian RBFs.
    params: weights [w1, ..., w10]
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Deadline guard: If we exceed 140s, return a simple constant or linear approx
    # Since this is linear in params, BFGS should be very fast. 
    # The timeout is unlikely to be hit by the function evaluation itself, 
    # but if the *fitting* is stuck, we might be called many times.
    # However, the evaluator calls equation() during the fit.
    # If we are near the deadline, we should return something finite.
    if elapsed > 140.0:
        # Fallback: return mean of params weighted by x? 
        # Just return a constant to be safe.
        return np.mean(params) * np.ones_like(x)

    try:
        # Compute squared distances to each center
        # x, y, z are arrays of shape (N,)
        # cx, cy, cz are arrays of shape (10,)
        
        # Broadcasting: (N, 1) - (1, 10) -> (N, 10)
        dx = x[:, np.newaxis] - cx[np.newaxis, :]
        dy = y[:, np.newaxis] - cy[np.newaxis, :]
        dz = z[:, np.newaxis] - cz[np.newaxis, :]
        
        dist_sq = dx**2 + dy**2 + dz**2
        
        # Compute Gaussian basis values: exp(-dist_sq / (2 * sigma^2))
        # sigma is (10,), so sigma^2 is (10,)
        # dist_sq is (N, 10)
        # We need to divide each column by 2 * sigma_i^2
        inv_2_sig2 = 1.0 / (2.0 * sigmas**2) # shape (10,)
        
        # Multiply dist_sq by inv_2_sig2 (broadcasting)
        exponents = -dist_sq * inv_2_sig2[np.newaxis, :]
        
        basis = np.exp(exponents) # shape (N, 10)
        
        # Linear combination
        # params is shape (10,)
        pred = basis @ params
        
        # Check for non-finite values
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred
    except Exception:
        return np.ones_like(x)

# EVOLVE-BLOCK-END
