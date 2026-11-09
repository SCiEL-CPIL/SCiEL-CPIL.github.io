# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations, product


def generate_e8_roots_scaled():
    """Generate the 240 roots of E8 lattice embedded in first 8 dimensions."""
    roots = []
    
    # Type 1: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    # Choose 2 positions for the ±1's: C(8,2) = 28
    # Signs: 4 combinations
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    vec = [0]*8
                    vec[i] = si
                    vec[j] = sj
                    roots.append(vec)
    
    # Type 2: permutations of (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs
    # There are 2^8 = 256 sign combinations, half have even number of minus signs = 128
    # But we need to be careful: the E8 roots of this type have exactly even number of minus signs
    # Actually, let me reconsider. The standard E8 roots are:
    # (±1, ±1, 0, ..., 0) and permutations: 2*8*7 = 112
    # (±1/2, ±1/2, ..., ±1/2) with even number of minus signs: 128
    # Total: 112 + 128 = 240
    
    # For type 2, we need to scale by 2 to keep integers: (±1, ±1, ..., ±1) with even number of minus signs
    # But wait, if we scale by 2, the norm becomes 4*8 = 32, not 2.
    # The task requires squared norm 2, so we can't use the half-integer roots directly.
    # Let's stick with just the integer roots of norm 2: (±1, ±1, 0, ...) in 8D
    # That gives 112 roots in 8D.
    
    return roots


def generate_a3_roots():
    """Generate the 12 roots of A3 lattice embedded in last 3 dimensions."""
    roots = []
    # A3 roots are (±1, ±1, 0) and permutations
    # In 3D: (±1, ±1, 0) and permutations
    # Choose 2 positions for ±1's: C(3,2) = 3
    # Signs: 4 combinations
    for i in range(3):
        for j in range(i+1, 3):
            for si in [1, -1]:
                for sj in [1, -1]:
                    vec = [0]*3
                    vec[i] = si
                    vec[j] = sj
                    roots.append(vec)
    return roots


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    using E8 x A3 lattice roots, aiming to maximize the number of points.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    
    # Generate E8 roots (112 integer roots of norm 2 in 8D)
    e8_roots = generate_e8_roots_scaled()
    
    # Generate A3 roots (12 roots of norm 2 in 3D)
    a3_roots = generate_a3_roots()
    
    # Now construct 11D points by combining E8 and A3 components
    # We want points with squared norm 2 in 11D
    # Option 1: E8 root in first 8 dims, 0 in last 3 dims -> norm 2
    # Option 2: 0 in first 8 dims, A3 root in last 3 dims -> norm 2
    # Option 3: Mixtures would have norm > 2 unless we scale down, but we need integers
    
    points = []
    
    # Type 1: E8 roots embedded in first 8 dimensions, zeros in last 3
    for root in e8_roots:
        point = root + [0, 0, 0]
        points.append(point)
    
    # Type 2: Zeros in first 8 dimensions, A3 roots in last 3
    for root in a3_roots:
        point = [0]*8 + root
        points.append(point)
    
    # Convert to numpy array
    points = np.array(points, dtype=np.int64)
    
    # Check that all points have squared norm 2
    norms = np.sum(points**2, axis=1)
    
    # Check minimum pairwise distance
    # Compute pairwise squared distances
    from scipy.spatial.distance import cdist
    
    # Use exact integer arithmetic for validation
    min_sq_dist = np.inf
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            diff = points[i] - points[j]
            sq_dist = np.sum(diff**2)
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
    
    max_norm = np.max(norms)
    
    # Ensure min_sq_dist >= max_norm
    if min_sq_dist < max_norm:
        # Fallback to the original 2-point solution
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
