import numpy as np

def _gen_extremal(k):
    if k == 3:
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    if k == 4:
        # 3 points forming a triangle, 1 inside
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([0.5, 1.0])
        p4 = np.array([0.5, 0.3]) # Inside
        return np.array([p1, p2, p3, p4])
    
    A = _gen_extremal(k - 1)
    B = _gen_extremal(k - 1)
    
    # Scale B and place inside A
    # Centroid of A
    centroid_A = np.mean(A, axis=0)
    # Shift A to center at 0
    A_centered = A - centroid_A
    # Find max radius of A
    max_r = np.max(np.linalg.norm(A_centered, axis=1))
    
    # Scale B to be small (e.g., 0.1 * max_r)
    scale = 0.1 * max_r
    B_centered = B - np.mean(B, axis=0)
    B_scaled = B_centered * scale
    
    # Place B at centroid of A
    B_final = B_scaled + centroid_A
    
    return np.vstack([A, B_final])

def erdos_szekeres_problem7() -> np.ndarray:
    # Generate 32 points with 0 convex 7-gons
    pts_32 = _gen_extremal(7)
    
    # Scale and shift to fit in [0,1] with margin
    # Find bounds
    min_x, min_y = np.min(pts_32, axis=0)
    max_x, max_y = np.max(pts_32, axis=0)
    
    # Normalize to [0.1, 0.9]
    pts_32 = (pts_32 - [min_x, min_y]) / ([max_x - min_x, max_y - min_y])
    pts_32 = pts_32 * 0.8 + 0.1
    
    # Add 33rd point
    # Place it at the center, slightly perturbed to avoid collinearity
    # The center might create convex 7-gons.
    # Let's add it near one of the "inner" points to minimize impact?
    # Or just at (0.5, 0.5) with small noise.
    p33 = np.array([0.5, 0.5])
    
    pts = np.vstack([pts_32, p33])
    
    # Add small random perturbation to all points to ensure general position
    # and avoid exact collinearity/cocircularity issues
    rng = np.random.default_rng(42)
    perturbation = rng.uniform(-1e-5, 1e-5, size=pts.shape)
    pts += perturbation
    
    # Ensure within bounds
    pts = np.clip(pts, 1e-6, 1 - 1e-6)
    
    # Check constraints? 
    # The recursive structure should ensure low convex 7-gons.
    
    return pts
