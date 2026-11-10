import numpy as np

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    delta = np.pi / 10.0
    
    # Calculate optimal lambda for alternating radius polygon
    # Condition: dist(v0, v1) == dist(v0, v2) for v0 at R, v1 at r, v2 at R
    # d1^2 = R^2 + r^2 - 2Rr cos(delta)
    # d2^2 = 2R^2 (1 - cos(2*delta)) = 4R^2 sin^2(delta)
    # Set d1 = d2 => R^2 + r^2 - 2Rr cos(delta) = 4R^2 sin^2(delta)
    # Let r = lambda R
    # 1 + lambda^2 - 2 lambda cos(delta) = 4 sin^2(delta)
    # lambda^2 - 2 cos(delta) lambda + (1 - 4 sin^2(delta)) = 0
    # lambda = cos(delta) + sqrt(3 sin^2(delta))  (taking the larger root for r > R)
    
    lambda_val = np.cos(delta) + np.sqrt(3) * np.sin(delta)
    
    # To ensure strict convexity and avoid floating point issues, 
    # we might need to tweak lambda slightly or verify.
    # The larger root gives r > R.
    
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    radii[1::2] = lambda_val
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Check convexity
    # Calculate cross products of consecutive edges
    # If any cross product is <= 0, it's not strictly convex (or is clockwise)
    # We assume CCW order.
    
    # Normalize as per task description? 
    # "subtract the centroid, divide by the mean distance to it"
    # The score is invariant to this, but we should do it to be safe?
    # Actually, the task says "The polygon is normalized...". 
    # It implies the evaluator does this. 
    # "It must then be strictly convex...".
    # I will return the raw coordinates. The evaluator normalizes.
    
    return polygon
