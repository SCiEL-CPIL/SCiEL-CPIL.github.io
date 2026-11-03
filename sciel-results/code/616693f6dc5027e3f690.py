import numpy as np
import time

N = 300

def generate_hex_grid(r):
    # Generate a hex grid that fits in radius 1-r
    # Spacing s = 2r
    # Points (i*s, j*s*sqrt(3)/2) with offset
    # Filter to circle
    # If not enough points, reduce spacing? 
    # For binary search, we assume r is close to optimal, so hex grid should be dense enough.
    # If hex grid has < N points, we need to add random points or use a different init.
    # But for r ~ 0.053, area is pi*1^2 / (pi*r^2) ~ 1/0.0028 ~ 350. So hex grid should have ~300 points.
    
    s = 2 * r
    # Grid size
    k = int(1.0 / s) + 2
    points = []
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1 - r:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        # Fallback: random points in circle
        # Or just return what we have? 
        # Better: random points in the annulus or disk
        theta = np.random.uniform(0, 2*np.pi, N)
        rho = np.sqrt(np.random.uniform(0, (1-r)**2, N))
        points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
    else:
        # Select N points closest to center? Or random N?
        # Random selection might be better to avoid bias, but center is dense.
        # Let's take the first N or random N.
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    return points

def check_feasibility(r, init_points, max_steps=100, alpha=0.01):
    # Relaxation
    C = init_points.copy()
    for step in range(max_steps):
        # Compute forces
        # Diff[i, j] = C[i] - C[j]
        Diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        Dist = np.sqrt(np.sum(Diff**2, axis=2))
        
        # Avoid division by zero
        np.fill_diagonal(Dist, 1.0)
        
        # Force magnitude: 1/d^2 if d < 2r, else 0
        # We want to push apart if overlapping
        mask = Dist < 2 * r
        # Force vector direction: Diff / Dist
        # Force magnitude: 1 / Dist^2
        # F = sum over j of (C_i - C_j) / Dist^2
        
        # Only apply force where mask is true
        # We can use a weighted sum
        # W[i, j] = 1/Dist[i, j]^2 if mask[i, j] else 0
        W = np.zeros_like(Dist)
        W[mask] = 1.0 / (Dist[mask] ** 2)
        # Set diagonal to 0
        np.fill_diagonal(W, 0.0)
        
        # Forces[i] = sum_j W[i, j] * Diff[i, j]
        Forces = np.sum(W[:, :, np.newaxis] * Diff, axis=1)
        
        # Boundary force
        # If |C_i| > 1 - r, push inward
        Rad = np.linalg.norm(C, axis=1)
        boundary_mask = Rad > (1 - r)
        if np.any(boundary_mask):
            # Force = -k * C_i / |C_i|
            # Scale force
            boundary_forces = -10.0 * C[boundary_mask] / Rad[boundary_mask, np.newaxis]
            Forces[boundary_mask] += boundary_forces
            
        # Update
        C += alpha * Forces
        
        # Optional: check convergence
        if np.linalg.norm(Forces) < 1e-6:
            break
            
    # Check validity
    # Min distance
    Dist = np.sqrt(np.sum(Diff**2, axis=2))
    np.fill_diagonal(Dist, np.inf)
    min_dist = np.min(Dist)
    
    # Max radius
    max_rad = np.max(np.linalg.norm(C, axis=1))
    
    return min_dist >= 2 * r - 1e-9 and max_rad <= 1 - r + 1e-9, C

def construct():
    start_time = time.time()
    deadline = 100.0 # Safety margin
    
    # Initial bounds
    lo = 0.05
    hi = 0.055
    
    # Check hi first to see if we can go higher? 
    # Best known is 0.05315.
    # Let's set hi = 0.054
    
    best_r = lo
    best_C = None
    
    # Binary search
    for _ in range(20):
        if time.time() - start_time > deadline:
            break
            
        mid = (lo + hi) / 2
        init = generate_hex_grid(mid)
        valid, C = check_feasibility(mid, init, max_steps=200, alpha=0.05)
        
        if valid:
            best_r = mid
            best_C = C
            lo = mid
        else:
            hi = mid
            
    # If no valid found, return seed-like or best effort
    if best_C is None:
        # Fallback to hex grid for lo
        best_C = generate_hex_grid(best_r)
        
    return best_C
