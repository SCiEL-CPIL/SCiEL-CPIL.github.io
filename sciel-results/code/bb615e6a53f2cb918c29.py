"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_objective(x, n):
    """
    x: flattened array [x0, y0, r0, x1, y1, r1, ...]
    Returns: -sum(r) + penalty for violations
    """
    x = np.array(x, dtype=np.float64)
    # Reshape into (n, 3)
    circles = x.reshape(n, 3)
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Constraints:
    # 1. Non-overlap: dist(i,j) >= r_i + r_j
    # 2. Boundary: r_i <= cx_i, r_i <= W - cx_i, r_i <= H - cy_i, r_i <= cy_i
    #    where W, H are determined by bounding box? 
    #    Wait, the task says "variable rectangle of perimeter 4".
    #    The bounding rectangle is the minimal axis-aligned rectangle containing all circles?
    #    Or is the rectangle fixed and we pack into it?
    #    "Pack 21 circles ... in a variable rectangle of perimeter 4"
    #    "bounding rectangle has perimeter at most 4"
    #    This implies the rectangle is the bounding box of the circles.
    #    Let W = max(x+r) - min(x-r), H = max(y+r) - min(y-r).
    #    Perimeter P = 2*(W+H) <= 4 => W+H <= 2.
    
    # This is a non-convex problem if W and H are variables derived from positions.
    # However, we can fix the rectangle aspect ratio or optimize over it.
    # Let's assume the rectangle is [0, W] x [0, H] with W+H=2.
    # We can parameterize by aspect ratio a = W/H, so W = 2a/(1+a), H = 2/(1+a).
    # Or simpler: iterate over a few aspect ratios.
    
    # For a fixed rectangle W, H:
    # Constraints:
    # cx_i - r_i >= 0
    # W - cx_i - r_i >= 0
    # cy_i - r_i >= 0
    # H - cy_i - r_i >= 0
    # (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= (r_i + r_j)^2
    
    # Since the rectangle is the bounding box, we must also ensure that the min/max are actually achieved by some circles, 
    # but if we just constrain all circles to be inside [0,W]x[0,H] and maximize sum(r) subject to W+H<=2,
    # the optimal solution will naturally push circles to the boundaries to maximize radii.
    # Actually, if we fix W and H, and just require circles inside, the bounding box might be smaller than W,H.
    # But the perimeter constraint is on the bounding box. If the bounding box is smaller, the perimeter is smaller, which is fine.
    # So we can just fix W and H such that W+H=2, and pack circles in [0,W]x[0,H].
    # If they don't touch the boundaries, we could scale up? No, radii are variables.
    # Maximizing sum(r) will push them to touch boundaries.
    
    penalty = 0.0
    for i in range(n):
        # Boundary constraints
        if cx[i] - r[i] < 0: penalty += (cx[i] - r[i])**2
        if W - cx[i] - r[i] < 0: penalty += (W - cx[i] - r[i])**2
        if cy[i] - r[i] < 0: penalty += (cy[i] - r[i])**2
        if H - cy[i] - r[i] < 0: penalty += (H - cy[i] - r[i])**2
        
        # Non-overlap
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist**2:
                penalty += (min_dist**2 - dist_sq)**2
    
    return obj + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_val = -1e9
    best_circles = None
    
    # Try different aspect ratios
    # W + H = 2
    # a = W/H
    # W = 2a/(1+a), H = 2/(1+a)
    aspect_ratios = [1.0, 0.5, 2.0, 0.7, 1.4, 0.8, 1.25]
    
    for a in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2.0 * a / (1.0 + a)
        H = 2.0 / (1.0 + a)
        
        # Initial guess: grid
        side = int(np.ceil(np.sqrt(N)))
        init_x = []
        for i in range(N):
            ix = i % side
            iy = i // side
            cx = (ix + 0.5) * W / side
            cy = (iy + 0.5) * H / side
            r = min(W, H) / (2*side) * 0.9
            init_x.extend([cx, cy, r])
        
        init_x = np.array(init_x)
        
        # Use L-BFGS-B
        try:
            res = minimize(_compute_objective, init_x, args=(N, W, H), method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_val:
                # Check validity
                circles = res.x.reshape(N, 3)
                valid = True
                # Check boundaries
                for i in range(N):
                    if circles[i,0] - circles[i,2] < -1e-5: valid = False
                    if W - circles[i,0] - circles[i,2] < -1e-5: valid = False
                    if circles[i,1] - circles[i,2] < -1e-5: valid = False
                    if H - circles[i,1] - circles[i,2] < -1e-5: valid = False
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dx = circles[i,0] - circles[j,0]
                        dy = circles[i,1] - circles[j,1]
                        if dx*dx + dy*dy < (circles[i,2] + circles[j,2])**2 - 1e-5:
                            valid = False
                
                if valid:
                    val = np.sum(circles[:, 2])
                    if val > best_val:
                        best_val = val
                        best_circles = circles.copy()
        except Exception:
            pass
    
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cells = []
        for i in range(N):
            ix = i % side
            iy = i // side
            cells.append([(ix + 0.5) * W / side, (iy + 0.5) * H / side, 0.5 * min(W, H) / side - 1e-3])
        best_circles = np.array(cells)
        
    return best_circles
# EVOLVE-BLOCK-END
