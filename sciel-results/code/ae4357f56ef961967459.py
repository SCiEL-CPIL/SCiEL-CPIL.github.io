"""Spectral Hexagonal Packing: optimize 12 unit hexagon positions."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    angles = base_angles + angle
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _poly_overlap_area(v1, v2):
    """Approximate overlap using vertex containment (fast check)."""
    # Simple check: if any vertex of one is inside the other
    def point_in_poly(pt, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > pt[1]) != (yj > pt[1])) and \
               (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    for pt in v1:
        if point_in_poly(pt, v2):
            return True
    for pt in v2:
        if point_in_poly(pt, v1):
            return True
    return False

def _min_pairwise_distance(centers, sides, angles):
    """Compute minimum distance between any two hexagon boundaries."""
    n = len(centers)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            # Distance between centers
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            center_dist = np.sqrt(dx*dx + dy*dy)
            # For two unit hexagons, if center distance < ~2, they may overlap
            # The minimum distance between boundaries is roughly center_dist - 2*inradius
            # inradius of unit hexagon = sqrt(3)/2
            inradius = np.sqrt(3)/2
            # This is a rough lower bound; actual depends on orientation
            approx_min = center_dist - 2*inradius
            if approx_min < min_dist:
                min_dist = approx_min
    return min_dist

def _outer_hexagon_side(centers, sides, angles, outer_center, outer_angle):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    # Transform inner hexagon vertices into outer hexagon's coordinate frame
    outer_angle_rad = np.radians(outer_angle)
    cos_a = np.cos(outer_angle_rad)
    sin_a = np.sin(outer_angle_rad)
    
    max_extent = 0
    for i in range(len(centers)):
        verts = _hexagon_vertices(centers[i, 0], centers[i, 1], sides[i], angles[i])
        # Translate to outer center
        verts = verts - np.array(outer_center)
        # Rotate by -outer_angle
        rotated = np.column_stack([
            verts[:, 0]*cos_a + verts[:, 1]*sin_a,
            -verts[:, 0]*sin_a + verts[:, 1]*cos_a
        ])
        # For a hexagon with flat sides, the constraint is on the support function
        # in 6 directions (0, 60, 120, 180, 240, 300 degrees)
        for k in range(6):
            angle = k * np.pi / 3
            # Project onto direction (cos(angle), sin(angle))
            proj = rotated[:, 0]*np.cos(angle) + rotated[:, 1]*np.sin(angle)
            extent = np.max(proj)
            if extent > max_extent:
                max_extent = extent
    # The outer hexagon side length s satisfies: support = s * cos(pi/6) = s * sqrt(3)/2
    # So s = max_extent / (sqrt(3)/2) = 2*max_extent/sqrt(3)
    return 2.0 * max_extent / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial guess: hexagonal cluster pattern
    # 12 hexagons in a pattern: center + 6 around + 5 more
    # Let's use a 3x4 grid-like arrangement optimized
    
    # Start with a reasonable initial configuration
    # Hexagonal lattice positions
    sqrt3 = np.sqrt(3)
    
    # Try a 2-3-3-2-2 pattern or similar
    # Let's use a compact arrangement
    init_centers = []
    # Row 0: 2 hexagons
    init_centers.append((-sqrt3/2, 0))
    init_centers.append((sqrt3/2, 0))
    # Row 1: 3 hexagons
    init_centers.append((-sqrt3, -1.5))
    init_centers.append((0, -1.5))
    init_centers.append((sqrt3, -1.5))
    # Row 2: 3 hexagons
    init_centers.append((-sqrt3/2, -3.0))
    init_centers.append((sqrt3/2, -3.0))
    init_centers.append((0, -3.0))
    # Row 3: 2 hexagons
    init_centers.append((-sqrt3, -4.5))
    init_centers.append((sqrt3, -4.5))
    # Row 4: 2 hexagons
    init_centers.append((-sqrt3/2, -6.0))
    init_centers.append((sqrt3/2, -6.0))
    
    init_centers = np.array(init_centers)
    init_angles = np.zeros(N)
    
    # Optimization variables: centers (2*N) + angles (N) + outer_center (2) + outer_angle (1) = 3N + 3
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10  # penalty for timeout
        
        # Unpack
        cx = params[0:N]
        cy = params[N:2*N]
        angles = params[2*N:3*N] * 180.0  # convert to degrees
        outer_cx = params[3*N]
        outer_cy = params[3*N+1]
        outer_angle = params[3*N+2] * 180.0
        
        centers = np.column_stack([cx, cy])
        sides = np.ones(N)
        
        # Check pairwise overlaps (soft penalty)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[j] - cx[i]
                dy = cy[j] - cy[i]
                dist = np.sqrt(dx*dx + dy*dy)
                # Minimum center distance for non-overlap of unit hexagons
                # Depends on relative orientation, but ~sqrt(3) is a good lower bound
                min_dist = 1.7  # slightly less than sqrt(3) ~ 1.732
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
                # Also check actual overlap
                v1 = _hexagon_vertices(cx[i], cy[i], 1.0, angles[i])
                v2 = _hexagon_vertices(cx[j], cy[j], 1.0, angles[j])
                if _poly_overlap_area(v1, v2):
                    penalty += 10.0
        
        # Outer hexagon side length
        outer_side = _outer_hexagon_side(centers, sides, angles, (outer_cx, outer_cy), outer_angle)
        
        return outer_side + 5.0 * penalty
    
    def gradient(params):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params_p = params.copy()
            params_p[i] += eps
            fp = objective(params_p)
            params_p[i] -= 2*eps
            fm = objective(params_p)
            params_p[i] += eps
            grad[i] = (fp - fm) / (2*eps)
        return grad
    
    # Initial parameters
    init_params = np.concatenate([init_centers.flatten(), init_angles, [0.0, 0.0, 0.0]])
    
    best_score = float('inf')
    best_params = init_params.copy()
    
    # Try multiple initial configurations
    n_restarts = 5
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            break
        
        if r == 0:
            params0 = init_params.copy()
        else:
            # Perturb initial configuration
            np.random.seed(r)
            params0 = init_params + np.random.randn(len(init_params)) * 0.5
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return best found so far
        pass
    
    # Extract best solution
    cx = best_params[0:N]
    cy = best_params[N:2*N]
    angles = best_params[2*N:3*N] * 180.0
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N+1]
    outer_angle = best_params[3*N+2] * 180.0
    
    inner = [[float(cx[i]), float(cy[i]), float(angles[i])] for i in range(N)]
    outer_center = (float(outer_cx), float(outer_cy))
    sides = np.ones(N)
    centers = np.column_stack([cx, cy])
    outer_side = _outer_hexagon_side(centers, sides, angles, outer_center, outer_angle)
    
    return inner, outer_center, float(outer_side), float(outer_angle)
# EVOLVE-BLOCK-END

