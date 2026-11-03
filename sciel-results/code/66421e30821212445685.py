"""Packing 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# Precompute unit hexagon vertices (flat-top orientation, side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
_UNIT_HEX_VERTS = np.array([
    [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)
])

def _hex_vertices(center, angle_deg):
    """Get vertices of a unit hexagon at given center and rotation."""
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return center + _UNIT_HEX_VERTS @ rot.T

def _outer_hex_vertices(center, side, angle_deg):
    """Get vertices of outer hexagon. Side length = distance between parallel sides / sqrt(3) * 2... 
    Actually for regular hexagon with side length s, the distance from center to vertex is s,
    and distance from center to side (apothem) is s*sqrt(3)/2."""
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    # Vertices at distance = side from center
    verts = side * _UNIT_HEX_VERTS @ rot.T
    return center + verts

def _support_function(point, direction):
    """Dot product of point with direction."""
    return np.dot(point, direction)

def _min_outer_side_for_config(inner_centers, inner_angles, outer_center, outer_angle):
    """
    Given inner hexagon positions and outer hexagon orientation,
    find the minimum side length of outer hexagon (centered at outer_center)
    that contains all inner hexagons.
    
    For a convex polygon (outer hexagon) with center c, orientation phi, side length s,
    a point p is inside if for each of the 6 outward normals n_i:
    dot(p - c, n_i) <= s * sqrt(3) / 2  (the apothem)
    
    The 6 outward normals for a hexagon with vertices at angles 0,60,...,300 are
    at angles 30, 90, 150, 210, 270, 330 (perpendicular to each side).
    """
    outer_theta = np.radians(outer_angle)
    # Outward normals: perpendicular to sides, pointing outward
    # For a flat-top hexagon (vertices at 0,60,...), sides are between consecutive vertices
    # Normal to side between vertex at angle a and a+60 is at angle a+30
    normal_angles = [np.radians(a + 30) for a in range(0, 360, 60)]
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    
    # For each inner hexagon, compute max support in each normal direction
    # The constraint is: max over all inner vertices of dot(v - outer_center, n_i) <= s * sqrt(3)/2
    # So s >= max_i dot(v - outer_center, n_i) / (sqrt(3)/2) for all i
    
    max_required = 0.0
    for ic, ia in zip(inner_centers, inner_angles):
        verts = _hex_vertices(ic, ia)
        # verts shape (6, 2)
        # For each normal, compute max dot product with (verts - outer_center)
        rel = verts - outer_center  # (6, 2)
        # dot products: (6, 6)
        dots = rel @ normals.T  # (6, 6)
        max_dots = np.max(dots, axis=0)  # (6,)
        max_required = max(max_required, np.max(max_dots))
    
    # s * sqrt(3)/2 >= max_required
    # s >= max_required / (sqrt(3)/2) = 2 * max_required / sqrt(3)
    return 2.0 * max_required / np.sqrt(3.0)

def _objective(params, outer_angle):
    """
    params: array of length 3*N, [x1, y1, a1, x2, y2, a2, ...]
    Returns the outer side length needed.
    We fix outer_center at origin and outer_angle as given.
    """
    inner_centers = []
    inner_angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_centers.append(np.array([x, y]))
        inner_angles.append(a)
    
    return _min_outer_side_for_config(inner_centers, inner_angles, np.array([0.0, 0.0]), outer_angle)

def construct():
    deadline = time.time() + 110.0  # Leave safety margin
    
    best_score = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    # Try several outer hexagon orientations
    outer_angles = [0.0, 15.0, 30.0, 45.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline:
            break
        
        # Initial guess: hexagonal lattice arrangement
        # For 12 hexagons, a 3x4 grid or similar
        # Hexagonal lattice spacing: for side-1 hexagons, center-to-center distance = sqrt(3) for adjacent
        dx = np.sqrt(3.0)
        dy = 1.5
        
        # Try a few initial configurations
        initial_configs = []
        
        # Config 1: 3 rows of 4
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx/2
                y = row * dy
                initial_configs.append([x, y, 0.0])
        
        # Config 2: 4 rows of 3
        for row in range(4):
            for col in range(3):
                x = col * dx + (row % 2) * dx/2
                y = row * dy
                initial_configs.append([x, y, 0.0])
        
        # Config 3: 2 rows of 6
        for row in range(2):
            for col in range(6):
                x = col * dx + (row % 2) * dx/2
                y = row * dy
                initial_configs.append([x, y, 0.0])
        
        # Center each config
        for config in initial_configs:
            if time.time() > deadline:
                break
            
            # Center the configuration
            xs = [c[0] for c in config]
            ys = [c[1] for c in config]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            config = [[c[0] - cx, c[1] - cy, c[2]] for c in config]
            
            params0 = np.array([v for c in config for v in c])
            
            # Use Nelder-Mead
            try:
                result = minimize(
                    _objective,
                    params0,
                    args=(outer_angle,),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
                    best_outer_angle = outer_angle
            except Exception:
                pass
            
            # Also try with some random perturbations
            for trial in range(3):
                if time.time() > deadline:
                    break
                
                perturbed = params0 + np.random.randn(len(params0)) * 0.5
                try:
                    result = minimize(
                        _objective,
                        perturbed,
                        args=(outer_angle,),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                    )
                    
                    if result.fun < best_score:
                        best_score = result.fun
                        best_params = result.x.copy()
                        best_outer_angle = outer_angle
                except Exception:
                    pass
    
    if best_params is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Construct the inner hexagons
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    return inner, (0.0, 0.0), best_score, best_outer_angle
