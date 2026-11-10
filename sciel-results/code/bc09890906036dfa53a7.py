# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    """
    ov_cx, ov_cy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    
    # The 6 outward normals to the sides of the outer hexagon
    # For a hexagon with vertices at angles (a + 60*i), the side normals point at (a + 30 + 60*i)
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each inner hexagon, find the maximum projection onto each normal
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        # Translate to outer center frame
        hv_rel = hv - np.array([ov_cx, ov_cy])
        projections = hv_rel @ normals.T  # shape (6, 6)
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    # The distance from center to each side is s * sqrt(3)/2
    # So we need s * sqrt(3)/2 >= max_projection for all 6 directions
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    
    return s_min


def _check_separation(inner):
    """
    Check if all pairs of inner hexagons are separated (non-overlapping) using SAT.
    Returns True if all pairs are separated.
    """
    n = len(inner)
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            # Get all edge normals from both hexagons
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > 1e-6 or gap2 > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _objective(params, outer_center):
    """
    Objective function: minimize the containing hexagon side length.
    params: 25 values = 12 * (x, y) for the 12 inner hexagons + 1 outer angle (radians)
    """
    inner = np.zeros((12, 3))
    inner[:, 0] = params[0:24:2]
    inner[:, 1] = params[1:24:2]
    inner[:, 2] = 0.0  # Fixed angles for inner hexagons
    
    outer_angle_deg = np.rad2deg(params[24])
    
    # Check separation
    if not _check_separation(inner):
        # Penalize overlaps heavily
        return 1000.0
    
    s = _min_containing_hex_side(inner, outer_center, outer_angle_deg)
    return s


def _get_lattice_config(coords):
    """
    Convert axial coordinates to Cartesian for unit hexagons.
    Axial (q, r) -> Cartesian (x, y)
    x = sqrt(3) * (q + r/2)
    y = 3/2 * r
    """
    pts = []
    for q, r in coords:
        x = np.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        pts.append([x, y])
    return np.array(pts)


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s limit
    
    outer_center = np.array([0.0, 0.0])
    
    # Define a few lattice topologies for 12 hexagons
    # Topology 1: Center + Ring 1 + 5 from Ring 2 (compact cluster)
    # Axial coords (q, r)
    top1 = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2)
    ]
    
    # Topology 2: A slightly different shape, e.g. more rectangular
    top2 = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
        (2, 0), (2, -1), (3, -1), (3, -2), (2, -2)
    ]
    
    # Topology 3: Another variation
    top3 = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
        (2, 0), (1, 1), (0, 2), (-1, 1), (-1, 2) # Wait, (1,1) is dist 2 from (0,0)? 
        # Let's stick to valid lattice neighbors.
        # Let's just use Topology 1 and 2.
    ]
    
    topologies = [top1, top2]
    
    best_params = None
    best_s = float('inf')
    
    for top in topologies:
        if time.time() - start_time > deadline - 100.0:
            break
            
        init_pts = _get_lattice_config(top)
        
        # Initial parameters: x, y for each, then angle (0.0)
        init_params = np.zeros(25)
        for i, (x, y) in enumerate(init_pts):
            init_params[2*i] = x
            init_params[2*i+1] = y
        init_params[24] = 0.0 # Initial outer angle
        
        # We can try a few initial outer angles to help the optimizer
        # Since the lattice has symmetry, rotating by 0, 15, 30 degrees might help
        # But the optimizer should find it. Let's just run once with 0.
        
        try:
            result = minimize(
                _objective,
                init_params,
                args=(outer_center,),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_s and result.fun < 100.0: # 100 is a bad penalty
                best_params = result.x
                best_s = result.fun
        except Exception as e:
            pass
            
    # If optimization failed or didn't find anything, fallback to a valid static configuration
    if best_params is None or best_s > 100.0:
        # Use Topology 1 static
        init_pts = _get_lattice_config(top1)
        best_params = np.zeros(25)
        for i, (x, y) in enumerate(init_pts):
            best_params[2*i] = x
            best_params[2*i+1] = y
        best_params[24] = 0.0
        best_s = _objective(best_params, outer_center)
        
    # Construct final inner hexagons
    inner = np.zeros((12, 3))
    inner[:, 0] = best_params[0:24:2]
    inner[:, 1] = best_params[1:24:2]
    inner[:, 2] = 0.0
    
    outer_angle_deg = np.rad2deg(best_params[24])
    
    # Compute final s with a small margin
    s_final = _min_containing_hex_side(inner, outer_center, outer_angle_deg) + 1e-4
    
    # Verify validity one more time
    if not _check_separation(inner):
        # Fallback to initial lattice configuration
        init_pts = _get_lattice_config(top1)
        inner = np.zeros((12, 3))
        for i, (x, y) in enumerate(init_pts):
            inner[i, 0] = x
            inner[i, 1] = y
            inner[i, 2] = 0.0
        outer_angle_deg = 0.0
        s_final = _min_containing_hex_side(inner, outer_center, outer_angle_deg) + 1e-4

    outer = np.array([0.0, 0.0, outer_angle_deg])
    
    return inner, outer, s_final


# EVOLVE-BLOCK-END
