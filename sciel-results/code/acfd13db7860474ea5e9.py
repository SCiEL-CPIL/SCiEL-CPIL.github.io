import numpy as np
import time
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _compute_required_side(centers, angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute minimum outer side length to contain all hexagons.
    """
    max_proj = 0.0
    oc = np.array(outer_center)
    oa = np.deg2rad(outer_angle)
    
    # Normals to the 6 sides of the outer hexagon
    for k in range(6):
        angle = oa + k * np.pi / 3
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v - oc, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _is_valid(centers, angles, side, outer_center=(0.0, 0.0), outer_angle=0.0, tol=1e-8):
    """
    Check if the configuration is valid:
    1. All hexagons are inside the outer hexagon.
    2. No hexagons overlap.
    """
    # 1. Containment
    for i in range(N):
        v1 = _hex_vertices(centers[i], 1.0, angles[i])
        for v in v1:
            inside = True
            for k in range(6):
                angle = np.deg2rad(outer_angle) + k * np.pi / 3
                normal = np.array([np.cos(angle), np.sin(angle)])
                apothem = side * np.sqrt(3) / 2
                proj = np.dot(v - np.array(outer_center), normal)
                if proj > apothem + tol:
                    inside = False
                    break
            if not inside:
                return False

    # 2. Overlap (SAT)
    for i in range(N):
        v1 = _hex_vertices(centers[i], 1.0, angles[i])
        for j in range(i + 1, N):
            v2 = _hex_vertices(centers[j], 1.0, angles[j])
            
            # Quick bounding box check
            if np.max(v1[:,0]) < np.min(v2[:,0]) - tol or np.max(v2[:,0]) < np.min(v1[:,0]) - tol:
                continue
            if np.max(v1[:,1]) < np.min(v2[:,1]) - tol or np.max(v2[:,1]) < np.min(v1[:,1]) - tol:
                continue
            
            # SAT Check
            separated = False
            for k in range(6):
                edge1 = v1[(k+1) % 6] - v1[k]
                normal1 = np.array([-edge1[1], edge1[0]])
                norm1 = np.linalg.norm(normal1)
                if norm1 < 1e-12: continue
                normal1 /= norm1
                
                proj1 = np.dot(v1, normal1)
                proj2 = np.dot(v2, normal1)
                
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    separated = True
                    break
                    
            if not separated:
                for k in range(6):
                    edge2 = v2[(k+1) % 6] - v2[k]
                    normal2 = np.array([-edge2[1], edge2[0]])
                    norm2 = np.linalg.norm(normal2)
                    if norm2 < 1e-12: continue
                    normal2 /= norm2
                    
                    proj1 = np.dot(v1, normal2)
                    proj2 = np.dot(v2, normal2)
                    
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        separated = True
                        break
                        
            if not separated:
                return False
                
    return True

def _generate_lattice_config(spacing, angle_offset=0.0):
    """
    Generate 12 hexagons on a triangular lattice.
    Spacing is the distance between adjacent centers.
    For unit hexagons, minimum spacing to touch is sqrt(3).
    We place them in a compact cluster.
    """
    centers = []
    # Basis vectors for triangular lattice
    # Standard orientation: one axis horizontal
    # v1 = (spacing, 0)
    # v2 = (spacing/2, spacing * sqrt(3)/2)
    
    # We want a cluster of 12. 
    # A hexagonal number 12 doesn't exist (1, 7, 19...).
    # So we take a patch from a larger hexagon.
    # Center (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: We need 5 more to make 12.
    # In a hex lattice, Ring 2 has 12 positions. We pick 5.
    # To be symmetric, we might pick 5 out of 6 in a semi-ring? Or just take a 3x4 rectangle?
    # Let's stick to the hexagonal patch idea: 1 + 6 + 5.
    # The 5 should be placed to minimize the bounding box.
    
    centers.append([0, 0])
    
    # Ring 1
    for i in range(6):
        a = i * np.pi / 3
        centers.append([spacing * np.cos(a), spacing * np.sin(a)])
        
    # Ring 2 (12 spots). We need 5.
    # The spots are at distance 2*spacing.
    # Angles: 0, 30, 60, ... 330.
    # To minimize the extent, we should probably fill a contiguous arc or balance it.
    # Let's try filling 5 contiguous spots starting from angle 0? Or centered?
    # Actually, a 3x4 rectangular grid on a triangular lattice is also a valid "lattice" configuration.
    # Let's generate a few different lattice patterns.
    
    # Pattern A: 1 + 6 + 5 (contiguous arc)
    # This is handled by the main function trying different offsets.
    # For now, return the 7 from above, and let the caller decide the rest?
    # No, this function should return a full config.
    # Let's implement a specific pattern: "3 rows of 4" in a skewed grid.
    
    # Reset centers
    centers = []
    # Triangular lattice coordinates (i, j)
    # x = i * spacing + j * spacing * 0.5
    # y = j * spacing * sqrt(3) / 2
    
    # We want 12 points.
    # A 3x4 block: i in [0, 2], j in [0, 3]
    for i in range(3):
        for j in range(4):
            x = i * spacing + j * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            # Center the cluster
            # We'll center it later.
            centers.append([x, y])
            
    # Center the cluster
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    for c in centers:
        c[0] -= cx
        c[1] -= cy
        
    return np.array(centers)

def _objective_lattice(params):
    """
    params: [spacing, outer_angle_deg, outer_cx, outer_cy]
    """
    spacing = params[0]
    outer_angle = params[1]
    outer_cx = params[2]
    outer_cy = params[3]
    
    centers = _generate_lattice_config(spacing)
    angles = np.zeros(N) # Fixed orientation for lattice
    
    # Compute required side
    side = _compute_required_side(centers, angles, (outer_cx, outer_cy), outer_angle)
    
    # Check validity (overlap is guaranteed by lattice if spacing >= sqrt(3))
    # But we need to ensure spacing is large enough.
    if spacing < np.sqrt(3) - 1e-6:
        return 1e10
        
    return side

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_spacing = np.sqrt(3)
    
    # 1. Grid Search over lattice spacing and outer angle
    # Spacing range: sqrt(3) to sqrt(3) + 0.5
    # Outer angle range: 0 to 60 (symmetry)
    
    spacing_vals = np.linspace(np.sqrt(3), np.sqrt(3) + 0.2, 20)
    angle_vals = np.linspace(0, 60, 20)
    
    for s in spacing_vals:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        for a in angle_vals:
            # Try outer center at (0,0)
            params = np.array([s, a, 0.0, 0.0])
            side = _objective_lattice(params)
            
            if side < best_side:
                # Verify validity
                centers = _generate_lattice_config(s)
                angles = np.zeros(N)
                if _is_valid(centers, angles, side, (0.0, 0.0), a):
                    best_side = side
                    best_inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = a
                    best_spacing = s

    # 2. Local Refinement around best found
    if best_inner is not None:
        if time.time() - start_time < TIME_LIMIT - 10:
            # Use Nelder-Mead to refine spacing, angle, center
            # We need a different objective that takes the full params
            def _obj_refine(params):
                s, a, cx, cy = params
                centers = _generate_lattice_config(s)
                angles = np.zeros(N)
                side = _compute_required_side(centers, angles, (cx, cy), a)
                if not _is_valid(centers, angles, side, (cx, cy), a):
                    return side + 100.0
                return side

            x0 = np.array([best_spacing, best_outer_angle, best_outer_center[0], best_outer_center[1]])
            try:
                result = minimize(_obj_refine, x0, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.fun < best_side:
                    s, a, cx, cy = result.x
                    centers = _generate_lattice_config(s)
                    angles = np.zeros(N)
                    if _is_valid(centers, angles, result.fun, (cx, cy), a):
                        best_side = result.fun
                        best_inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
                        best_outer_center = (cx, cy)
                        best_outer_angle = a
            except Exception:
                pass

    # Fallback if nothing found (should not happen)
    if best_inner is None:
        # Return a valid, albeit large, configuration
        centers = _generate_lattice_config(np.sqrt(3) + 1)
        angles = np.zeros(N)
        side = _compute_required_side(centers, angles, (0,0), 0)
        return (
            [[centers[i][0], centers[i][1], angles[i]] for i in range(N)],
            (0.0, 0.0),
            side,
            0.0
        )

    return (best_inner, best_outer_center, best_side, best_outer_angle)
