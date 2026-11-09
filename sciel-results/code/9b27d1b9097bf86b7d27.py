import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def _hex_verts(cx, cy, theta_rad):
    """
    Calculate vertices of a unit regular hexagon centered at (cx, cy) 
    with rotation theta_rad.
    """
    # Base vertices for theta=0, side=1
    # Pointy top: vertices at 0, 60, 120, 180, 240, 300 degrees?
    # Standard regular hexagon side 1:
    # If flat top, vertices are at (1,0), (0.5, sqrt(3)/2), ...
    # Let's use pointy top for consistency with common lattice, or just fixed.
    # Vertices: (cos(60i), sin(60i)) for i=0..5 gives pointy top (vertex at 0 deg).
    angles = np.radians(60 * np.arange(6))
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    
    # Rotation matrix
    # x' = x*c - y*s
    # y' = x*s + y*c
    rx = base_x * c - base_y * s
    ry = base_x * s + base_y * c
    
    return np.column_stack((rx + cx, ry + cy))

def _sat_overlap_depth(h1, h2):
    """
    Calculate minimum penetration depth between two convex polygons using SAT.
    Returns 0 if disjoint.
    """
    min_depth = float('inf')
    
    # Get axes from edges of h1 and h2
    axes = []
    for h in (h1, h2):
        for i in range(6):
            p1 = h[i]
            p2 = h[(i + 1) % 6]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
            
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth

def _compute_side_length_and_penalty(params):
    """
    Compute the required outer side length and the total overlap penalty.
    params: array of shape (36,) -> [x0, y0, t0, x1, y1, t1, ...]
    Returns: (side_length, overlap_penalty)
    """
    # Outer hexagon is fixed at center (0,0), angle 0.
    # Normals for outer hexagon (pointy top alignment with inner base? No, outer is fixed).
    # Let's assume outer hexagon is pointy top (vertex at 0 deg) or flat top?
    # It doesn't matter as long as we are consistent.
    # Let's use the same base orientation for outer: vertices at 0, 60, ...
    # Normals for containment check:
    # The support function of a regular hexagon with side S centered at 0:
    # The distance from center to side (apothem) is A = S * sqrt(3) / 2.
    # The projection of any point P on the normal of a side must be <= A.
    # For a pointy-top hexagon (vertex at 0 deg), the sides are at angles 30, 90, 150, ...
    # Wait, if vertices are at 0, 60, 120..., the edges are between 0 and 60.
    # The normal to the edge between 0 and 60 deg vertices is at 30 deg.
    # So the 6 normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    outer_normals_angles = np.radians(30 + 60 * np.arange(6))
    outer_normals = np.column_stack((np.cos(outer_normals_angles), np.sin(outer_normals_angles)))
    
    max_proj = 0.0
    vertices_list = []
    
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        theta = params[3*i+2]
        
        verts = _hex_verts(cx, cy, theta)
        vertices_list.append(verts)
        
        # Project all vertices onto outer normals
        # rel = verts (since center is 0,0)
        projs = verts @ outer_normals.T
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    # max_proj is the max distance from center to any vertex projected on normals.
    # This max_proj must be <= Apothem of outer hexagon.
    # Apothem A = S * sqrt(3) / 2
    # So S = 2 * A / sqrt(3) = 2 * max_proj / sqrt(3)
    side_length = 2.0 * max_proj / math.sqrt(3.0)
    
    # Calculate overlaps
    total_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(vertices_list[i], vertices_list[j])
            if depth > 0:
                total_overlap += depth
                
    return side_length, total_overlap

def _objective(params):
    """
    Objective function for optimization.
    """
    side_length, overlap = _compute_side_length_and_penalty(params)
    
    if overlap > 1e-6:
        # Penalize overlap heavily
        return side_length + 1000.0 * overlap
    else:
        return side_length

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial guess: Hexagonal lattice
    # 3 rows x 4 cols
    # Spacing: x = 2, y = sqrt(3)
    # Offset odd rows by 1
    init_positions = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * math.sqrt(3)
            init_positions.append((x, y))
            
    # Center the lattice
    xs = [p[0] for p in init_positions]
    ys = [p[1] for p in init_positions]
    cx_avg = sum(xs) / len(xs)
    cy_avg = sum(ys) / len(ys)
    
    base_params = []
    for x, y in init_positions:
        base_params.extend([x - cx_avg, y - cy_avg, 0.0])
    base_params = np.array(base_params)
    
    best_result = None
    best_score = float('inf')
    best_valid_result = None
    
    # Try multiple random perturbations of the base lattice
    # This is more robust than DE for this specific problem within time limits
    np.random.seed(42)
    num_restarts = 10
    
    for k in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb positions and angles slightly
        noise = np.random.normal(0, 0.1, size=36)
        noise[2::3] *= 0.5 # Less angle noise
        current_params = base_params + noise
        
        try:
            result = minimize(
                _objective,
                current_params,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
            # Check validity
            side_len, overlap = _compute_side_length_and_penalty(result.x)
            if overlap < 1e-6:
                if best_valid_result is None or side_len < best_valid_result[1]:
                    best_valid_result = (result.x, side_len)
                    
        except Exception:
            pass

    # If we found a valid result, use it
    if best_valid_result is not None:
        final_params = best_valid_result[0]
        final_side = best_valid_result[1]
    else:
        # Fallback to best score (might have overlaps, but try to fix)
        if best_result is not None:
            final_params = best_result
            final_side, _ = _compute_side_length_and_penalty(final_params)
        else:
            final_params = base_params
            final_side, _ = _compute_side_length_and_penalty(final_params)
            
        # Attempt a simple separation heuristic if overlaps exist
        # This is a last resort to ensure validity
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            vertices_list = []
            for i in range(N):
                cx = final_params[3*i]
                cy = final_params[3*i+1]
                theta = final_params[3*i+2]
                vertices_list.append(_hex_verts(cx, cy, theta))
                
            has_overlap = False
            for i in range(N):
                for j in range(i + 1, N):
                    depth = _sat_overlap_depth(vertices_list[i], vertices_list[j])
                    if depth > 1e-6:
                        has_overlap = True
                        # Calculate MTV (approximate by moving centers apart)
                        # Get center of each
                        c1 = np.array([final_params[3*i], final_params[3*i+1]])
                        c2 = np.array([final_params[3*j], final_params[3*j+1]])
                        vec = c2 - c1
                        norm = np.linalg.norm(vec)
                        if norm < 1e-6:
                            vec = np.array([1,0])
                            norm = 1.0
                        else:
                            vec = vec / norm
                            
                        # Push apart by half the depth
                        push = depth * 0.5 * 0.1 # Damped
                        final_params[3*i] -= vec[0] * push
                        final_params[3*i+1] -= vec[1] * push
                        final_params[3*j] += vec[0] * push
                        final_params[3*j+1] += vec[1] * push
                        
            if not has_overlap:
                break

    # Final validation and formatting
    side_length, overlap = _compute_side_length_and_penalty(final_params)
    
    # If still overlapping, we might fail, but this is the best we can do
    # Construct the return value
    inner = []
    for i in range(N):
        x = float(final_params[3*i])
        y = float(final_params[3*i+1])
        angle_deg = float(final_params[3*i+2])
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), float(side_length), 0.0
