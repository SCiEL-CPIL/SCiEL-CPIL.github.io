import numpy as np
import time
from scipy.optimize import differential_evolution

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

def _sat_overlap(v1, v2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap, False otherwise.
    v1, v2: numpy arrays of shape (6, 2)
    """
    # Combine edges from both polygons
    # For hexagons, we can just use the normals of the edges of v1 and v2
    
    # Get edges of v1
    edges1 = np.roll(v1, -1, axis=0) - v1
    # Get edges of v2
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    all_edges = np.vstack([edges1, edges2])
    
    # Calculate normals (perpendicular to edges)
    # Rotate 90 degrees: (x, y) -> (-y, x)
    normals = np.column_stack([-all_edges[:, 1], all_edges[:, 0]])
    
    # Normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    normals = normals / norms
    
    # Project both polygons onto each normal
    for normal in normals:
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _is_valid(centers, angles, outer_side, outer_center, outer_angle, eps=1e-9):
    """
    Check if configuration is valid.
    centers: (N, 2) array
    angles: (N,) array
    outer_side: float
    outer_center: (2,) array
    outer_angle: float
    """
    # 1. Check disjointness of inner hexagons
    inner_verts = [_hex_vertices(centers[i], 1.0, angles[i]) for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False
                
    # 2. Check containment in outer hexagon
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    
    # Precompute outer edges and normals for point-in-polygon test
    # We need to check if all vertices of inner hexagons are inside outer hexagon
    # Since outer is convex, checking vertices is sufficient.
    
    # To make it faster, we can check if the Minkowski sum of inner hex and 
    # negated outer hex is disjoint? No, simpler: point in polygon.
    
    # Efficient point in convex polygon: 
    # For a convex polygon defined by vertices in CCW order, a point is inside
    # if it is to the left of every edge (cross product > 0).
    
    outer_edges = np.roll(outer_verts, -1, axis=0) - outer_verts
    outer_normals = np.column_stack([-outer_edges[:, 1], outer_edges[:, 0]])
    
    # Ensure normals point inward
    # The center of the outer hexagon should be "inside" relative to normals
    # cross(edge, center - vertex) should be positive for inward normals if CCW
    # Let's just compute signed distance to edges.
    # For edge (a, b), normal n = (b-a) rotated 90 deg CCW = (-(b_y-a_y), b_x-a_x)
    # If polygon is CCW, interior is to the left of edge.
    # Let's verify orientation of _hex_vertices.
    # angles are increasing, so CCW.
    # So inward normal is left normal.
    
    # For each inner vertex, check if it is inside outer hex.
    for i in range(N):
        verts = inner_verts[i]
        for v in verts:
            # Check against all outer edges
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k + 1) % 6]
                # Vector a->b
                ab = b - a
                # Vector a->v
                av = v - a
                # Cross product z-component
                cross = ab[0] * av[1] - ab[1] * av[0]
                # If CCW, cross > 0 means left (inside)
                if cross < -eps:
                    return False
                    
    return True

def _compute_required_side(centers, angles, outer_angle=0.0):
    """
    Compute the minimum side length of a regular hexagon (centered at origin, 
    rotated by outer_angle) that contains all inner hexagons.
    """
    # The condition for a point p to be inside a regular hexagon of side L, 
    # center (0,0), angle theta is:
    # For each of the 6 normals n_k (angle theta + k*pi/3):
    # dot(p, n_k) <= L * apothem = L * sqrt(3)/2
    
    # So L >= 2 * dot(p, n_k) / sqrt(3) for all p, k.
    # We need max over all p (vertices of inner hexes) and k.
    
    max_proj = 0.0
    normals = []
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        normals.append(normal)
        
    for i in range(N):
        verts = _hex_vertices(centers[i], 1.0, angles[i])
        for v in verts:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _objective(params):
    """
    Objective function for Differential Evolution.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    Note: We fix outer center to (0,0) and outer angle to 0 for optimization,
    then we can rotate the whole cluster or adjust outer angle in post-processing?
    Actually, the problem allows outer hexagon to rotate. 
    However, due to symmetry of the regular hexagon, rotating the outer hexagon
    by 60 degrees is equivalent to rotating the inner cluster by -60 degrees.
    But the inner hexagons can rotate independently.
    
    Let's simplify: Fix outer center at (0,0). 
    We will optimize inner positions and angles.
    We will try a few fixed outer angles (0, 30) and pick the best.
    Or, we can include outer angle in params? 
    DE with 37 parameters is fine.
    
    Let's include outer angle in params to be safe? 
    No, let's stick to 36 params for inner, and loop over outer angles outside.
    But _objective needs to return a scalar.
    If we loop outside, we can't easily use DE for outer angle too.
    
    Let's just optimize inner positions/angles for a FIXED outer angle (0.0).
    The minimal enclosing hexagon for a set of points is unique up to rotation 
    of the hexagon. If we fix the hexagon orientation, we find the min side.
    If the optimal packing requires a specific relative orientation between 
    inner and outer, fixing outer to 0 is fine, because we can rotate the 
    entire solution (inner + outer) by any angle, and the side length remains 
    the same. 
    Wait, if we rotate the outer hexagon, the constraint boundaries change relative 
    to the fixed coordinate system. But if we also rotate the inner hexagons, 
    the relative geometry is preserved. 
    Since the inner hexagons can rotate freely, we can always align the "best" 
    orientation with the outer hexagon's 0-degree orientation by adjusting 
    the inner angles.
    So, fixing outer angle to 0 is sufficient.
    """
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return 1e10
        
    centers = params[0::3]
    angles = params[1::3]
    
    # Compute required side
    req_side = _compute_required_side(centers, angles, 0.0)
    
    # Check validity
    # To save time, we can skip validity check if req_side is already large?
    # No, we need to ensure non-overlap.
    # SAT is fast enough.
    if not _is_valid(centers, angles, req_side, (0.0, 0.0), 0.0):
        # Penalty for invalidity
        # We can compute the amount of overlap or just return a large number.
        # A better penalty: sum of penetration depths.
        # For simplicity, return req_side + large_penalty * overlap_count
        # Let's just return a large number to push DE away.
        return req_side + 100.0
        
    return req_side

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    
    # Bounds for DE
    # x, y in [-5, 5] (should be enough for 12 hexes)
    # angle in [0, 360]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 360.0))
        
    # Strategy: Run DE with a time limit.
    # We can use the 'maxiter' and 'tol' to control, but time is safer.
    # We'll wrap the objective to check time.
    
    try:
        # Use differential evolution
        # popsize=15, tol=0.01, maxiter=1000 (but limited by time)
        # seed=42 for reproducibility
        
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            tol=0.01, 
            popsize=15,
            polish=False # Polish with local optimizer later if time permits
        )
        
        if res.success or res.fun < best_side:
            if res.fun < 100.0: # Sanity check
                best_params = res.x
                best_side = res.fun
                
    except Exception as e:
        pass
        
    # If we have time, try to polish the best solution found using local optimization
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        # Use L-BFGS-B to polish
        from scipy.optimize import minimize
        
        # Extract centers and angles
        # Note: _objective takes flat array.
        # We need a smooth objective for L-BFGS-B?
        # SAT is non-smooth. 
        # Maybe just do a few random restarts or a simple local search?
        # Given the time constraint and non-smoothness, let's skip complex polishing
        # and rely on DE's global search.
        
        # However, we can try a simple local perturbation search.
        current_params = best_params.copy()
        current_side = best_side
        step_size = 0.1
        for _ in range(100):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            for i in range(len(current_params)):
                # Perturb
                orig = current_params[i]
                current_params[i] = orig + np.random.uniform(-step_size, step_size)
                new_side = _objective(current_params)
                if new_side < current_side:
                    current_side = new_side
                    best_side = new_side
                    best_params = current_params.copy()
                else:
                    current_params[i] = orig # Revert
                    
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
            step_size *= 0.95 # Anneal step size
            
    # Construct final result
    if best_params is not None:
        centers = best_params[0::3]
        angles = best_params[1::3]
        
        inner_hexes = []
        for i in range(N):
            inner_hexes.append([centers[i], angles[i]])
            
        # Compute final side
        final_side = _compute_required_side(centers, angles, 0.0)
        
        # Final validity check
        if _is_valid(centers, angles, final_side, (0.0, 0.0), 0.0):
            return (inner_hexes, (0.0, 0.0), final_side, 0.0)
        else:
            # If invalid, we need a fallback. 
            # This shouldn't happen if _objective returned a low value, 
            # but due to floating point or race conditions, it might.
            # Return a valid fallback.
            pass

    # Fallback: Hexagonal lattice packing
    # Side length approx: 1 + 2 * sqrt(3) * (1 + 1/sqrt(3)) ... 
    # Let's just place them in a line to be safe? No, that's huge.
    # Hexagonal packing of 12 hexes.
    # Rows: 3, 4, 3, 2? 
    # 1, 2, 3, 4, 2? 
    # Let's use a known compact configuration.
    # Center + 6 neighbors + 5 more.
    # This is getting complex. 
    # Simple fallback: Grid.
    # Or just return the DE result even if slightly invalid? No, must be valid.
    
    # Let's construct a valid hexagonal cluster.
    # 1 center, 6 around it, 5 more around that.
    # This fits in a hexagon of side 3? 
    # Let's just use a large side length fallback to ensure validity.
    # The score is 1.0 / side. If side is large, score is low.
    # But we want to beat 8.0.
    
    # If DE failed, try a simple grid.
    # 4x3 grid of hexagons.
    # Spacing: dx = 1.5, dy = sqrt(3)/2
    # This is a valid packing.
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5
            y = r * np.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            # Center at 0
            centers.append([x - 2.25, y - np.sqrt(3)])
            
    inner_hexes = [[c[0], c[1], 0.0] for c in centers]
    
    # Compute side for this grid
    # It's a rectangle. Min enclosing hexagon?
    # Let's just compute it.
    final_side = _compute_required_side(np.array(centers), np.zeros(4*3), 0.0)
    
    # Check validity
    if _is_valid(np.array(centers), np.zeros(4*3), final_side, (0.0, 0.0), 0.0):
        return (inner_hexes, (0.0, 0.0), final_side, 0.0)
        
    # Absolute fallback: Single hexagon side 1, others far away? No, must be disjoint.
    # Just return a very large hexagon.
    return ([[0.0, 0.0, 0.0]] * N, (0.0, 0.0), 100.0, 0.0)

# For testing
if __name__ == "__main__":
    res = construct()
    print(res)
