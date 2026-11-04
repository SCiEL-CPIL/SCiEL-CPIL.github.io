import numpy as np
import time
import itertools

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _get_axes(hex_verts):
    """Get the 6 edge normals (axes) of a hexagon."""
    axes = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        axes.append(normal)
    return np.array(axes)

def _sat_overlap(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap (interiors intersect).
    """
    axes = np.vstack([_get_axes(v1), _get_axes(v2)])
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
            
    return True

def _point_in_hex(p, center, side, angle_deg, eps=1e-9):
    """Check if point p is inside or on the boundary of the hexagon."""
    verts = _hex_vertices(center, side, angle_deg)
    inside = True
    for k in range(6):
        a = verts[k]
        b = verts[(k+1)%6]
        # Cross product of (b-a) and (p-a)
        # If all cross products have the same sign (or zero), p is inside
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross < -eps:
            inside = False
            break
    return inside

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment
    for iv in inner_verts:
        for v in iv:
            if not _point_in_hex(v, outer_center, outer_side, outer_angle, eps):
                return False, "outside"
                
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
                
    return True, "valid"

def _min_side_for_cluster(cluster_verts, outer_angle_deg, outer_center=(0,0)):
    """
    Given a set of inner hexagon vertices (centered at 0,0, rotated to match cluster),
    find the minimum side length of an outer hexagon (centered at outer_center, rotated by outer_angle_deg)
    that contains all vertices.
    
    Since the inner hexagons are rigid and we are just scaling the outer one (and it's convex),
    the minimum side length S is determined by the maximum 'distance' of any inner vertex 
    from the center in the direction of the outer hexagon's apothems and vertices?
    
    Actually, for a regular hexagon defined by center C, side S, angle A:
    It is the intersection of 6 half-planes.
    The distance from C to each edge is S * sqrt(3) / 2.
    The direction of the normal to edge k is A + k * 60 degrees.
    
    So, for a point P to be inside, its projection onto each normal n_k must be <= S * sqrt(3) / 2.
    i.e., dot(P - C, n_k) <= S * sqrt(3) / 2.
    
    Therefore, S must be >= dot(P - C, n_k) * 2 / sqrt(3) for all P and all k.
    So S_min = max_{P, k} ( dot(P - C, n_k) * 2 / sqrt(3) ).
    Note: We must also ensure S is large enough to cover the vertices in the 'corner' directions?
    The condition dot(P-C, n_k) <= Apothem is necessary and sufficient for a convex polygon 
    defined by those 6 half-planes.
    """
    theta = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        a = theta + k * np.pi / 3
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for v in cluster_verts:
        # v is relative to (0,0). outer_center is (0,0) WLOG.
        # proj = dot(v, n)
        projs = np.dot(normals, v)
        max_proj = max(max_proj, np.max(projs))
        
    # S * sqrt(3) / 2 >= max_proj
    # S >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    # 1. Define the rigid cluster of 12 hexagons.
    # Use a honeycomb lattice.
    # Lattice vectors for hexagonal packing of unit hexagons:
    # Distance between centers = sqrt(3).
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Generate a pool of candidate centers around the origin.
    # We want 12 points that form a compact shape.
    # A 3x4 rectangular block in the hex lattice is a good candidate.
    # Or a hexagonal shape of radius 1 (center + 6 + 5? No, 1+6+12=19 is too many. 1+6=7 is too few.
    # A "hexagon" of side 2 has 1+6+12 = 19 hexagons.
    # A "hexagon" of side 1 has 7 hexagons.
    # We need 12.
    # Let's take the 7 central ones, and add 5 from the next ring.
    # Or simply take a 3x4 grid in the lattice coordinates.
    
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            lattice_points.append(p)
            
    lattice_points = np.array(lattice_points)
    
    # Select 12 points closest to origin? 
    # This might not be the most compact shape.
    # Let's try to pick a specific compact shape.
    # Shape: 3 rows of 4?
    # In hex lattice, a "rectangle" of width w and height h (in lattice steps) 
    # has area roughly w*h.
    # Let's just pick the 12 closest to the origin. This usually yields a compact blob.
    
    dists = np.sum(lattice_points**2, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Check if the 12 closest are distinct (they are, lattice points are distinct)
    # And if they form a valid non-overlapping cluster (they do, lattice spacing is touching distance).
    # Touching is allowed (disjoint means no interior overlap).
    
    selected_centers = lattice_points[sorted_idx[:12]]
    
    # Center the cluster at (0,0)
    centroid = np.mean(selected_centers, axis=0)
    selected_centers = selected_centers - centroid
    
    # The inner hexagons have side 1 and angle 0 relative to the cluster frame.
    # We will rotate the whole cluster.
    
    # Pre-compute vertices of the 12 hexagons in the cluster frame (angle 0)
    base_verts_list = []
    for c in selected_centers:
        verts = _hex_vertices(c, 1.0, 0.0)
        base_verts_list.append(verts)
        
    # 2. Search over global rotation of the cluster and outer hexagon.
    # Due to symmetry, we can fix the outer hexagon angle to 0 and rotate the cluster?
    # Or fix cluster angle to 0 and rotate outer?
    # It doesn't matter. Let's fix outer angle to 0.
    # We vary the cluster rotation angle phi from 0 to 60 degrees (periodicity of hexagon).
    # Actually, the cluster is not perfectly symmetric, so we should check 0 to 360.
    # But the outer hexagon is symmetric under 60 deg rotation.
    # So relative angle matters modulo 60?
    # Let's just scan 0 to 60 degrees with fine steps.
    
    best_side = float('inf')
    best_phi = 0.0
    
    # Coarse search
    step_deg = 0.1
    for phi_deg in np.arange(0, 60, step_deg):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        phi_rad = np.deg2rad(phi_deg)
        rot_matrix = np.array([
            [np.cos(phi_rad), -np.sin(phi_rad)],
            [np.sin(phi_rad), np.cos(phi_rad)]
        ])
        
        # Rotate all vertices
        all_verts = []
        for verts in base_verts_list:
            rotated = verts @ rot_matrix.T
            all_verts.append(rotated)
            
        # Flatten for _min_side_for_cluster
        flat_verts = np.vstack(all_verts)
        
        # Calculate min side for outer hexagon with angle 0
        S = _min_side_for_cluster(flat_verts, 0.0, (0,0))
        
        if S < best_side:
            best_side = S
            best_phi = phi_deg
            
    # Fine search around best_phi
    if time.time() - start_time < TIME_LIMIT - 2:
        fine_step = 0.001
        # Search in a window of +/- 1 degree
        search_range = np.arange(best_phi - 1.0, best_phi + 1.0, fine_step)
        for phi_deg in search_range:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            if phi_deg < 0 or phi_deg >= 60:
                continue
                
            phi_rad = np.deg2rad(phi_deg)
            rot_matrix = np.array([
                [np.cos(phi_rad), -np.sin(phi_rad)],
                [np.sin(phi_rad), np.cos(phi_rad)]
            ])
            
            all_verts = []
            for verts in base_verts_list:
                rotated = verts @ rot_matrix.T
                all_verts.append(rotated)
                
            flat_verts = np.vstack(all_verts)
            S = _min_side_for_cluster(flat_verts, 0.0, (0,0))
            
            if S < best_side:
                best_side = S
                best_phi = phi_deg

    # Construct the final result
    # Inner hexagons: centers rotated by best_phi, angle = best_phi
    # Outer hexagon: center (0,0), side best_side, angle 0
    
    phi_rad = np.deg2rad(best_phi)
    rot_matrix = np.array([
        [np.cos(phi_rad), -np.sin(phi_rad)],
        [np.sin(phi_rad), np.cos(phi_rad)]
    ])
    
    final_centers = selected_centers @ rot_matrix.T
    final_inner = []
    for c in final_centers:
        final_inner.append([c[0], c[1], best_phi])
        
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Verify validity
    is_valid, msg = _check_validity(final_inner, best_side, outer_center, outer_angle)
    
    if not is_valid:
        # If invalid (shouldn't happen with SDF logic, but SAT is strict),
        # add a tiny epsilon to side length
        best_side *= 1.00001
        is_valid, msg = _check_validity(final_inner, best_side, outer_center, outer_angle)
        
    if not is_valid:
        # Fallback to a known valid configuration if something went wrong
        # Just return a large valid one
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return (final_inner, outer_center, best_side, outer_angle)

if __name__ == "__main__":
    # Quick test
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    print(f"Time taken: {time.time() - START_TIME}")
