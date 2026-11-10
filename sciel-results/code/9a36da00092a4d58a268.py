# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon with side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _all_vertices(inner_data, s=1.0):
    """Compute all vertices for all inner hexagons. Returns array of shape (n, 6, 2)."""
    n = len(inner_data)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a = inner_data[i]
        angles = np.deg2rad(a + 60 * np.arange(6))
        verts[i, :, 0] = x + s * np.cos(angles)
        verts[i, :, 1] = y + s * np.sin(angles)
    return verts

def _min_outer_side(verts, outer_center=None):
    """Find minimum side length of a regular hexagon (any orientation) containing all vertices.
    
    For a regular hexagon with side s, the distance from center to any vertex is s.
    The hexagon contains a point if the point is within distance s in all 6 normal directions.
    Actually, a regular hexagon with side s has inradius = s*sqrt(3)/2 and circumradius = s.
    
    For a hexagon centered at origin with angle 0 (flat-top), the half-width in x is s,
    and the half-height in y is s*sqrt(3)/2.
    
    More generally, for a hexagon with side s and rotation angle a, a point (x,y) is inside
    iff for each of the 6 edge normals n_i, the dot product (p-c)·n_i <= s*sqrt(3)/2 (the inradius).
    
    Actually let me reconsider. A regular hexagon with side length s:
    - vertices at distance s from center
    - inradius (apothem) = s * sqrt(3)/2
    
    The constraint for a point to be inside is: for each edge, the signed distance from the edge
    (measured inward) must be >= 0.
    
    For edge i with normal n_i (pointing inward) and edge at distance d_i = s*sqrt(3)/2 from center:
    (p - center) · n_i <= s*sqrt(3)/2
    
    So we need s*sqrt(3)/2 >= max_i [(p - center) · n_i] for all points p.
    Thus s >= (2/sqrt(3)) * max_i [(p - center) · n_i]
    
    But we can also rotate the hexagon, so we minimize over the rotation angle.
    """
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])
    
    # Shift points relative to center
    pts = verts.reshape(-1, 2) - outer_center
    
    # For a given rotation angle a, the 6 outward normals are at angles a + 60*i + 30 (for flat-top)
    # Actually, let me think about this more carefully.
    # 
    # A regular hexagon with side s, centered at origin, with one vertex at angle 0:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, at distance s.
    # Edges connect consecutive vertices. The normal to edge between vertex at angle 0 and 60
    # points in direction 30 degrees (bisector).
    # 
    # For a hexagon rotated by angle a, the 6 outward normals are at angles a + 30 + 60*i for i=0..5
    # The distance from center to each edge (inradius) is s * cos(30°) = s * sqrt(3)/2
    # 
    # A point p is inside iff for all 6 normals n_i: p · n_i <= s * sqrt(3)/2
    # So s >= (2/sqrt(3)) * max_i (p · n_i)
    # 
    # We want to minimize s over the choice of rotation a.
    # s(a) = (2/sqrt(3)) * max_i max_p (p · n_i(a))
    # 
    # To find the minimum, we can sample a or use optimization.
    
    # For efficiency, let's compute the support function for a range of angles
    # and find the minimum.
    
    # Actually, let's be smarter. The minimum enclosing hexagon side length is:
    # s_min = min_a (2/sqrt(3)) * max_{i=0..5} max_p (p · n_i(a))
    
    # Let's try a grid of angles first, then refine.
    best_s = np.inf
    best_a = 0.0
    
    # The normals for angle a are at a + 30 + 60*i
    # Let's try angles from 0 to 60 degrees (by symmetry, 60 degrees is enough)
    a_vals = np.linspace(0, 60, 1201)
    
    for a_deg in a_vals:
        a = np.deg2rad(a_deg)
        # 6 outward normals
        normal_angles = a + np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # For each point, compute max dot product with any normal
        # dots shape: (n_points, 6)
        dots = pts @ normals.T
        max_dot = np.max(dots, axis=1)  # max over normals for each point
        max_over_all = np.max(max_dot)
        
        s_candidate = (2.0 / np.sqrt(3.0)) * max_over_all
        if s_candidate < best_s:
            best_s = s_candidate
            best_a = a_deg
    
    return best_s, best_a

def _check_overlap(inner_data, tol=1e-6):
    """Check if any two inner hexagons overlap. Returns True if no overlap."""
    n = len(inner_data)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
            vj = _hex_vertices(inner_data[j, 0], inner_data[j, 1], 1.0, inner_data[j, 2])
            
            # Check separation using edge normals
            # For hexagon i, compute its 6 edge normals (outward)
            # A pair of hexagons is separated if on at least one of the 12 edge normals,
            # the projected intervals are apart by more than tol.
            
            # Get edge normals for hex i
            for k in range(6):
                # Edge from vertex k to vertex (k+1)%6
                p1 = vi[k]
                p2 = vi[(k+1) % 6]
                edge_vec = p2 - p1
                # Outward normal (perpendicular, pointing away from center)
                center_i = np.array([inner_data[i, 0], inner_data[i, 1]])
                mid = (p1 + p2) / 2
                normal = np.array([-edge_vec[1], edge_vec[0]])
                # Ensure it points outward
                if np.dot(normal, mid - center_i) < 0:
                    normal = -normal
                normal = normal / np.linalg.norm(normal)
                
                # Project both hexagons onto this normal
                proj_i = vi @ normal
                proj_j = vj @ normal
                
                # Check if intervals are separated
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                # Separated if max_i < min_j - tol or max_j < min_i - tol
                if max_i < min_j - tol or max_j < min_i - tol:
                    break
            else:
                # Check hex j's normals too
                separated = False
                for k in range(6):
                    p1 = vj[k]
                    p2 = vj[(k+1) % 6]
                    edge_vec = p2 - p1
                    center_j = np.array([inner_data[j, 0], inner_data[j, 1]])
                    mid = (p1 + p2) / 2
                    normal = np.array([-edge_vec[1], edge_vec[0]])
                    if np.dot(normal, mid - center_j) < 0:
                        normal = -normal
                    normal = normal / np.linalg.norm(normal)
                    
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    
                    if max_i < min_j - tol or max_j < min_i - tol:
                        separated = True
                        break
                
                if not separated:
                    return False
    return True

def _check_containment(verts, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if all inner vertices are inside the outer hexagon."""
    a = np.deg2rad(outer_angle)
    normal_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    pts = verts.reshape(-1, 2) - outer_center
    dots = pts @ normals.T
    
    # Each dot must be <= outer_side * sqrt(3)/2
    max_dot = np.max(dots, axis=1)
    inradius = outer_side * np.sqrt(3) / 2
    
    return np.all(max_dot <= inradius + tol)

def _objective_and_check(inner_data, outer_center, outer_side, outer_angle):
    """Compute objective (1/side) and check validity."""
    verts = _all_vertices(inner_data)
    
    # Check containment
    if not _check_containment(verts, outer_center, outer_side, outer_angle):
        return -1.0, False
    
    # Check overlap
    if not _check_overlap(inner_data):
        return -1.0, False
    
    # Check side >= sqrt(11)
    if outer_side < np.sqrt(11.0) - 1e-9:
        return -1.0, False
    
    # Check edge lengths
    for i in range(len(inner_data)):
        vi = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(vi[(k+1)%6] - vi[k])
            if abs(edge_len - 1.0) > 1e-8:
                return -1.0, False
    
    return 1.0 / outer_side, True

def _optimize_from_initial(initial_inner, time_budget):
    """Optimize starting from initial configuration."""
    start_time = time.time()
    n = len(initial_inner)
    
    # Current best
    best_inner = initial_inner.copy()
    best_outer_center = np.array([0.0, 0.0])
    best_outer_side = np.inf
    best_outer_angle = 0.0
    best_score = -1.0
    
    # Initial evaluation
    verts = _all_vertices(best_inner)
    best_outer_side, best_outer_angle = _min_outer_side(verts, best_outer_center)
    score, valid = _objective_and_check(best_inner, best_outer_center, best_outer_side, best_outer_angle)
    if valid:
        best_score = score
    
    # Coordinate descent on inner positions and angles
    # Variables: x, y, angle for each hexagon
    # We'll optimize by moving one hexagon at a time
    
    step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step in step_sizes:
        if time.time() - start_time > time_budget - 10:
            break
        
        improved = True
        while improved:
            if time.time() - start_time > time_budget - 10:
                break
            improved = False
            
            for i in range(n):
                if time.time() - start_time > time_budget - 10:
                    break
                
                # Try moving hexagon i in small steps
                base = best_inner[i].copy()
                
                # Try x, y, and angle perturbations
                for dx, dy, da in [(step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0),
                                    (0, 0, step), (0, 0, -step),
                                    (step*0.7, step*0.7, 0), (step*0.7, -step*0.7, 0),
                                    (-step*0.7, step*0.7, 0), (-step*0.7, -step*0.7, 0)]:
                    new_inner = best_inner.copy()
                    new_inner[i, 0] = base[0] + dx
                    new_inner[i, 1] = base[1] + dy
                    new_inner[i, 2] = (base[2] + da) % 360
                    
                    # Recompute outer
                    verts_new = _all_vertices(new_inner)
                    s_new, a_new = _min_outer_side(verts_new, best_outer_center)
                    
                    score_new, valid_new = _objective_and_check(new_inner, best_outer_center, s_new, a_new)
                    
                    if valid_new and score_new > best_score + 1e-10:
                        best_inner = new_inner
                        best_outer_side = s_new
                        best_outer_angle = a_new
                        best_score = score_new
                        improved = True
                        break  # Restart the while loop with this improvement
    
    # Also try optimizing the outer center and angle
    if time.time() - start_time < time_budget - 5:
        # Try different outer centers
        for cx in np.linspace(-2, 2, 11):
            for cy in np.linspace(-2, 2, 11):
                if time.time() - start_time > time_budget - 5:
                    break
                oc = np.array([cx, cy])
                verts = _all_vertices(best_inner)
                s_try, a_try = _min_outer_side(verts, oc)
                score_try, valid_try = _objective_and_check(best_inner, oc, s_try, a_try)
                if valid_try and score_try > best_score + 1e-10:
                    best_outer_center = oc
                    best_outer_side = s_try
                    best_outer_angle = a_try
                    best_score = score_try
    
    return best_inner, best_outer_center, best_outer_side, best_outer_angle, best_score

def _make_initial_config(config_type):
    """Generate initial configurations."""
    if config_type == 'hex7_plus4':
        # Central hex + 6 surrounding + 4 more
        # The 7-hex cluster has centers at:
        # (0,0), (2,0), (1,sqrt(3)), (-1,sqrt(3)), (-2,0), (-1,-sqrt(3)), (1,-sqrt(3))
        # Wait, for unit hexagons (side=1), the distance between centers of adjacent hexagons
        # in a hexagonal lattice is 2*inradius = 2*sqrt(3)/2 = sqrt(3)
        # Actually, for two unit hexagons to be tangent, center distance = 2 * inradius = sqrt(3)
        # No wait. Two unit hexagons side by side: the distance between centers is 2 * (sqrt(3)/2) * 2 = 2*sqrt(3)? 
        # Let me reconsider. A unit hexagon has inradius sqrt(3)/2. Two hexagons sharing an edge
        # have centers separated by 2 * inradius = sqrt(3).
        
        sqrt3 = np.sqrt(3)
        # 7-hex cluster: center at (0,0), neighbors at distance sqrt(3) in 6 directions
        # Directions: 0, 60, 120, 180, 240, 300 degrees
        # But for flat-top hexagons, the neighbors are at angles 0, 60, 120, 180, 240, 300
        # with distance sqrt(3)? Let me verify.
        # 
        # Actually, for pointy-top hexagons (vertex at top), the 6 neighbors are at:
        # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
        # Distance = sqrt(3) for horizontal neighbors, sqrt(3/4 + 9/4) = sqrt(3) for others. Yes!
        #
        # For flat-top hexagons (flat edge at top), neighbors are at:
        # (0, sqrt(3)), (sqrt(3)/2, sqrt(3)/2), (sqrt(3)/2, -sqrt(3)/2), (0, -sqrt(3)), (-sqrt(3)/2, -sqrt(3)/2), (-sqrt(3)/2, sqrt(3)/2)
        # Same distances.
        
        # Let's use pointy-top orientation (angle=30 for the hexagon so a vertex points up)
        # Actually, the angle parameter defines where vertex 0 is. Let's use angle=0 (vertex at right).
        # For angle=0, vertices are at 0, 60, 120, 180, 240, 300 degrees.
        # The 6 neighbors for a flat arrangement would be at distance 2*inradius = sqrt(3)
        # in directions perpendicular to the edges.
        
        # Let me just use a known good starting configuration.
        # 7 hexagons in a flower pattern, then 4 more at the "gaps"
        
        d = sqrt3  # distance between adjacent centers
        # 7-hex cluster (center + 6 neighbors)
        centers = [
            [0, 0],
            [d, 0],
            [d/2, d*np.sqrt(3)/2],
            [-d/2, d*np.sqrt(3)/2],
            [-d, 0],
            [-d/2, -d*np.sqrt(3)/2],
            [d/2, -d*np.sqrt(3)/2],
        ]
        # Add 4 more at the "corners" of the hexagonal arrangement
        # The 7-hex cluster has a hexagonal shape. The "corners" are at distance 2*d from center
        # in directions 30, 90, 150, 210, 270, 330 (for pointy-top) or similar.
        # Let's add them at the 6 "outer" positions, but we only need 4.
        # Try adding at 4 of the 6 outer corner positions.
        
        # For a pointy-top 7-hex cluster, the outer vertices of the overall shape are at:
        # The cluster spans from -d to d in x and -d*sqrt(3)/2 to d*sqrt(3)/2 in y
        # The "corners" where we can add more hexagons are at the 6 positions that are
        # adjacent to 2 of the outer hexagons.
        
        # Let's just place 4 more hexagons at reasonable positions and let the optimizer fix them.
        # Positions at distance ~2*d from center in 4 directions
        r2 = 2 * d
        extra = [
            [r2 * np.cos(np.deg2rad(90)), r2 * np.sin(np.deg2rad(90))],
            [r2 * np.cos(np.deg2rad(210)), r2 * np.sin(np.deg2rad(210))],
            [r2 * np.cos(np.deg2rad(330)), r2 * np.sin(np.deg2rad(330))],
            [r2 * np.cos(np.deg2rad(0)), r2 * np.sin(np.deg2rad(0))],
        ]
        
        all_centers = centers + extra
        inner = np.array([[c[0], c[1], 0.0] for c in all_centers])
        return inner
    
    elif config_type == 'linear':
        # 11 hexagons in a row
        d = sqrt3
        inner = np.array([[i * d, 0, 0.0] for i in range(-5, 6)])
        return inner
    
    elif config_type == 'random':
        d = sqrt3
        # Random positions in a hexagonal region
        np.random.seed(42)
        inner = np.zeros((11, 3))
        inner[0] = [0, 0, 0]
        for i in range(1, 11):
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0.5, 3) * d
            inner[i] = [r * np.cos(angle), r * np.sin(angle), 0.0]
        return inner
    
    else:
        # Default: hex7_plus4
        return _make_initial_config('hex7_plus4')

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length (maximizing 1/side).
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Try multiple initial configurations
    configs = ['hex7_plus4', 'linear', 'random']
    
    best_inner = None
    best_outer_center = None
    best_outer_side = np.inf
    best_outer_angle = 0.0
    best_score = -1.0
    
    for cfg in configs:
        if time.time() > deadline - 60:
            break
        
        initial = _make_initial_config(cfg)
        time_budget = min(deadline - time.time(), 180)
        
        inner, oc, side, angle, score = _optimize_from_initial(initial, time_budget)
        
        if score > best_score:
            best_inner = inner
            best_outer_center = oc
            best_outer_side = side
            best_outer_angle = angle
            best_score = score
    
    # Final verification
    if best_inner is not None:
        verts = _all_vertices(best_inner)
        # Recompute optimal outer
        best_outer_side, best_outer_angle = _min_outer_side(verts, best_outer_center)
        
        # Final check
        score_final, valid_final = _objective_and_check(
            best_inner, best_outer_center, best_outer_side, best_outer_angle
        )
        
        if valid_final:
            return best_inner, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]), best_outer_side
    
    # Fallback: return a valid but not optimal solution
    # Use the seed program's solution as fallback
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
