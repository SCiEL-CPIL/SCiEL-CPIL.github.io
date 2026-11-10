# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_enclosing_hex_side(vertices, angle_deg):
    """
    Given a set of vertices and a hexagon orientation angle, find the minimum side length
    of a regular hexagon (centered at origin) with that orientation that contains all vertices.
    
    For a regular hexagon with side s and angle a, the support function in direction u is:
    h(u) = s * max_i |cos(u - (a + 60*i))|  for i=0..5
    
    Actually, for a regular hexagon with vertices at angles a+60i, the support function is:
    h(u) = s * max_i cos(u - (a + 60*i))  but only for directions where the max is positive.
    
    More precisely, the distance from center to edge in direction u is:
    s * cos(30 * (u mod 60 - 30)) ... let me think again.
    
    A regular hexagon with side s has inradius r = s * sqrt(3)/2.
    The edges are at angles a + 30 + 60*i for i=0..5 (perpendicular to edge normals).
    Actually, the edge normals are at angles a + 60*i + 30 for i=0..5? No.
    
    Let me reconsider. Vertices are at angles a, a+60, ..., a+300.
    The edges connect consecutive vertices. The normal to the edge between vertices at angles a and a+60
    is at angle a+30 (pointing outward).
    
    So the 6 outward normals are at angles a + 30 + 60*i for i=0..5.
    
    The distance from center to each edge (inradius) is s * sqrt(3)/2.
    
    For a point p to be inside the hexagon, we need:
    p · n_i <= s * sqrt(3)/2  for all i, where n_i is the unit normal at angle a+30+60*i.
    
    So the minimum s such that all points are inside is:
    s = max_i (p_i · n_i) / (sqrt(3)/2) = (2/sqrt(3)) * max_i (p_i · n_i)
    
    But we need the max over all points AND all normals.
    """
    a = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = a + np.deg2rad(30 + 60 * i)
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)  # shape (6, 2)
    
    # For each point and each normal, compute dot product
    # vertices shape: (N, 2)
    dots = vertices @ normals.T  # shape (N, 6)
    max_dot = np.max(dots)
    
    return max_dot / (np.sqrt(3) / 2)


def _objective(params, fixed_angles):
    """
    Objective function: minimize the side length of the enclosing hexagon.
    params: array of shape (22,) containing (x0,y0, x1,y1, ..., x10,y10)
    fixed_angles: array of shape (11,) with angles for each hexagon
    """
    centers = params.reshape(11, 2)
    
    # Compute all vertices of all inner hexagons
    all_vertices = []
    for i in range(11):
        cx, cy = centers[i]
        ang = fixed_angles[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # shape (66, 2)
    
    # The enclosing hexagon is centered at origin with angle 0
    side = _min_enclosing_hex_side(all_vertices, 0.0)
    
    return side


def _check_validity(inner_hex_data, outer_hex_data, outer_side):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if outer_side < np.sqrt(11):
        return False
    
    # Check separation between all pairs of inner hexagons
    n = len(inner_hex_data)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = inner_hex_data[i]
            cx2, cy2, a2 = inner_hex_data[j]
            
            # Get vertices
            v1 = _hex_vertices(cx1, cy1, 1.0, a1)
            v2 = _hex_vertices(cx2, cy2, 1.0, a2)
            
            # Check all 12 edge normals of hexagon 1 and 12 of hexagon 2
            # Actually, we need to check separation on at least one of the 12 edge normals
            # The 12 edge normals are the 6 normals of hex1 and 6 of hex2
            
            separated = False
            for k in range(6):
                # Normals of hexagon 1
                theta = np.deg2rad(a1) + np.deg2rad(30 + 60 * k)
                n_vec = np.array([np.cos(theta), np.sin(theta)])
                
                proj1 = v1 @ n_vec
                proj2 = v2 @ n_vec
                
                max1 = np.max(proj1)
                min2 = np.min(proj2)
                max2 = np.max(proj2)
                min1 = np.min(proj1)
                
                if max1 - min2 < -1e-6 or max2 - min1 < -1e-6:
                    separated = True
                    break
            
            if not separated:
                for k in range(6):
                    theta = np.deg2rad(a2) + np.deg2rad(30 + 60 * k)
                    n_vec = np.array([np.cos(theta), np.sin(theta)])
                    
                    proj1 = v1 @ n_vec
                    proj2 = v2 @ n_vec
                    
                    max1 = np.max(proj1)
                    min2 = np.min(proj2)
                    max2 = np.max(proj2)
                    min1 = np.min(proj1)
                    
                    if max1 - min2 < -1e-6 or max2 - min1 < -1e-6:
                        separated = True
                        break
            
            if not separated:
                return False
    
    # Check that all inner vertices are inside the outer hexagon
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, outer_side, oa)
    
    for i in range(n):
        cx, cy, a = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        for p in verts:
            # Check against each edge of outer hexagon
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge_vec = p2 - p1
                to_p = p - p1
                cross_val = edge_vec[0] * to_p[1] - edge_vec[1] * to_p[0]
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check edge lengths of inner hexagons
    for i in range(n):
        cx, cy, a = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        for k in range(6):
            d = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(d - 1.0) > 1e-8:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    # Use symmetry: fix the outer hexagon at origin with angle 0
    # Use 6-fold symmetry for the inner hexagons where possible
    
    # Initial configuration: center + 6 around + 4 in gaps
    # Using hexagonal coordinates
    # Center at (0,0)
    # 6 neighbors at distance 2 in directions 0, 60, 120, 180, 240, 300
    
    # For a hexagon packing, adjacent hexagons (sharing an edge) have centers at distance 2*inradius = 2*sqrt(3)/2 * 2 = 2*sqrt(3)? 
    # No. For unit side hexagons, the distance between centers of edge-sharing hexagons is 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3) * 2 / 2... 
    # Inradius of unit hexagon = sqrt(3)/2. So center-to-center for edge-sharing = 2 * sqrt(3)/2 = sqrt(3).
    # Wait, that's not right either. Let me think.
    
    # A regular hexagon with side s has inradius r = s * sqrt(3)/2 and circumradius R = s.
    # Two hexagons sharing an edge: the distance between their centers is 2r = s*sqrt(3).
    # For s=1, distance = sqrt(3) ≈ 1.732.
    
    # Actually no. If two hexagons share an edge, the edge is common. The centers are on opposite sides of the edge.
    # The distance from center to edge is r = s*sqrt(3)/2. So center-to-center = 2r = s*sqrt(3).
    # For s=1: distance = sqrt(3) ≈ 1.732.
    
    # But wait, that seems too close. Let me verify with coordinates.
    # Hexagon 1 centered at (0,0) with vertices at angles 0,60,...,300.
    # Vertex at angle 0: (1, 0). Vertex at angle 60: (0.5, sqrt(3)/2).
    # The edge between these two vertices has midpoint at (0.75, sqrt(3)/4).
    # The normal to this edge points outward at angle 30 degrees.
    # A second hexagon sharing this edge would have its center at distance r = sqrt(3)/2 from the edge, on the other side.
    # So center of hex 2 = midpoint + r * (cos(30), sin(30)) = (0.75 + sqrt(3)/2 * cos(30), sqrt(3)/4 + sqrt(3)/2 * sin(30))
    # = (0.75 + sqrt(3)/2 * sqrt(3)/2, sqrt(3)/4 + sqrt(3)/2 * 1/2)
    # = (0.75 + 3/4, sqrt(3)/4 + sqrt(3)/4)
    # = (1.5, sqrt(3)/2)
    # Distance from (0,0) to (1.5, sqrt(3)/2) = sqrt(2.25 + 3/4) = sqrt(3) ≈ 1.732. Correct.
    
    # So for 7 hexagons in a flower pattern: center + 6 at distance sqrt(3) in directions 0,60,...,300.
    # But we need 11, so we add 4 more.
    
    # Let's use a different approach: start with a good initial guess and optimize.
    
    # Initial guess: 11 hexagons in a compact arrangement
    # Using the fact that the optimal packing likely has some symmetry
    
    # Start with 7 in a flower + 4 more
    fixed_angles = np.zeros(11)  # All hexagons have angle 0 initially
    
    # Initial positions
    r = np.sqrt(3)  # Distance for edge-sharing
    initial = np.array([
        [0, 0],           # 0: center
        [r, 0],           # 1
        [r/2, r*np.sqrt(3)/2],  # 2
        [-r/2, r*np.sqrt(3)/2], # 3
        [-r, 0],          # 4
        [-r/2, -r*np.sqrt(3)/2], # 5
        [r/2, -r*np.sqrt(3)/2],  # 6
        # 4 more in the gaps
        [3*r/2, r*np.sqrt(3)/2],  # 7
        [0, r*np.sqrt(3)],        # 8
        [-3*r/2, r*np.sqrt(3)/2], # 9
        [3*r/2, -r*np.sqrt(3)/2], # 10
    ])
    
    best_side = np.inf
    best_params = initial.copy()
    
    # Try multiple optimization runs with different initial guesses
    num_trials = 20
    rng = np.random.default_rng(42)
    
    for trial in range(num_trials):
        if time.time() > deadline - 60:  # Safety margin
            break
        
        if trial == 0:
            params0 = initial.copy()
        else:
            # Perturb the initial guess
            params0 = initial + rng.normal(0, 0.1, size=(22,))
        
        result = minimize(
            _objective,
            params0,
            args=(fixed_angles,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if result.success or result.fun < best_side:
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x.copy()
    
    # Construct the final result
    centers = best_params.reshape(11, 2)
    inner_hex_data = np.column_stack([centers, fixed_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_side)
    
    # Validate
    if not _check_validity(inner_hex_data, outer_hex_data, outer_hex_side_length):
        # Fall back to a known valid solution
        inner_hex_data = np.array([
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
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
