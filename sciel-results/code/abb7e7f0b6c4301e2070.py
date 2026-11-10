# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, a_deg):
    """Vertices of a regular hexagon: center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return unit edge normals (outward) for a convex hexagon given its 6 vertices in order."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        e = p2 - p1
        # outward normal: rotate edge by -90 deg (for CCW vertices, outward is right-hand normal)
        # For CCW polygon, outward normal is (dy, -dx) normalized
        nx, ny = e[1], -e[0]
        norm = np.hypot(nx, ny)
        if norm > 1e-15:
            normals[i] = [nx / norm, ny / norm]
    return normals

def _sat_overlap_penalty(v1, v2, tol=1e-6):
    """
    Separating axis test between two convex hexagons.
    Returns 0 if separated (no overlap), else a positive penalty proportional to overlap depth.
    v1, v2: (6,2) arrays of vertices in CCW order.
    """
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    max_pen = 0.0
    for n in np.vstack([n1, n2]):
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # overlap if intervals intersect
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > tol:
            max_pen = max(max_pen, overlap - tol)
    return max_pen

def _point_outside_hex_penalty(v, outer_verts, tol=1e-6):
    """
    Check if point v is outside outer hexagon (CCW vertices).
    For each edge (p1, p2), cross(p2-p1, v-p1) must be >= -tol (inside for CCW).
    Returns max violation (0 if inside).
    """
    n = len(outer_verts)
    max_viol = 0.0
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        e = p2 - p1
        # cross product e x (v - p1)
        cross = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
        if cross < -tol:
            max_viol = max(max_viol, -tol - cross)
    return max_viol

def _compute_outer_side(centers, angles, outer_angle_deg):
    """
    Given inner hex centers/angles and outer hex angle, compute minimal outer side length s
    such that all inner hex vertices fit inside outer hex centered at origin.
    Outer hex centered at origin, side s, angle outer_angle_deg.
    """
    # Outer hex vertices at origin with side s
    # We need to find minimal s such that all inner vertices are inside.
    # For a regular hexagon centered at origin with side s and angle a,
    # the distance from center to each edge (apothem) is s * cos(30 deg) = s * sqrt(3)/2
    # But the distance to a vertex is s.
    # The condition for a point to be inside: for each edge, the signed distance >= 0.
    # Edge i: from vertex i to vertex i+1.
    # For a hexagon with vertices at angles a + 60*i, the edge between vertex i and i+1
    # has a normal pointing outward. The distance from center to edge is s*cos(30) = s*sqrt(3)/2.
    # So the constraint is: for each inner vertex v, and for each outer edge normal n_i,
    # v . n_i <= s * sqrt(3)/2 (since the edge is at distance s*sqrt(3)/2 from center along n_i)
    # Wait, let me reconsider.
    
    # Outer hex vertices: s * (cos(a+60i), sin(a+60i))
    # Edge from vertex i to vertex i+1.
    # The outward normal to edge i: the edge direction is vertex_{i+1} - vertex_i.
    # For a regular hexagon, the outward normal to edge i points in direction a + 60*i + 30 deg? 
    # Actually, let me just compute it directly.
    
    # Simpler: for each inner vertex v, the constraint is that v is inside the outer hex.
    # The outer hex with center 0, side s, angle a has the property that a point (x,y) is inside
    # iff for all 6 edge normals n_j, v . n_j <= apothem = s * cos(30 deg) = s * sqrt(3)/2.
    # The edge normals are at angles a + 30 + 60*j (perpendicular to edges, pointing outward).
    # Wait: vertex j is at angle a + 60*j. Edge j is between vertex j and vertex j+1.
    # The midpoint of edge j is at angle a + 60*j + 30, and the outward normal points in that direction.
    # So outward normal j is at angle a + 30 + 60*j.
    
    a = np.deg2rad(outer_angle_deg)
    normal_angles = a + np.deg2rad(30) + np.deg2rad(np.arange(6) * 60.0)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Collect all inner vertices
    all_verts = []
    for c, ang in zip(centers, angles):
        v = _hex_vertices(c[0], c[1], 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # For each vertex, compute max over normals of (v . n)
    # Constraint: v . n_j <= s * sqrt(3)/2 for all j
    # So s >= (v . n_j) / (sqrt(3)/2) for all j
    # s >= max over all v, j of (v . n_j) / (sqrt(3)/2)
    
    dots = all_verts @ normals.T  # (72, 6)
    max_dot = dots.max()
    s = max_dot / (np.sqrt(3) / 2)
    return max(s, np.sqrt(12))  # ensure s >= sqrt(12)

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    
    np.random.seed(42)
    
    # Initial configuration: hexagonal grid
    # 12 hexagons in a hexagonal arrangement
    # Let's use a 3-4-3 or similar arrangement
    # Actually, let's start with a reasonable initial guess
    
    # Hexagonal lattice: spacing between adjacent hex centers for non-overlapping unit hexagons
    # Two unit hexagons side by side: center distance = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ~ 1.732
    # But they can be closer if rotated. For axis-aligned, min distance is sqrt(3).
    
    # Start with a compact arrangement
    # 12 hexagons: maybe 4 in a row, 3, 3, 2 or similar
    # Let's use a roughly hexagonal cluster
    
    # Initial positions (roughly hexagonal packing)
    dx = np.sqrt(3)  # horizontal spacing for aligned hexagons
    dy = 1.5  # vertical spacing
    
    initial_centers = np.array([
        [0, 0],
        [dx, 0],
        [-dx, 0],
        [0.5*dx, 1.5*dy],
        [-0.5*dx, 1.5*dy],
        [0.5*dx, -1.5*dy],
        [-0.5*dx, -1.5*dy],
        [1.5*dx, 1.5*dy],
        [-1.5*dx, 1.5*dy],
        [1.5*dx, -1.5*dy],
        [-1.5*dx, -1.5*dy],
        [0, -3*dy],
    ])
    
    initial_angles = np.zeros(12)
    outer_angle = 0.0
    
    # State: centers (12,2), angles (12,), outer_angle (1,)
    best_centers = initial_centers.copy()
    best_angles = initial_angles.copy()
    best_outer_angle = outer_angle
    best_s = _compute_outer_side(best_centers, best_angles, best_outer_angle)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # Compute current objective
    def objective(centers, angles, outer_angle):
        # Penalty for overlaps
        overlap_pen = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                v1 = _hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
                v2 = _hex_vertices(centers[j,0], centers[j,1], 1.0, angles[j])
                pen = _sat_overlap_penalty(v1, v2)
                overlap_pen += pen * pen  # quadratic penalty
        
        # Compute outer side
        s = _compute_outer_side(centers, angles, outer_angle)
        
        # Total: minimize s, penalize overlaps heavily
        total = s + 100.0 * overlap_pen
        return total, s, overlap_pen
    
    current_centers = initial_centers.copy()
    current_angles = initial_angles.copy()
    current_outer_angle = outer_angle
    current_obj, current_s, current_overlap = objective(current_centers, current_angles, current_outer_angle)
    
    best_obj = current_obj
    
    step_size = 0.1
    
    while time.time() < deadline:
        # Generate a perturbation
        new_centers = current_centers.copy()
        new_angles = current_angles.copy()
        new_outer_angle = current_outer_angle
        
        # Perturb one hexagon's position
        idx = np.random.randint(12)
        new_centers[idx, 0] += np.random.randn() * step_size
        new_centers[idx, 1] += np.random.randn() * step_size
        
        # Sometimes perturb angle
        if np.random.rand() < 0.3:
            new_angles[idx] += np.random.randn() * 5.0
        
        # Sometimes perturb outer angle
        if np.random.rand() < 0.1:
            new_outer_angle += np.random.randn() * 2.0
        
        # Evaluate
        new_obj, new_s, new_overlap = objective(new_centers, new_angles, new_outer_angle)
        
        # Accept if better or with probability
        if new_obj < current_obj:
            current_centers = new_centers
            current_angles = new_angles
            current_outer_angle = new_outer_angle
            current_obj = new_obj
            current_s = new_s
            current_overlap = new_overlap
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_centers = current_centers.copy()
                best_angles = current_angles.copy()
                best_outer_angle = current_outer_angle
                best_s = new_s
        else:
            # Metropolis criterion
            delta = new_obj - current_obj
            if T > T_min:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_centers = new_centers
                    current_angles = new_angles
                    current_outer_angle = new_outer_angle
                    current_obj = new_obj
                    current_s = new_s
                    current_overlap = new_overlap
        
        # Cool down
        T *= cooling
        
        # Adjust step size based on acceptance
        if T < 0.1:
            step_size = max(0.01, step_size * 0.99)
    
    # Final refinement: try to reduce s by adjusting outer angle
    # The best_s is already computed
    
    # Construct output
    inner = np.column_stack([best_centers[:, 0], best_centers[:, 1], best_angles])
    outer = np.array([0.0, 0.0, best_outer_angle])
    s = best_s
    
    # Verify: ensure s >= sqrt(12)
    if s < np.sqrt(12):
        s = np.sqrt(12)
    
    return inner, outer, s

# EVOLVE-BLOCK-END
