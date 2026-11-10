# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _rigid_cluster_positions(angle_deg):
    """
    Generate positions of 12 unit hexagons in a fixed hexagonal cluster configuration,
    rotated by angle_deg around the origin.
    
    The cluster uses a hexagonal close-packed arrangement:
    Center: (0,0)
    Ring 1: 6 hexagons at distance 2.5 in hexagonal directions
    Ring 2: 5 additional hexagons to make 12 total
    
    For unit hexagons (side=1), the distance between adjacent centers in a hex lattice is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464
    Actually, for regular hexagons with side 1, the distance between centers of adjacent hexagons is 2*sqrt(3)/2 * 2... 
    Let me recalculate: a regular hexagon with side s has width 2s and height sqrt(3)*s.
    For side=1: width=2, height=sqrt(3).
    Adjacent hex centers are separated by 2*sqrt(3)/2 * ... 
    
    Actually, two unit regular hexagons that share an edge have their centers separated by sqrt(3) (the distance between parallel sides is sqrt(3)*1 = sqrt(3)).
    Wait, no. The distance between opposite sides of a unit hexagon is sqrt(3). So adjacent centers are sqrt(3) apart? No...
    
    For a regular hexagon with side length 1:
    - The distance from center to any vertex is 1
    - The distance from center to the midpoint of any side is cos(30°) = sqrt(3)/2
    - The distance between two parallel sides is 2 * sqrt(3)/2 = sqrt(3)
    
    So two unit hexagons sharing an edge have centers separated by sqrt(3).
    
    In a hexagonal lattice, the 6 nearest neighbors are at distance sqrt(3) in directions 0°, 60°, 120°, 180°, 240°, 300°.
    """
    d = np.sqrt(3.0)  # distance between adjacent hex centers
    
    # Define the 12 positions in the cluster (before rotation)
    # Using axial coordinates for hex grid, converted to Cartesian
    # Center at (0,0)
    # 6 neighbors at distance d in 60-degree increments
    # Then 5 more at the next ring positions
    
    positions_xy = []
    # Center
    positions_xy.append([0.0, 0.0])
    
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = np.radians(60 * i)
        positions_xy.append([d * np.cos(angle), d * np.sin(angle)])
    
    # Ring 2: need 5 more. In a hex grid, the second ring has 12 positions.
    # We'll pick 5 that form a compact shape.
    # Positions at distance 2*d in directions 0,60,120,180,240,300
    # And positions at distance sqrt(3)*d at 30,90,150,210,270,330
    # Actually, let me use the proper hex lattice second ring.
    # In axial coordinates (q,r), second ring positions are:
    # (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (1,2), (2,1), (3,-1)... 
    # Let me just use Cartesian: the 12 second-ring positions are at:
    # 6 at distance 2d in directions 0,60,...,300
    # 6 at distance d*sqrt(3) in directions 30,90,...,330
    
    d2 = 2.0 * d  # distance for the "straight" second ring
    d2b = d * np.sqrt(3.0)  # distance for the "diagonal" second ring
    
    # Pick 5 from the second ring to make a nice shape
    # Let's pick: 0°, 60°, 120°, 180°, 240° at distance 2d
    for i in range(5):
        angle = np.radians(60 * i)
        positions_xy.append([d2 * np.cos(angle), d2 * np.sin(angle)])
    
    positions_xy = np.array(positions_xy)
    
    # Rotate by angle_deg
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    return positions_xy @ rot.T

def _min_containing_hex_side_rigid(positions, outer_angle_deg, outer_center):
    """
    Compute the minimum side length of a regular hexagon with given angle and center
    that contains all inner hexagons (each unit hex at given positions, angle 0).
    """
    # Get all 72 vertices
    all_verts = np.empty((12 * 6, 2))
    for i in range(12):
        cx, cy = positions[i]
        a = 0.0  # inner hexagons have angle 0
        angles = a + np.radians(60 * np.arange(6))
        all_verts[i*6:(i+1)*6] = np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    # Shift vertices relative to outer center
    shifted = all_verts - outer_center
    
    # For a regular hexagon with center at origin, angle outer_angle_deg, side s:
    # The support function in direction theta is s * cos(theta - outer_angle - 30°) for the 6 edge normals
    # Actually, the hexagon contains point p iff for all 6 edge normals n_k, p·n_k <= s*sqrt(3)/2
    # where n_k are unit normals to the edges, pointing outward.
    
    a_rad = np.radians(outer_angle_deg)
    # Edge normals are at angles outer_angle + 30 + 60*k for k=0..5
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    dots = shifted @ normals.T  # (72, 6)
    max_dot = np.max(dots, axis=0)  # (6,)
    
    # The constraint is max_dot[k] <= s * sqrt(3)/2 for each k
    # So s >= max_dot[k] * 2/sqrt(3) for each k
    s = np.max(max_dot) * (2.0 / np.sqrt(3.0))
    return s

def _objective_rigid(params):
    """
    Objective: minimize s given rotation angle and translation of the rigid cluster.
    params: [angle_deg, tx, ty]
    """
    angle_deg = params[0]
    tx = params[1]
    ty = params[2]
    
    positions = _rigid_cluster_positions(angle_deg) + np.array([tx, ty])
    
    # Outer center is at origin (we optimize tx,ty to center the cluster)
    outer_center = np.array([0.0, 0.0])
    
    s = _min_containing_hex_side_rigid(positions, 0.0, outer_center)
    return s

def _objective_rigid2(params):
    """
    Objective: minimize s given rotation angle, translation, and outer hex angle.
    params: [angle_deg, tx, ty, outer_angle_deg]
    """
    angle_deg = params[0]
    tx = params[1]
    ty = params[2]
    outer_angle = params[3]
    
    positions = _rigid_cluster_positions(angle_deg) + np.array([tx, ty])
    
    outer_center = np.array([0.0, 0.0])
    
    s = _min_containing_hex_side_rigid(positions, outer_angle, outer_center)
    return s

def _check_separation_fast(positions):
    """
    Check if all pairs of unit hexagons at given positions are separated.
    Returns True if separated, False otherwise.
    Uses Separating Axis Theorem.
    """
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0))
    
    for i in range(n):
        vi = verts_list[i]
        for j in range(i+1, n):
            vj = verts_list[j]
            found_sep = False
            for k in range(6):
                p1 = vi[k]
                p2 = vi[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                    found_sep = True
                    break
            if found_sep:
                continue
            for k in range(6):
                p1 = vj[k]
                p2 = vj[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                    found_sep = True
                    break
            if not found_sep:
                return False
    return True

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a rigid cluster approach: fix the topology, optimize rotation and translation.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    # Try optimizing the rigid cluster with different initial angles
    # The cluster is fixed in topology, so we just rotate and translate it
    
    # First, verify the base cluster is valid (separated)
    base_pos = _rigid_cluster_positions(0.0)
    if not _check_separation_fast(base_pos):
        # Fallback to the old approach if the cluster isn't valid
        return _hexagon_packing_12_fallback()
    
    # Optimize: [cluster_angle, tx, ty, outer_angle]
    inits = []
    for cluster_angle in [0, 10, 20, 30, -10, -20]:
        for outer_angle in [0, 15, 30, -15]:
            init = np.array([float(cluster_angle), 0.0, 0.0, float(outer_angle)])
            inits.append(init)
    
    for _ in range(10):
        init = np.array([
            np.random.uniform(-30, 30),
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(-30, 30)
        ])
        inits.append(init)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                _objective_rigid2,
                init,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
        except Exception:
            continue
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
    
    # Also try with fixed outer angle = 0 for simplicity
    for cluster_angle in [0, 15, 30, -15]:
        if time.time() > deadline - 10:
            break
        init = np.array([float(cluster_angle), 0.0, 0.0])
        try:
            res = minimize(
                _objective_rigid,
                init,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
        except Exception:
            continue
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = np.append(res.x, 0.0)  # append outer_angle=0
    
    # Extract results
    if best_params is None or best_s > 100:
        # Fallback
        return _hexagon_packing_12_fallback()
    
    cluster_angle = best_params[0]
    tx = best_params[1]
    ty = best_params[2]
    outer_angle = best_params[3] if len(best_params) > 3 else 0.0
    
    positions = _rigid_cluster_positions(cluster_angle) + np.array([tx, ty])
    
    # Verify separation
    if not _check_separation_fast(positions):
        return _hexagon_packing_12_fallback()
    
    outer_center = np.array([0.0, 0.0])
    
    # Recompute s to be safe
    final_s = _min_containing_hex_side_rigid(positions, outer_angle, outer_center)
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = positions[:, 0]
    inner_hex_data[:, 1] = positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, final_s

def _hexagon_packing_12_fallback():
    """Fallback to the original optimization approach."""
    start_time = time.time()
    deadline = start_time + 580
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    def _hex_lattice_positions(spacing_scale=1.0):
        s = spacing_scale
        h = 2.5 * s
        v = 2.165 * s
        positions = np.array([
            [0.0, 0.0],
            [-h, 0.0],
            [h, 0.0],
            [-h/2, v],
            [h/2, v],
            [-h/2, -v],
            [h/2, -v],
            [-3*h/2, v],
            [3*h/2, v],
            [-3*h/2, -v],
            [3*h/2, -v],
            [0.0, -2*v],
        ])
        return positions

    def _circular_positions(radius=2.0):
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    def _get_all_inner_vertices(positions):
        n = len(positions)
        all_verts = np.empty((n * 6, 2))
        for i in range(n):
            cx, cy = positions[i]
            verts = _hex_vertices(cx, cy, 1.0, 0)
            all_verts[i*6:(i+1)*6] = verts
        return all_verts

    def _min_containing_hex_side(positions, angle_deg, outer_center):
        all_verts = _get_all_inner_vertices(positions)
        shifted = all_verts - outer_center
        a_rad = np.radians(angle_deg)
        normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        dots = shifted @ normals.T
        max_dot = np.max(np.abs(dots))
        s = (2.0 / np.sqrt(3.0)) * max_dot
        return s

    def _objective(params):
        positions = params[:24].reshape(12, 2)
        outer_center = params[24:26]
        angle_deg = params[26]
        if not _check_separation_fast(positions):
            return 1000.0
        s = _min_containing_hex_side(positions, angle_deg, outer_center)
        return s

    inits = []
    base_lattice = _hex_lattice_positions(1.0)
    inits.append(base_lattice)
    for scale in [0.9, 1.1]:
        inits.append(_hex_lattice_positions(scale))
    for r in [1.5, 2.0, 2.5]:
        inits.append(_circular_positions(r))
    for _ in range(5):
        inits.append(np.random.randn(12, 2) * 3.0)
    for _ in range(5):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.5)

    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 20:
            break
        for init_angle in [0, 30, 10, -10]:
            if time.time() > deadline - 20:
                break
            init_params = np.zeros(27)
            init_params[:24] = init_pos.flatten()
            init_params[26] = init_angle
            try:
                res = minimize(_objective, init_params, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                continue
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()

    if best_params is None or best_s > 100:
        best_pos = _hex_lattice_positions(1.0)
        best_angle = 0
        best_center = np.array([0.0, 0.0])
        best_s = _min_containing_hex_side(best_pos, best_angle, best_center)
        best_params = np.zeros(27)
        best_params[:24] = best_pos.flatten()
        best_params[26] = best_angle

    final_positions = best_params[:24].reshape(12, 2)
    final_center = best_params[24:26]
    final_angle = best_params[26]
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    outer_hex_data = np.array([final_center[0], final_center[1], final_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
