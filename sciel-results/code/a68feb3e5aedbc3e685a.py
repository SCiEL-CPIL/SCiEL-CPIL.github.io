# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, s, angle_deg):
    """Compute vertices of a regular hexagon centered at (cx, cy) with side s and angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_separated(hex1, hex2, tol=1e-6):
    """
    Check if two unit hexagons are separated (no intersection).
    hex1, hex2: (cx, cy, angle_deg) tuples
    A hexagon is separated if on at least one of the 12 edge normals, the projected intervals are apart by more than tol.
    """
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    v1 = _hex_vertices(cx1, cy1, 1.0, a1)
    v2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    # Get all 12 edge normals (6 from each hexagon)
    for vertices in [v1, v2]:
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            # Normal to the edge
            normal = np.array([-edge[1], edge[0]])
            normal_len = np.linalg.norm(normal)
            if normal_len < 1e-12:
                continue
            normal = normal / normal_len
            
            # Project both hexagons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check if intervals are apart by more than tol
            if max1 < min2 - tol or max2 < min1 - tol:
                return True
    
    return False

def _is_valid(inner_hex_data, outer_hex_data, side, tol=1e-6):
    """
    Validate the packing configuration.
    inner_hex_data: (11, 3) array of (x, y, angle_deg)
    outer_hex_data: (3,) array of (x, y, angle_deg)
    side: float, side length of outer hexagon
    """
    # Check side >= sqrt(11)
    if side < np.sqrt(11) - 1e-9:
        return False
    
    # Check all values finite
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if not np.isfinite(side):
        return False
    
    # Shift coordinates by outer center
    ox, oy, oa = outer_hex_data
    shifted_inner = inner_hex_data.copy()
    shifted_inner[:, 0] -= ox
    shifted_inner[:, 1] -= oy
    
    # Check edge lengths are within 1e-8 of 1
    for i in range(11):
        v = _hex_vertices(shifted_inner[i, 0], shifted_inner[i, 1], 1.0, shifted_inner[i, 2])
        for j in range(6):
            edge_len = np.linalg.norm(v[(j + 1) % 6] - v[j])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check every pair of inner hexagons is separated
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hex_separated(shifted_inner[i], shifted_inner[j], tol):
                return False
    
    # Check containment: every inner vertex must be inside or on the outer hexagon
    outer_v = _hex_vertices(ox, oy, side, oa)
    for i in range(11):
        inner_v = _hex_vertices(shifted_inner[i, 0], shifted_inner[i, 1], 1.0, shifted_inner[i, 2])
        for p in inner_v:
            for k in range(6):
                p1 = outer_v[k]
                p2 = outer_v[(k + 1) % 6]
                # cross(p2 - p1, p - p1) >= -tol
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -tol:
                    return False
    
    return True

def _required_outer_side(inner_hex_data, angle_deg=0):
    """
    Compute the minimum side length of a regular hexagon (with given angle) centered at origin
    that contains all inner hexagons.
    """
    # For a regular hexagon with side s and angle a, the constraint is that each vertex
    # of each inner hexagon must satisfy the 6 linear inequalities.
    # We need to find the minimum s such that all constraints are satisfied.
    
    ox, oy = 0, 0
    oa = angle_deg
    
    # Get all inner vertices
    all_vertices = []
    for i in range(11):
        v = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    # The outer hexagon with side s, angle a has vertices at s*cos(a + 60*i), s*sin(a + 60*i)
    # The half-width in direction of each edge normal is s * cos(30 deg) = s * sqrt(3)/2
    # More precisely, for a regular hexagon, the distance from center to each edge is s * sqrt(3)/2
    
    # For each of the 6 edge directions, the maximum projection of any vertex onto the normal
    # must be <= s * sqrt(3)/2
    
    max_s = np.sqrt(11)  # minimum allowed
    angles = np.radians(angle_deg + 60 * np.arange(6))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    for normal in normals:
        proj = np.dot(all_vertices, normal)
        max_proj = np.max(proj)
        # We need max_proj <= s * sqrt(3)/2, so s >= max_proj / (sqrt(3)/2)
        s_needed = max_proj / (np.sqrt(3) / 2)
        if s_needed > max_s:
            max_s = s_needed
    
    return max_s

def _build_hex_cluster_11(pattern="staggered"):
    """
    Build an initial configuration of 11 hexagons in a compact cluster.
    """
    if pattern == "staggered":
        # 3x4 grid with staggering
        return np.array([
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
    elif pattern == "square":
        # 3x3 + 2
        return np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [0, 2.17, 0],
            [0, -2.17, 0],
            [-2.5, 2.17, 0],
            [2.5, 2.17, 0],
            [-2.5, -2.17, 0],
            [2.5, -2.17, 0],
            [-1.25, 4.34, 0],
            [1.25, 4.34, 0],
        ])
    elif pattern == "rhombus":
        return np.array([
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
    else:
        return _build_hex_cluster_11("staggered")

def _local_search(initial_config, max_iter=5000, step_size=0.1, cooling=0.999, T_init=1.0, T_min=1e-4):
    """
    Perform local search (simulated annealing) to optimize the configuration.
    """
    pos = initial_config[:, :2].copy()
    angles = np.zeros(11)  # keep angles at 0 for now
    current_side = _required_outer_side(initial_config)
    best_side = current_side
    best_pos = pos.copy()
    best_angles = angles.copy()
    
    T = T_init
    for _ in range(max_iter):
        if T < T_min:
            break
        
        # Pick a random hexagon to move
        i = np.random.randint(0, 11)
        delta_x = np.random.uniform(-step_size, step_size)
        delta_y = np.random.uniform(-step_size, step_size)
        
        new_pos = pos.copy()
        new_pos[i, 0] += delta_x
        new_pos[i, 1] += delta_y
        
        new_config = np.column_stack([new_pos, angles])
        new_side = _required_outer_side(new_config)
        
        # Check if valid (no overlaps)
        valid = True
        for j in range(11):
            for k in range(j + 1, 11):
                if not _hex_separated(new_config[j], new_config[k], 1e-6):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            if new_side < best_side:
                best_side = new_side
                best_pos = new_pos.copy()
                best_angles = angles.copy()
            
            delta = new_side - current_side
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                pos = new_pos
                current_side = new_side
        else:
            # Penalize invalid configurations
            penalty = 10.0
            delta = (new_side + penalty) - current_side
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                pos = new_pos
                current_side = new_side + penalty
        
        T *= cooling
    
    return best_side, best_pos, best_angles

def _compute_side_with_containment(pos, angles, outer_angle=0):
    """
    Compute the required outer side length given positions and angles.
    Also check for overlaps.
    """
    config = np.column_stack([pos, angles])
    # Check overlaps
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hex_separated(config[i], config[j], 1e-6):
                return np.inf
    
    return _required_outer_side(config, outer_angle)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    np.random.seed(42)
    
    best_side = np.inf
    best_pos = None
    best_angles = None
    
    # Try multiple initial patterns
    patterns = ["staggered", "square", "rhombus"]
    for pattern in patterns:
        initial_config = _build_hex_cluster_11(pattern)
        side, pos, angles = _local_search(initial_config, max_iter=2000, step_size=0.15)
        if side < best_side:
            best_side = side
            best_pos = pos
            best_angles = angles
    
    # Also try with random perturbations
    for _ in range(3):
        initial_config = _build_hex_cluster_11("staggered")
        initial_config[:, 0] += np.random.uniform(-0.5, 0.5, 11)
        initial_config[:, 1] += np.random.uniform(-0.5, 0.5, 11)
        side, pos, angles = _local_search(initial_config, max_iter=2000, step_size=0.12)
        if side < best_side:
            best_side = side
            best_pos = pos
            best_angles = angles
    
    # Final refinement with smaller steps
    if best_pos is not None:
        side, pos, angles = _local_search(
            np.column_stack([best_pos, best_angles]),
            max_iter=3000, step_size=0.05
        )
        if side < best_side:
            best_side = side
            best_pos = pos
            best_angles = angles
    
    # Construct return values
    inner_hex_data = np.column_stack([best_pos, best_angles])
    outer_hex_data = np.array([0, 0, 0])
    
    # Ensure side is at least sqrt(11)
    final_side = max(best_side, np.sqrt(11))
    
    return inner_hex_data, outer_hex_data, final_side

# EVOLVE-BLOCK-END
