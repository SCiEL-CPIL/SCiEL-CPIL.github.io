# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_containing_hex_side(positions, angle_deg=0):
    """
    Compute the minimum side length s of a regular hexagon (centered at origin, angle angle_deg)
    that contains all unit hexagons at given positions (with angle 0).
    
    For a hexagon with side s and angle a, the vertices are at distance s from center at angles a+60*i.
    A point (px, py) is inside if for each edge, the point is on the correct side.
    
    We binary search on s.
    """
    n = len(positions)
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(n):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    a_rad = np.radians(angle_deg)
    
    def _is_inside(s):
        """Check if all vertices are inside the outer hexagon with side s."""
        # Outer hexagon vertices
        outer_verts = _hex_vertices(0, 0, s, angle_deg)
        # For each edge (p1, p2), check cross(p2-p1, v-p1) >= -1e-6
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1) % 6]
            edge = p2 - p1
            # Cross product for all vertices
            # cross(edge, v - p1) = edge[0]*(v[1]-p1[1]) - edge[1]*(v[0]-p1[0])
            diffs = all_verts - p1
            crosses = edge[0] * diffs[:, 1] - edge[1] * diffs[:, 0]
            if np.any(crosses < -1e-6):
                return False
        return True
    
    # Binary search for minimum s
    lo = np.sqrt(12)  # minimum possible
    hi = 20.0  # upper bound
    
    for _ in range(60):
        mid = (lo + hi) / 2
        if _is_inside(mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def _check_separation(positions):
    """Check if all pairs of unit hexagons at given positions are separated (non-overlapping)."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0)
            vj = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, 0)
            # Separating axis test: check all 12 edge normals
            for verts in [vi, vj]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1) % 6]
                    edge = p2 - p1
                    # Normal perpendicular to edge
                    normal = np.array([-edge[1], edge[0]])
                    # Project all vertices of both hexagons onto normal
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    min_i = proj_i.min()
                    max_i = proj_i.max()
                    min_j = proj_j.min()
                    max_j = proj_j.max()
                    # If separated: max_i < min_j or max_j < min_i (with tolerance)
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        break
                else:
                    continue
                break
            else:
                return False  # Found overlap
    return True

def _objective(positions_flat):
    """Objective function: return negative of 1/s (to minimize), or large penalty if invalid."""
    positions = positions_flat.reshape(-1, 2)
    
    # Check separation
    if not _check_separation(positions):
        return 1e10  # Large penalty
    
    # Compute minimal containing hexagon side
    s = _min_containing_hex_side(positions)
    
    return s  # We want to minimize s

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Initial positions: a reasonable starting configuration
    # Use a hexagonal lattice arrangement
    # Unit hexagons tile with spacing: horizontal 2.5, vertical 2.165 (sqrt(3)*1.25)
    init_positions = np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
        [0.0, -4.33],
    ])
    
    # Try multiple random restarts with local optimization
    best_positions = init_positions.copy()
    best_s = _min_containing_hex_side(best_positions)
    
    # Simple gradient-free optimization using random perturbations
    np.random.seed(42)
    n_restarts = 5
    max_iter = 500
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            positions = init_positions.copy()
        else:
            # Random perturbation of initial positions
            positions = init_positions + np.random.randn(12, 2) * 0.5
        
        # Local search: try small random moves
        current_s = _min_containing_hex_side(positions)
        step_size = 0.1
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Try a random perturbation
            delta = np.random.randn(12, 2) * step_size
            new_positions = positions + delta
            
            if _check_separation(new_positions):
                new_s = _min_containing_hex_side(new_positions)
                if new_s < current_s:
                    positions = new_positions
                    current_s = new_s
                    # Decrease step size occasionally
                    if it % 100 == 0 and step_size > 0.01:
                        step_size *= 0.9
                else:
                    # No improvement, try smaller step
                    if it % 50 == 0 and step_size > 0.01:
                        step_size *= 0.95
            else:
                # Separation violated, reduce step
                step_size *= 0.9
            
            if current_s < best_s:
                best_s = current_s
                best_positions = positions.copy()
    
    # Construct final result
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = best_positions[:, 0]
    inner_hex_data[:, 1] = best_positions[:, 1]
    inner_hex_data[:, 2] = 0.0  # angle in degrees
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
