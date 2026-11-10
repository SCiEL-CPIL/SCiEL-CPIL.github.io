# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _get_all_inner_vertices(inner_data):
    """Get all 72 vertices from all 12 inner hexagons."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return all_verts

def _min_s_for_all_vertices(all_inner_verts, outer_center, angle_deg):
    """Find minimum s such that all inner vertices fit in outer hexagon."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_proj = 0
    for verts in all_inner_verts:
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (math.sqrt(3) / 2)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initial arrangement: 12 hexagons in a flower pattern
    # Center hexagon + 6 surrounding + 6 more
    inner = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.165, 0],
        [1.25, 2.165, 0],
        [-1.25, -2.165, 0],
        [1.25, -2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, -2.165, 0],
        [3.75, -2.165, 0],
        [0, -4.33, 0]
    ])
    
    # Compute initial center of mass
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Find best initial configuration by trying different outer angles
    best_s = float('inf')
    best_inner = inner.copy()
    best_outer = np.array([0, 0, 0.0])
    
    all_verts = _get_all_inner_vertices(inner)
    
    for angle_deg in np.linspace(0, 60, 13):
        s_cand = _min_s_for_all_vertices(all_verts, np.array([0, 0]), angle_deg)
        if s_cand < best_s:
            best_s = s_cand
            best_inner = inner.copy()
            best_outer = np.array([0, 0, angle_deg])
    
    # Local optimization: iteratively adjust positions and outer angle
    current_inner = best_inner.copy()
    current_outer = best_outer.copy()
    current_s = best_s
    
    # Gradient-free local search with adaptive step size
    np.random.seed(42)
    step_size = 0.5
    no_improve_count = 0
    max_no_improve = 50
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Try adjusting one inner hexagon position
        idx = np.random.randint(0, 12)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        
        trial_inner = current_inner.copy()
        trial_inner[idx, 0] += dx
        trial_inner[idx, 1] += dy
        
        # Quick check: verify separation from neighbors
        valid = True
        trial_verts = _hex_vertices(trial_inner[idx, 0], trial_inner[idx, 1], 1.0, trial_inner[idx, 2])
        for j in range(12):
            if j == idx:
                continue
            other_verts = _hex_vertices(trial_inner[j, 0], trial_inner[j, 1], 1.0, trial_inner[j, 2])
            if not _check_separation(trial_verts, other_verts):
                valid = False
                break
        
        if valid:
            # Recompute all vertices and find new optimal s
            trial_all_verts = _get_all_inner_vertices(trial_inner)
            
            # Try a few outer angles
            best_trial_s = float('inf')
            best_trial_angle = current_outer[2]
            for angle_offset in np.linspace(-3, 3, 7):
                trial_angle = current_outer[2] + angle_offset
                s_cand = _min_s_for_all_vertices(trial_all_verts, np.array([0, 0]), trial_angle)
                if s_cand < best_trial_s:
                    best_trial_s = s_cand
                    best_trial_angle = trial_angle
            
            if best_trial_s < current_s * (1 - 1e-8):
                # Full validation
                trial_outer = np.array([0, 0, best_trial_angle])
                if _is_valid(trial_inner, trial_outer, best_trial_s):
                    current_inner = trial_inner
                    current_outer = trial_outer
                    current_s = best_trial_s
                    no_improve_count = 0
                    step_size = min(step_size * 1.1, 2.0)
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            step_size *= 0.8
            no_improve_count = 0
            if step_size < 0.01:
                break
    
    # Final validation with fallback
    if not _is_valid(current_inner, current_outer, current_s):
        # Use the seed configuration
        inner_seed = np.array([
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
            [0, -4, 0]
        ])
        com_seed = np.mean(inner_seed[:, :2], axis=0)
        inner_seed[:, :2] -= com_seed
        current_inner = inner_seed
        current_outer = np.array([0, 0, 0])
        current_s = 8.0
    
    return current_inner, current_outer, current_s

# EVOLVE-BLOCK-END
