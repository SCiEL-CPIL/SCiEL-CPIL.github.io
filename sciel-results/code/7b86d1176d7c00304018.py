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

def _check_separation_fast(hex1_verts, hex2_verts):
    """
    Fast separation check.
    If they are far apart, return True (separated).
    If bounding boxes don't overlap, return True.
    Else do SAT.
    """
    # Bounding box check
    min1 = hex1_verts.min(axis=0)
    max1 = hex1_verts.max(axis=0)
    min2 = hex2_verts.min(axis=0)
    max2 = hex2_verts.max(axis=0)
    
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return True

    # SAT
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-15:
                continue
            n = n / norm
            proj1 = hex1_verts @ n
            proj2 = hex2_verts @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

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
        # Check inside
        for vpt in v:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                cross = (p2[0] - p1[0]) * (vpt[1] - p1[1]) - (p2[1] - p1[1]) * (vpt[0] - p1[0])
                if cross < -1e-6:
                    return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation_fast(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _min_s_for_config(inner_data, outer_center, outer_angle_deg):
    """
    Calculate minimum s for given inner config and outer angle.
    """
    a = math.radians(outer_angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_proj = 0
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        rel_v = v - outer_center
        for n in normals:
            proj = np.max(np.abs(rel_v @ n))
            if proj > max_proj:
                max_proj = proj
    return max_proj / (math.sqrt(3) / 2)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    # Initial Configuration: Hexagonal Lattice
    e1 = np.array([1.5, math.sqrt(3)/2])
    e2 = np.array([0, math.sqrt(3)])
    
    # Generate a 3x4 grid, center it
    positions = []
    for i in range(3):
        for j in range(4):
            pos = i * e1 + j * e2
            positions.append([pos[0], pos[1], 0.0])
            
    inner = np.array(positions)
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Optimize outer angle for initial
    best_s = float('inf')
    best_outer_angle = 0.0
    for angle_deg in np.linspace(0, 60, 61):
        s_cand = _min_s_for_config(inner, np.array([0, 0]), angle_deg)
        if s_cand < best_s:
            best_s = s_cand
            best_outer_angle = angle_deg
            
    current_inner = inner.copy()
    current_outer = np.array([0, 0, best_outer_angle])
    current_s = best_s
    best_result = (current_inner.copy(), current_outer.copy(), current_s)
    
    # Simulated Annealing
    T = 0.1
    step_size = 0.2
    np.random.seed(42)
    
    # Pre-compute normals for current outer angle? 
    # It's better to recompute or update if angle changes.
    # We will perturb positions and outer angle slightly.
    
    iterations = 0
    max_iterations = 50000
    no_improve_count = 0
    improve_threshold = 1e-5
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Perturb
        trial_inner = current_inner.copy()
        trial_outer = current_outer.copy()
        
        idx = np.random.randint(0, 12)
        trial_inner[idx, 0] += np.random.uniform(-step_size, step_size)
        trial_inner[idx, 1] += np.random.uniform(-step_size, step_size)
        # Small angle perturbation
        trial_inner[idx, 2] += np.random.uniform(-5, 5)
        
        # Perturb outer angle
        trial_outer[2] += np.random.uniform(-2, 2)
        
        # Check Overlaps (Fast)
        valid = True
        # Compute vertices for trial inner
        trial_verts_list = []
        for i in range(12):
            v = _hex_vertices(trial_inner[i, 0], trial_inner[i, 1], 1.0, trial_inner[i, 2])
            trial_verts_list.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                if not _check_separation_fast(trial_verts_list[i], trial_verts_list[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Penalize or reject
            if np.random.random() < math.exp(-1.0 / T):
                # Accept even if invalid? No, we want valid solutions.
                # But for SA to escape, we might need to allow temporary invalidity?
                # The objective is 1/s. Invalid is -inf.
                # Let's just reject invalid moves unless we are in a "penalty" phase.
                # For now, reject.
                pass
            else:
                # Reject
                # Cool down slightly
                T = max(T * 0.999, 0.001)
                step_size = max(step_size * 0.999, 0.01)
                continue
                
        # Calculate new S
        trial_s = _min_s_for_config(trial_inner, np.array([0, 0]), trial_outer[2])
        
        # Check if valid (containment)
        # _min_s_for_config gives the s required. If we use that s, it is valid by construction for containment.
        # We just need to ensure s >= sqrt(12) which is ~3.46. Our s is ~5.
        
        delta_s = current_s - trial_s
        
        if delta_s > 0:
            current_inner = trial_inner
            current_outer = trial_outer
            current_s = trial_s
            T = max(T * 0.999, 0.001)
            step_size = max(step_size * 0.999, 0.01)
            
            if current_s < best_result[2] - improve_threshold:
                best_result = (current_inner.copy(), current_outer.copy(), current_s)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            if np.random.random() < math.exp(-delta_s / T):
                current_inner = trial_inner
                current_outer = trial_outer
                current_s = trial_s
                T = max(T * 0.999, 0.001)
                step_size = max(step_size * 0.999, 0.01)
            else:
                T = max(T * 0.999, 0.001)
                step_size = max(step_size * 0.999, 0.01)
                
        # If no improvement for long, restart with smaller step or re-seed
        if no_improve_count > 2000:
            # Try a random restart from best
            current_inner = best_result[0].copy()
            current_outer = best_result[1].copy()
            current_s = best_result[2]
            T = 0.05
            step_size = 0.1
            no_improve_count = 0

    # Final Verification
    if not _is_valid(best_result[0], best_result[1], best_result[2]):
        # Fallback to initial
        return inner, np.array([0, 0, best_outer_angle]), best_s
        
    return best_result

# EVOLVE-BLOCK-END
