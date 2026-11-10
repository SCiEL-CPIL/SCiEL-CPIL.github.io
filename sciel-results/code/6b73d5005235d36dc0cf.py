# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
from itertools import combinations, product

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a simulated annealing approach.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    # Constants
    sqrt3 = math.sqrt(3)
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return verts

    # Helper: Get edge normals (outward) for a convex polygon
    def get_axes(verts):
        axes = []
        for i in range(len(verts)):
            p1 = np.array(verts[i])
            p2 = np.array(verts[(i+1)%len(verts)])
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        return axes

    def project_polygon(verts, axis):
        dots = [np.dot(np.array(v), axis) for v in verts]
        return min(dots), max(dots)

    def are_hexagons_separated(verts1, verts2, eps=1e-6):
        axes = get_axes(verts1) + get_axes(verts2)
        
        for axis in axes:
            min1, max1 = project_polygon(verts1, axis)
            min2, max2 = project_polygon(verts2, axis)
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def is_point_in_hexagon(point, hex_verts, eps=1e-6):
        px, py = point
        for i in range(len(hex_verts)):
            p1 = np.array(hex_verts[i])
            p2 = np.array(hex_verts[(i+1)%len(hex_verts)])
            edge = p2 - p1
            vec = np.array([px, py]) - p1
            cross = edge[0]*vec[1] - edge[1]*vec[0]
            if cross < -eps:
                return False
        return True

    def is_inside_outer(inner_verts, outer_verts, eps=1e-6):
        for v in inner_verts:
            if not is_point_in_hexagon(v, outer_verts, eps):
                return False
        return True

    def min_s_for_hexagons(inner_hexes, angle_deg=0):
        all_verts = []
        for cx, cy, ang in inner_hexes:
            verts = get_hex_vertices(cx, cy, 1.0, ang)
            all_verts.extend(verts)
        
        if not all_verts:
            return 0.0
            
        normals = []
        base_angle = math.radians(angle_deg)
        for i in range(6):
            normal_angle = base_angle + math.pi/6 + i * math.pi / 3
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))
            
        max_proj = 0.0
        for v in all_verts:
            for n in normals:
                proj = v[0]*n[0] + v[1]*n[1]
                if proj > max_proj:
                    max_proj = proj
        
        return max_proj * 2.0 / sqrt3

    def validate_config(inner_hexes, s, outer_center=(0,0), outer_angle=0):
        verts_list = []
        for cx, cy, ang in inner_hexes:
            verts = get_hex_vertices(cx, cy, 1.0, ang)
            verts_list.append(verts)
        
        # Check collisions
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                if not are_hexagons_separated(verts_list[i], verts_list[j], 1e-6):
                    return False, 0.0
        
        # Check containment
        outer_verts = get_hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
        for verts in verts_list:
            for v in verts:
                if not is_point_in_hexagon(v, outer_verts, 1e-6):
                    return False, 0.0
                
        return True, 1.0/s

    def score_config(inner_hexes):
        centers = np.array([(cx, cy) for cx, cy, _ in inner_hexes])
        centroid = np.mean(centers, axis=0)
        shifted_inner = [(cx - centroid[0], cy - centroid[1], ang) for cx, cy, ang in inner_hexes]
        
        s = min_s_for_hexagons(shifted_inner, angle_deg=0)
        
        valid, val = validate_config(shifted_inner, s, outer_center=(0,0), outer_angle=0)
        if valid:
            return val, shifted_inner, s
        else:
            return -1, None, None

    # Generate initial configuration: Hexagonal cluster
    # 12 hexes. 
    # A 3x4 rectangle of hexes? Or a cluster?
    # Let's try a "flower" of 7 plus 5 more?
    # Or just a dense lattice patch.
    # Lattice vectors for hex grid (side 1, angle 0):
    # v1 = (2.5, sqrt(3)/2 * 2) = (2.5, sqrt(3))
    # v2 = (0, sqrt(3)) ? No.
    # Centers:
    # (0,0)
    # (2.5, sqrt(3)/2)
    # (1.25, sqrt(3))
    # etc.
    
    # Let's just use a rectangular grid of 4x3 with hex spacing.
    # x spacing: 2.5
    # y spacing: sqrt(3)
    # Offset odd rows by 1.25
    
    def generate_initial():
        init = []
        # 4 columns, 3 rows
        for r in range(3):
            for c in range(4):
                x = c * 2.5
                y = r * sqrt3
                if r % 2 == 1:
                    x += 1.25
                init.append([x, y, 0.0])
        return init

    best_val = 0.0
    best_inner = None
    best_s = float('inf')
    
    def simulated_annealing(init_inner, max_iter=5000):
        current_inner = [list(h) for h in init_inner]
        current_val, _, _ = score_config(current_inner)
        if current_val <= 0:
            # Try to fix initial config
            # If invalid, perturb randomly until valid
            for _ in range(100):
                for i in range(12):
                    current_inner[i][0] += random.uniform(-1, 1)
                    current_inner[i][1] += random.uniform(-1, 1)
                current_val, _, _ = score_config(current_inner)
                if current_val > 0:
                    break
            if current_val <= 0:
                return -1, None, None

        best_local_val = current_val
        best_local_inner = [h[:] for h in current_inner]
        
        T = 1.0
        cooling = 0.995
        step_size = 0.5
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 11)
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            
            new_inner = [h[:] for h in current_inner]
            new_inner[idx][0] += dx
            new_inner[idx][1] += dy
            
            new_val, _, _ = score_config(new_inner)
            
            if new_val > current_val:
                current_inner = new_inner
                current_val = new_val
                if current_val > best_local_val:
                    best_local_val = current_val
                    best_local_inner = [h[:] for h in current_inner]
            else:
                # Accept worse state with probability exp(-delta / T)
                # If new_val is -1 (invalid), delta is large, prob ~ 0
                if new_val > 0:
                    delta = current_val - new_val
                    if random.random() < math.exp(-delta / T):
                        current_inner = new_inner
                        current_val = new_val
                
            T *= cooling
            if T < 0.01:
                break
                
        return best_local_val, best_local_inner, min_s_for_hexagons(best_local_inner)

    # Run SA on initial config
    init_cfg = generate_initial()
    val, inner, s = simulated_annealing(init_cfg)
    if val > best_val:
        best_val = val
        best_inner = inner
        best_s = s

    # Run SA on a few other random restarts if time permits
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        # Random perturbation of the best found so far
        if best_inner is not None:
            rand_inner = [h[:] for h in best_inner]
            for h in rand_inner:
                h[0] += random.uniform(-2, 2)
                h[1] += random.uniform(-2, 2)
        else:
            rand_inner = generate_initial()
            
        val, inner, s = simulated_annealing(rand_inner)
        if val > best_val:
            best_val = val
            best_inner = inner
            best_s = s

    # Finalize
    if best_inner is None:
        best_inner = generate_initial()
        val, _, _ = score_config(best_inner)
        best_val = val
        best_s = min_s_for_hexagons(best_inner)

    # Recalculate shifted version of best_inner to ensure consistency
    centers = np.array([(cx, cy) for cx, cy, _ in best_inner])
    centroid = np.mean(centers, axis=0)
    final_inner = [(cx - centroid[0], cy - centroid[1], ang) for cx, cy, ang in best_inner]
    
    # Verify final_inner is valid with s
    final_val, _, final_s = score_config(final_inner)
    if final_val <= 0:
        # Fallback
        final_inner = [list(h) for h in generate_initial()]
        final_val, _, final_s = score_config(final_inner)
        if final_val <= 0:
            final_s = 10.0
            final_val = 0.1
            
    inner_ret = np.array(final_inner)
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, final_s

# EVOLVE-BLOCK-END
