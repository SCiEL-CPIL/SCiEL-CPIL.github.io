# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Strategy:
    1. Initialize with a symmetric subset of the hexagonal lattice (1 center, 6 ring 1, 5 ring 2).
    2. Use Differential Evolution for global search to minimize outer hexagon size s.
    3. Use smooth penalty functions for collision and containment.
    4. Polish with L-BFGS-B.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return verts

    def compute_s(inner_centers):
        """
        Computes the minimum side length s of a regular hexagon (center 0, angle 0)
        that contains all inner hexagons (side 1, angle 0).
        """
        normals = []
        for i in range(6):
            a = math.pi/6 + i * math.pi / 3
            normals.append((math.cos(a), math.sin(a)))
        
        max_proj = 0.0
        for cx, cy in inner_centers:
            verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in verts:
                for n in normals:
                    proj = v[0]*n[0] + v[1]*n[1]
                    if proj > max_proj:
                        max_proj = proj
        
        return max_proj * 2.0 / sqrt3

    def smooth_collision_penalty(verts1, verts2):
        """
        Computes a smooth penalty based on the minimum distance between two convex polygons.
        Uses the Separating Axis Theorem to find the minimum overlap.
        If separated, penalty is 0. If overlapping, penalty is overlap^2.
        """
        axes = []
        for i in range(6):
            p1 = np.array(verts1[i])
            p2 = np.array(verts1[(i+1)%6])
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        for i in range(6):
            p1 = np.array(verts2[i])
            p2 = np.array(verts2[(i+1)%6])
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        
        min_overlap = float('inf')
        for axis in axes:
            dots1 = [np.dot(np.array(v), axis) for v in verts1]
            dots2 = [np.dot(np.array(v), axis) for v in verts2]
            min1, max1 = min(dots1), max(dots1)
            min2, max2 = min(dots2), max(dots2)
            
            # Calculate separation on this axis
            # If max1 < min2, separation is min2 - max1
            # If max2 < min1, separation is min1 - max2
            # Else, overlap is min(max1, max2) - max(min1, min2)
            
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                sep = -(min(max1, max2) - max(min1, min2))
            
            # We want the minimum separation (which is negative if overlapping)
            # Actually, we want to know if they are separated.
            # If any axis has sep > 0, they are separated.
            # The "overlap" is the minimum of the separations? No.
            # If they overlap, ALL axes have sep < 0 (overlap).
            # The "depth" of overlap is the minimum of the negative separations?
            # No, the depth is the maximum of the overlaps?
            # For SAT, if they overlap, the minimum separation distance is the minimum of the overlaps?
            # Actually, the minimum distance between two convex polygons is the minimum of the separations on all axes.
            # If min_sep > 0, distance is min_sep.
            # If min_sep < 0, they overlap by -min_sep.
            
            if sep < min_overlap:
                min_overlap = sep
        
        if min_overlap > 0:
            return 0.0
        else:
            return min_overlap * min_overlap

    def containment_penalty(inner_centers, s):
        """
        Penalty if any inner vertex is outside the outer hexagon.
        """
        outer_verts = get_hex_vertices(0, 0, s, 0)
        penalty = 0.0
        
        for cx, cy in inner_centers:
            inner_verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in inner_verts:
                for i in range(6):
                    p1 = np.array(outer_verts[i])
                    p2 = np.array(outer_verts[(i+1)%6])
                    edge = p2 - p1
                    vec = np.array(v) - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    if cross < 0:
                        penalty += cross * cross
        
        return penalty

    def objective(params):
        inner_centers = params.reshape(12, 2)
        
        # Compute s
        s = compute_s(inner_centers)
        
        # Compute collision penalties
        collision_pen = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                verts_i = get_hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, 0)
                verts_j = get_hex_vertices(inner_centers[j][0], inner_centers[j][1], 1.0, 0)
                collision_pen += smooth_collision_penalty(verts_i, verts_j)
        
        # Compute containment penalty
        cont_pen = containment_penalty(inner_centers, s)
        
        # Weighted sum
        lambda1 = 50.0
        lambda2 = 50.0
        
        return s + lambda1 * collision_pen + lambda2 * cont_pen

    def validate_config(inner_centers, s):
        """
        Strict validation.
        """
        # Check collisions
        for i in range(12):
            for j in range(i+1, 12):
                verts_i = get_hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, 0)
                verts_j = get_hex_vertices(inner_centers[j][0], inner_centers[j][1], 1.0, 0)
                
                axes = []
                for k in range(6):
                    p1 = np.array(verts_i[k])
                    p2 = np.array(verts_i[(k+1)%6])
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    if np.linalg.norm(normal) > 1e-9:
                        axes.append(normal / np.linalg.norm(normal))
                for k in range(6):
                    p1 = np.array(verts_j[k])
                    p2 = np.array(verts_j[(k+1)%6])
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    if np.linalg.norm(normal) > 1e-9:
                        axes.append(normal / np.linalg.norm(normal))
                
                separated = False
                for axis in axes:
                    dots_i = [np.dot(np.array(v), axis) for v in verts_i]
                    dots_j = [np.dot(np.array(v), axis) for v in verts_j]
                    min_i, max_i = min(dots_i), max(dots_i)
                    min_j, max_j = min(dots_j), max(dots_j)
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        separated = True
                        break
                if not separated:
                    return False
        
        # Check containment
        outer_verts = get_hex_vertices(0, 0, s, 0)
        for cx, cy in inner_centers:
            inner_verts = get_hex_vertices(cx, cy, 1.0, 0)
            for v in inner_verts:
                for i in range(6):
                    p1 = np.array(outer_verts[i])
                    p2 = np.array(outer_verts[(i+1)%6])
                    edge = p2 - p1
                    vec = np.array(v) - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    if cross < -1e-6:
                        return False
        
        return True

    # 1. Initial Configuration: Symmetric subset of the 19-hexagon lattice
    def get_initial_config():
        centers = []
        # Center
        centers.append((0.0, 0.0))
        
        # Ring 1 (6 hexagons)
        for i in range(6):
            a = i * math.pi / 3
            x = sqrt3 * math.cos(a)
            y = sqrt3 * math.sin(a)
            centers.append((x, y))
            
        # Ring 2 (12 hexagons in full lattice, we pick 5 to make 12 total)
        # To maintain symmetry, we can't just pick 5 arbitrarily.
        # Let's pick 5 that are spaced out.
        # Angles for Ring 2 in lattice:
        # The lattice points at distance 2*sqrt(3) are at angles:
        # 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330?
        # Actually, the second ring of a hexagonal lattice has 12 points.
        # Let's generate them properly.
        # Basis vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
        # Points at "radius" 2:
        # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) in lattice coords?
        # No, let's just use polar angles.
        # The 12 points of the second ring are at angles:
        # 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 degrees?
        # Distance is 2*sqrt(3).
        # Let's pick 5 points. To be symmetric, maybe we break symmetry slightly 
        # or just use a valid non-symmetric start and let the optimizer fix it?
        # Better: Use a 3-4-3-2 strip? No, stick to radial.
        # Let's pick angles: 0, 72, 144, 216, 288 (pentagonal symmetry? No, hexagon is 6-fold).
        # Let's pick 5 points from the 12 available.
        # Indices 0, 2, 4, 6, 8 (every other one)? That's 6 points.
        # We need 5.
        # Let's just pick 5 random ones from the 12, but keep the first 7 fixed?
        # No, let's use a known good configuration:
        # 1 center, 6 ring 1, 5 ring 2.
        # To minimize s, the 5 ring 2 hexagons should be placed to "fill in" the gaps 
        # or extend the shape minimally.
        
        # Let's initialize with the 12 points of a "truncated" hexagon.
        # Actually, just use the first 5 points of the second ring.
        ring2_angles = [i * math.pi / 6 for i in range(12)]
        for i in range(5):
            a = ring2_angles[i]
            x = 2 * sqrt3 * math.cos(a)
            y = 2 * sqrt3 * math.sin(a)
            centers.append((x, y))
            
        return np.array(centers)

    init = get_initial_config()
    
    best_val = 0.0
    best_inner = None
    best_s = float('inf')
    
    # Run Differential Evolution
    # Bounds: x, y in [-10, 10]
    bounds = [(-10, 10)] * 24
    
    try:
        # DE is slow, so limit iterations/popsize
        result_de = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-4, seed=42)
        
        if result_de.success or result_de.fun < objective(init):
            inner_centers = result_de.x.reshape(12, 2)
            s = compute_s(inner_centers)
            if validate_config(inner_centers, s):
                val = 1.0 / s
                if val > best_val:
                    best_val = val
                    best_inner = inner_centers
                    best_s = s
            else:
                # Try to polish the DE result with L-BFGS-B to fix constraints
                try:
                    result_lbfgs = minimize(objective, result_de.x, method='L-BFGS-B', options={'maxiter': 200})
                    inner_centers = result_lbfgs.x.reshape(12, 2)
                    s = compute_s(inner_centers)
                    if validate_config(inner_centers, s):
                        val = 1.0 / s
                        if val > best_val:
                            best_val = val
                            best_inner = inner_centers
                            best_s = s
                except:
                    pass
    except Exception as e:
        pass
    
    # Also try L-BFGS-B from the initial config
    try:
        result_lbfgs = minimize(objective, init.flatten(), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        inner_centers = result_lbfgs.x.reshape(12, 2)
        s = compute_s(inner_centers)
        if validate_config(inner_centers, s):
            val = 1.0 / s
            if val > best_val:
                best_val = val
                best_inner = inner_centers
                best_s = s
    except Exception as e:
        pass

    # If no valid configuration found, use the initial config (which is valid)
    if best_inner is None:
        best_inner = init
        best_s = compute_s(init)
        best_val = 1.0 / best_s

    # Prepare return values
    inner_arr = np.zeros((12, 3))
    for i in range(12):
        inner_arr[i][0] = best_inner[i][0]
        inner_arr[i][1] = best_inner[i][1]
        inner_arr[i][2] = 0.0

    outer_arr = np.array([0.0, 0.0, 0.0])
    
    return inner_arr, outer_arr, best_s

# EVOLVE-BLOCK-END
