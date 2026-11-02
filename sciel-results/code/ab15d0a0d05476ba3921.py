import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 330 (flat top? No, pointy top usually 0 is right)
# Standard regular hexagon with side 1:
# Vertices at distance 1 from center.
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon centered at origin
# Edge midpoints are at angles 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto outward normals
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center=(0.0, 0.0), outer_side=None, outer_angle=0.0, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_required_side(centers, outer_center, outer_angle)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def generate_rigid_clusters():
    """
    Generate a set of rigid cluster shapes (relative positions) for the 12 hexagons.
    We use subsets of the hexagonal lattice that are compact.
    """
    clusters = []
    
    # Lattice basis vectors for unit hexagons (tangent to each other)
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3)
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    # Pattern 1: 3-4-3-2 (Compact block)
    # Coordinates in (i, j) lattice
    p1_coords = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Pattern 2: 2-4-4-2
    p2_coords = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Pattern 3: 1-3-4-3-1 (Diamond-ish)
    p3_coords = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    
    # Pattern 4: 4-4-4 (Rectangle)
    p4_coords = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    
    # Pattern 5: 3-3-3-3 (Taller rectangle)
    p5_coords = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    # Pattern 6: Hexagon of radius 2 with center removed and some others? 
    # Let's just use a dense 12-node patch from the infinite lattice
    # Center at (0,0). Neighbors:
    # Layer 0: (0,0)
    # Layer 1: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 6 nodes. Total 7.
    # Need 5 more from Layer 2.
    # Layer 2 nodes: (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2), (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)
    # Pick 5 that form a convex shape.
    p6_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1)
    ]
    
    for coords in [p1_coords, p2_coords, p3_coords, p4_coords, p5_coords, p6_coords]:
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], 0.0])
        # Center the cluster
        centers = np.array(centers)
        center_of_mass = np.mean(centers[:, :2], axis=0)
        centers[:, :2] -= center_of_mass
        clusters.append(centers)
        
    return clusters

def objective_cluster(params, cluster_shape):
    """
    Objective function for optimizing the rigid cluster's position and orientation.
    params: [x, y, angle_deg]
    cluster_shape: (N, 3) array of relative positions and angles (angles are 0 for rigid)
    """
    x, y, angle_deg = params
    angle_rad = np.radians(angle_deg)
    
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate and translate the cluster
    positions = cluster_shape[:, :2] @ R.T + np.array([x, y])
    
    # Construct full centers array for get_hex_vertices
    centers = np.zeros((N, 3))
    centers[:, :2] = positions
    centers[:, 2] = angle_deg # All hexagons have same orientation in rigid cluster
    
    # Calculate required side length
    side = compute_required_side(centers)
    
    # Add a penalty for overlap to ensure validity during optimization?
    # Since the cluster is rigid and valid by construction (lattice), 
    # we just need to minimize the bounding box.
    # However, if we allow rotation, the lattice remains valid.
    
    return side

def optimize_cluster(cluster_shape, initial_params=None):
    """
    Use scipy.optimize to find the best x, y, angle for a rigid cluster.
    """
    if initial_params is None:
        initial_params = [0.0, 0.0, 0.0]
    
    bounds = [(-10, 10), (-10, 10), (0, 360)]
    
    # Try multiple initial points to avoid local minima in the non-convex landscape
    best_x = None
    best_fval = float('inf')
    
    # Grid search for angle first, then refine x,y
    for angle_init in [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        # For a given angle, the optimal x,y is determined by centering the rotated cluster
        # But we can let the optimizer handle it.
        # To speed up, we can initialize x,y to 0,0 and let it converge.
        # Actually, for a given angle, the optimal center is the centroid of the rotated vertices?
        # No, the optimal center minimizes the max extent.
        
        start = [0.0, 0.0, float(angle_init)]
        
        try:
            res = minimize(objective_cluster, start, args=(cluster_shape,), method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000})
            if res.fun < best_fval:
                best_fval = res.fun
                best_x = res.x
        except Exception:
            pass
            
    return best_x, best_fval

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    clusters = generate_rigid_clusters()
    
    for cluster in clusters:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        try:
            # Optimize the rigid cluster
            # We can also try perturbing the cluster slightly to break symmetry?
            # The directive says "Lattice Translation Group" which implies rigid.
            # Let's stick to rigid for the structural change.
            
            optimal_params, cost = optimize_cluster(cluster)
            
            if optimal_params is not None:
                x, y, angle = optimal_params
                
                # Reconstruct the centers
                angle_rad = np.radians(angle)
                c = np.cos(angle_rad)
                s = np.sin(angle_rad)
                R = np.array([[c, -s], [s, c]])
                
                positions = cluster[:, :2] @ R.T + np.array([x, y])
                centers = np.zeros((N, 3))
                centers[:, :2] = positions
                centers[:, 2] = angle
                
                # Verify validity
                req_side = compute_required_side(centers)
                
                if check_validity(centers, outer_side=req_side):
                    if req_side < best_score:
                        best_score = req_side
                        best_result = (centers.tolist(), (0.0, 0.0), req_side, 0.0)
                        
        except Exception:
            continue

    if best_result is not None:
        return best_result
    
    # Fallback: Simple lattice configuration
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    p1_coords = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    centers = []
    for i, j in p1_coords:
        pos = i * a1 + j * a2
        centers.append([pos[0], pos[1], 0.0])
    
    centers = np.array(centers)
    center_of_mass = np.mean(centers[:, :2], axis=0)
    centers[:, :2] -= center_of_mass
    
    required_side = compute_required_side(centers)
    return centers.tolist(), (0.0, 0.0), required_side, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
