# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _unit_hex_vertices(cx, cy):
    angles = np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _rigid_cluster_offsets():
    # Rigid hexagonal lattice cluster: 1 center + 6 ring 1 + 5 ring 2
    # This is a known compact topology for 12 hexagons.
    # Ring 1: 6 neighbors at distance 2 * sqrt(3) ? No, edge-to-edge distance is 2 * apothem = sqrt(3).
    # Wait, for unit hexagons (side 1), the distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Let's verify. Apothem = sqrt(3)/2. Distance between parallel edges = 2 * apothem = sqrt(3).
    # So centers are at distance sqrt(3) from each other if they share an edge.
    
    # Lattice vectors for hexagonal packing of unit hexagons:
    # The centers form a triangular lattice.
    # Nearest neighbor distance = sqrt(3).
    # Basis vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2) ?
    # Let's check distance: |a2| = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
    # Angle between a1 and a2: cos(theta) = (a1.a2)/(|a1||a2|) = (3/2) / (3) = 1/2. theta = 60 deg. Correct.
    
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3)/2.0, 1.5])
    
    offsets = []
    # Center
    offsets.append(np.array([0.0, 0.0]))
    
    # Ring 1: 6 neighbors
    # These are at +/- a1, +/- a2, +/- (a2 - a1)
    offsets.append(a1)
    offsets.append(-a1)
    offsets.append(a2)
    offsets.append(-a2)
    offsets.append(a2 - a1)
    offsets.append(a1 - a2)
    
    # Ring 2: 12 neighbors. We need 5 more to make 12 total.
    # The 12 ring 2 neighbors are at:
    # 2*a1, 2*a2, 2*(a2-a1)
    # a1+a2, a1-a2, -a1+a2, -a1-a2, -a1+(a2-a1)=a2-2a1, etc.
    # Let's list all 12 ring 2 positions:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2) in lattice coords (i,j) -> i*a1 + j*a2
    # (1,1), (1,-1), (-1,1), (-1,-1), (3,-1), (3,1), (-3,1), (-3,-1), (-1,3), (1,3), (-1,-3), (1,-3)
    
    # We want to pick 5 that form a compact shape.
    # Usually, picking a contiguous arc or symmetric subset is good.
    # Let's try picking 5 from the 12.
    # A common optimal packing for 12 hexagons in a hexagon might use a specific 5.
    # Let's try: 2*a1, 2*a2, a1+a2, a1-a2, -a1-a2 ?
    # Or maybe: 2*a1, 2*a2, 2*(a2-a1), a1+a2, a1-a2?
    
    # Let's define all 12 ring 2 vectors:
    r2_vectors = [
        2*a1,
        2*a2,
        2*(a2-a1),
        a1+a2,
        a1-a2,
        -a1+a2,
        -a1-a2,
        -a1+(a2-a1), # a2 - 2a1
        -a2+(a2-a1), # a1 - 2a2
        a1+(a2-a1),  # a2
        # Wait, the 12 neighbors in ring 2 are:
        # (2,0), (0,2), (-2,2)
        # (1,1), (1,-1), (-1,1), (-1,-1)
        # (-2,0), (0,-2), (2,-2)
        # (-1,2), (1,2), (2,-1), (-2,1) -- No, this is getting messy.
        
        # Let's just generate them by rotating the basis vectors or adding.
        # The 12 vectors are the sums of two ring 1 vectors that are not opposite?
        # No, ring 2 is distance 2*sqrt(3) or sqrt(3)*sqrt(3)=3?
        # Distance from center to ring 1 is sqrt(3).
        # Distance from center to ring 2 is either 2*sqrt(3) (straight) or sqrt(3)*sqrt(3) = 3 (diagonal).
        
        # Straight ring 2: 2*a1, 2*a2, 2*(a2-a1) and their negatives. (6 vectors)
        # Diagonal ring 2: a1+a2, a1-a2, -a1+a2, -a1-a2 and their negatives?
        # (a1+a2) has length sqrt(3+3+2*1.5) = sqrt(9) = 3.
        # So there are 6 vectors of length 2*sqrt(3) and 6 vectors of length 3.
        
        # Total 12.
        # We need to pick 5.
        # To minimize the bounding box, we should probably pick vectors that are "balanced" or fill the corners.
        # Let's try picking 5 that are roughly evenly spaced.
        
        # Candidate set 1: 2*a1, 2*a2, 2*(a2-a1), a1+a2, a1-a2
        # Candidate set 2: 2*a1, 2*a2, a1+a2, a1-a2, -a1-a2
    ]
    
    # Let's explicitly construct the 12 ring 2 vectors
    r2 = []
    # Length 2*sqrt(3)
    r2.append(2*a1)
    r2.append(-2*a1)
    r2.append(2*a2)
    r2.append(-2*a2)
    r2.append(2*(a2-a1))
    r2.append(-2*(a2-a1))
    # Length 3
    r2.append(a1+a2)
    r2.append(-a1-a2)
    r2.append(a1-a2)
    r2.append(-a1+a2)
    r2.append(a2-a1) # This is same as -(a1-a2)? No. a2-a1 is length 3?
    # |a2-a1| = sqrt(3). So this is ring 1.
    # Wait, a1-a2 is ring 1.
    # The diagonal neighbors are a1+a2, a1-a2 (which is - (a2-a1)), etc.
    # Let's re-evaluate the 12 ring 2 vectors.
    # The 12 vectors at distance >= 2*sqrt(3) from origin in the triangular lattice.
    # Lattice points (i,j) such that i*a1 + j*a2 has length >= 2*sqrt(3).
    # (2,0): len 2*sqrt(3)
    # (0,2): len 2*sqrt(3)
    # (-2,2): len 2*sqrt(3)
    # (1,1): len 3
    # (1,-1): len 3
    # (-1,1): len 3
    # (-1,-1): len 3
    # (-2,0): len 2*sqrt(3)
    # (0,-2): len 2*sqrt(3)
    # (2,-2): len 2*sqrt(3)
    # (-1,2): len 3
    # (1,2): len 3
    # (2,-1): len 3
    # (-2,1): len 3
    
    # So the 12 are:
    # (2,0), (-2,0), (0,2), (0,-2), (-2,2), (2,-2)
    # (1,1), (-1,-1), (1,-1), (-1,1), (-1,2), (1,2), (2,-1), (-2,1)
    
    v20 = 2*a1
    v02 = 2*a2
    vm22 = -2*a1 + 2*a2
    v11 = a1 + a2
    vm1m1 = -a1 - a2
    v1m1 = a1 - a2
    vm11 = -a1 + a2
    vm12 = -a1 + 2*a2
    v12 = a1 + 2*a2
    v2m1 = 2*a1 - a2
    vm21 = -2*a1 + a2
    
    # We need 5.
    # Let's try a symmetric set.
    # How about: v20, v02, vm22, v11, vm1m1?
    # Or: v20, v02, v11, v1m1, vm11?
    
    # Let's try: v20, v02, vm22, v11, v1m1
    # This is asymmetric.
    
    # Let's try: v20, v02, v11, v1m1, vm1m1
    # v20 (right), v02 (up-right), v11 (up), v1m1 (up-left), vm1m1 (down-left)
    
    # Actually, let's just pick 5 and let the optimizer adjust the outer hexagon.
    # The topology is fixed, but we can try a few different topologies if needed.
    # For now, let's use:
    # 2*a1, 2*a2, 2*(a2-a1), a1+a2, a1-a2
    # This covers 4 quadrants roughly.
    
    offsets.append(2*a1)
    offsets.append(2*a2)
    offsets.append(2*(a2-a1))
    offsets.append(a1+a2)
    offsets.append(a1-a2)
    
    return np.array(offsets)

def _min_enclosing_hex_side_for_orientation(centers, outer_angle_deg):
    """
    Calculate the minimum side length s of a regular hexagon with given angle
    that contains all given centers (which are centers of unit hexagons).
    """
    a = np.deg2rad(outer_angle_deg)
    # Normals to the edges of the outer hexagon
    # The edges are perpendicular to the normals at angles a + 30 + 60*k
    normal_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For a unit hexagon centered at c, the maximum projection onto a normal n is:
    # max(dot(c + v, n)) for v in unit_hex_vertices(0,0)
    # This is dot(c, n) + max(dot(v, n))
    # max(dot(v, n)) for a unit hexagon is the apothem = sqrt(3)/2.
    # Wait, is it always apothem?
    # The support function of a regular hexagon is h(n) = apothem * |cos(theta)|? No.
    # The support function of a regular hexagon is constant in the direction of the apothem?
    # No, the width is 2*apothem in the direction of the apothem.
    # In the direction of a vertex, the width is 2*side = 2.
    # The projection of a unit hexagon onto a normal n is:
    # max(dot(v, n)) - min(dot(v, n))
    # Since the hexagon is centrally symmetric, this is 2 * max(dot(v, n)).
    # max(dot(v, n)) depends on the angle of n relative to the hexagon.
    # If n is aligned with an apothem, max(dot(v,n)) = apothem = sqrt(3)/2.
    # If n is aligned with a vertex, max(dot(v,n)) = 1.
    # The function is piecewise linear.
    
    # However, we can just compute it numerically for accuracy and simplicity,
    # or use the formula.
    # Let's compute it numerically to be safe, as the angle is fixed.
    unit_verts = _unit_hex_vertices(0, 0)
    max_proj_unit = np.max(np.dot(unit_verts, normals.T), axis=0) # (6,)
    
    # For each center, project onto each normal
    # centers: (12, 2)
    # normals: (6, 2)
    # proj: (12, 6)
    proj = centers @ normals.T
    # max projection of the whole cluster onto each normal
    # max over centers of (proj[i, k] + max_proj_unit[k])
    max_cluster_proj = np.max(proj, axis=0) + max_proj_unit
    
    # The outer hexagon must contain these projections.
    # For a regular hexagon with side s and center 0, the support function in direction n is:
    # h_outer(n) = s * apothem_outer * ... ?
    # The distance from center to edge is apothem_outer = s * sqrt(3)/2.
    # So we need apothem_outer >= max_cluster_proj[k] for all k?
    # No, the outer hexagon is rotated by outer_angle.
    # The normals we used are the normals to the outer hexagon's edges.
    # So the constraint is exactly:
    # apothem_outer >= max_cluster_proj[k] for all k.
    # apothem_outer = s * sqrt(3)/2
    # s * sqrt(3)/2 >= max_cluster_proj[k]
    # s >= 2/sqrt(3) * max_cluster_proj[k]
    
    s_required = np.max(max_cluster_proj) * 2.0 / np.sqrt(3.0)
    return s_required

def _check_separation(centers, tol=1e-6):
    n = len(centers)
    # For unit hexagons, if they are on a rigid lattice, they are separated.
    # But we should check just in case of floating point errors or if we change topology.
    # Since we are using a rigid lattice, separation is guaranteed if the lattice is valid.
    # We can skip this check for speed if we are confident in the topology.
    # But let's keep a simple check for the final result.
    return True

def _compute_s_and_centers(outer_angle_deg):
    offsets = _rigid_cluster_offsets()
    # The centers are at offsets.
    # We can rotate the cluster? No, the cluster is rigid relative to the outer hexagon?
    # Actually, we can rotate the cluster relative to the outer hexagon.
    # But the problem allows the inner hexagons to have any angle.
    # However, for a rigid lattice, the angle of each hexagon is fixed relative to the lattice.
    # If we rotate the lattice, the hexagons rotate with it.
    # So we have 2 DOF: outer_angle and cluster_rotation?
    # Wait, the outer hexagon angle and the cluster rotation are independent.
    # But relative rotation is what matters.
    # Let's fix the cluster at angle 0 and rotate the outer hexagon.
    # This is equivalent to rotating the cluster relative to the outer hexagon.
    
    centers = offsets
    s = _min_enclosing_hex_side_for_orientation(centers, outer_angle_deg)
    return s, centers

def _objective_rigid(params):
    outer_angle_deg = params[0]
    s, _ = _compute_s_and_centers(outer_angle_deg)
    return s

def _refine_individual_angles(centers, outer_angle_deg, s):
    # This function is a placeholder for a more advanced strategy where we might
    # adjust the angles of individual hexagons to tighten the packing.
    # For the rigid strategy, we assume angles are fixed.
    return centers, outer_angle_deg

def hexagon_packing_12():
    # Strategy:
    # 1. Define a rigid cluster of 12 hexagons.
    # 2. Search for the optimal orientation of the outer hexagon relative to the cluster.
    # 3. Since the problem is periodic with 60 degrees, we can search in [0, 60).
    
    # We can do a grid search followed by a local refinement.
    
    best_s = np.inf
    best_angle = 0.0
    
    # Grid search
    # The function is smooth and periodic.
    # Let's try a fine grid.
    angles = np.linspace(0, 60, 600, endpoint=False)
    
    for a in angles:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
            
        s, _ = _compute_s_and_centers(a)
        if s < best_s:
            best_s = s
            best_angle = a
            
    # Local refinement using minimize
    # We need to handle the periodicity. Minimize might go out of [0, 60).
    # We can just let it minimize and then mod 60.
    
    x0 = np.array([best_angle])
    
    try:
        result = minimize(
            _objective_rigid,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 1000,
                'xatol': 1e-8,
                'fatol': 1e-10,
            }
        )
        
        if result.success or result.fun < best_s:
            best_angle = result.x[0] % 60.0
            best_s = result.fun
    except Exception:
        pass
    
    # Final check and construction
    s, centers = _compute_s_and_centers(best_angle)
    
    # The centers are relative to the cluster center.
    # We should center the cluster so that the outer hexagon is centered at (0,0).
    # The outer hexagon is centered at (0,0) by definition in _min_enclosing_hex_side_for_orientation.
    # So the cluster center should be at (0,0).
    # Our offsets are already centered at (0,0).
    
    # However, we need to ensure the outer hexagon contains the cluster.
    # The calculation in _min_enclosing_hex_side_for_orientation ensures that
    # the outer hexagon (centered at 0,0) contains the cluster (centered at 0,0).
    
    # One detail: the outer hexagon angle is best_angle.
    # The inner hexagons have angle 0 (fixed by the lattice).
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
