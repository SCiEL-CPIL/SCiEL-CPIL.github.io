"""
Strategy: Kinematic Chain Optimization for Hexagon Packing.

This approach parameterizes the 11 unit hexagons as a kinematic tree.
1. Topology: A spanning tree defines the adjacency. Each edge has a length L (>= min_sep)
   and an angle theta relative to the parent's orientation.
2. Global Transform: The entire cluster is rotated and translated.
3. Optimization: We optimize the edge lengths, angles, and global transform.
   The objective function is the outer side length plus a penalty for overlaps.
   Since the objective is differentiable (except at boundaries), we can use
   efficient gradient-based or quasi-Newton methods like Nelder-Mead.
4. Benefits: 
   - Reduced dimensionality: ~22 vars for tree edges + 3 for global = 25 vars 
     vs 33 vars for absolute coords + angles.
   - Natural constraints: The tree structure keeps the cluster connected, 
     preventing the "exploding" of the cluster into disjoint islands.
   - Breathing: Varying edge lengths allows the cluster to compress, 
     which is crucial for fitting into the tight outer hexagon.
"""

import time
import numpy as np
import math
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

# Minimum distance between centers of two non-overlapping unit hexagons
# For regular hexagons of side 1, the distance between parallel sides is sqrt(3).
# The minimum center distance when touching is sqrt(3).
MIN_SEP = math.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_overlap_fast(pos, ang, tol=1e-7):
    """
    Fast overlap check using bounding boxes first, then SAT.
    Returns True if overlap exists.
    """
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(N):
        verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        verts_list.append(verts)
        
    # Bounding box pre-check
    bboxes = []
    for v in verts_list:
        min_x, min_y = np.min(v, axis=0)
        max_x, max_y = np.max(v, axis=0)
        bboxes.append((min_x, min_y, max_x, max_y))
        
    for i in range(N):
        for j in range(i+1, N):
            # Check bbox overlap
            if (bboxes[i][0] > bboxes[j][2] or bboxes[i][2] < bboxes[j][0] or
                bboxes[i][1] > bboxes[j][3] or bboxes[i][3] < bboxes[j][1]):
                continue
                
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # SAT Check
            axes = []
            for k in range(6):
                edge1 = v1[(k+1)%6] - v1[k]
                normal = np.array([-edge1[1], edge1[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
                    
                edge2 = v2[(k+1)%6] - v2[k]
                normal = np.array([-edge2[1], edge2[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
            
            overlap = True
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                    overlap = False
                    break
            
            if overlap:
                return True
    return False

def _build_tree_topology():
    """
    Define a spanning tree for the 11 hexagons.
    Node 0 is the root.
    We use a 'flower' like topology but as a tree.
    0 is center.
    1-6 are children of 0.
    7-10 are children of 1, 3, 5, 7? No, let's just pick a valid tree.
    
    Let's use a specific compact tree:
    0 is center.
    1, 2, 3, 4, 5, 6 are neighbors of 0.
    7 is neighbor of 1.
    8 is neighbor of 3.
    9 is neighbor of 5.
    10 is neighbor of 7? Or maybe a different arrangement.
    
    Actually, for the optimizer to work well, the initial guess should be close to valid.
    We will define the tree structure (parent indices) and initialize edge lengths/angles
    based on the standard hexagonal lattice.
    """
    # Parent array: parent[i] is the parent of node i. parent[0] = -1.
    # Let's use a topology that allows for some flexibility.
    # 0: Center
    # 1, 3, 5: Alternating neighbors of 0
    # 2, 4, 6: Other neighbors of 0
    # 7: Neighbor of 1
    # 8: Neighbor of 3
    # 9: Neighbor of 5
    # 10: Neighbor of 7 (extending the chain)
    
    # This might be too sparse. Let's try a denser one.
    # 0
    # 1, 2, 3, 4, 5, 6 (all children of 0)
    # 7 (child of 1)
    # 8 (child of 2)
    # 9 (child of 3)
    # 10 (child of 4)
    
    parents = [-1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    
    # Initial angles relative to parent's "up" direction?
    # It's easier to define angles in global coordinates initially, 
    # then convert to relative if we wanted, but for optimization,
    # let's just optimize global angles of the edges in the tree frame?
    # No, kinematic chain implies local frames.
    # Let's simplify: We optimize the absolute positions of the nodes in the tree
    # by optimizing the edge vectors (dx, dy) for each edge.
    # This is effectively optimizing coordinates but with a tree structure constraint 
    # that we *initialize* well and *penalize* if the tree breaks (though we don't 
    # explicitly penalize tree breaking, we just use the tree to generate positions).
    # Wait, if we optimize edge vectors, the positions are determined.
    # The "breathing" is the magnitude of these vectors.
    
    return parents

def _generate_positions_from_tree(params, parents):
    """
    params: array of shape (10, 2) for edge vectors (dx, dy) of nodes 1..10.
            Node 0 is at (0,0) locally.
    returns: positions (11, 2), angles (11,)
    """
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # Node 0 is at origin
    pos[0] = [0, 0]
    ang[0] = 0.0
    
    for i in range(1, N):
        p = parents[i]
        # The edge vector is params[i-1]
        # The angle of the child is the angle of the edge + parent's angle?
        # To allow rotation, we should propagate angles.
        # Let's assume the edge vector is in the global frame of the parent?
        # No, that's rigid.
        # Let's say params[i-1] is (length, angle_global).
        # Then pos[i] = pos[p] + length * (cos(angle), sin(angle))
        # And ang[i] = angle_global + ang[p]? No, that couples rotation.
        
        # Simpler: Optimize (dx, dy) for each edge in the GLOBAL frame of the cluster.
        # This is just coordinate optimization but structured.
        # But we want "breathing". The length is sqrt(dx^2+dy^2).
        # Let's stick to (dx, dy) global.
        pos[i] = pos[p] + params[i-1]
        # Angle: We can optimize a global rotation for the whole cluster later.
        # For now, let's assume all hexagons have the same orientation relative to the cluster frame?
        # No, individual rotation is allowed.
        # Let's add an angle parameter for each node.
        # Total params: 10 edges * 2 (dx,dy) + 11 angles = 31 params.
        # Plus global transform: 2 (tx, ty) + 1 (rot) = 3 params.
        # Total 34 params.
        
    return pos, ang

def _objective(params, parents):
    """
    params structure:
    0-19: Edge vectors (dx, dy) for nodes 1..10 (10 edges * 2)
    20-30: Angles for nodes 0..10 (11 angles)
    31-33: Global transform (tx, ty, rot)
    """
    if time.time() > DEADLINE:
        return 1e10
        
    # Extract params
    edge_vecs = params[0:20].reshape(10, 2)
    angles_local = params[20:31]
    tx, ty, rot = params[31:34]
    
    # Compute local positions
    pos_local = np.zeros((N, 2))
    pos_local[0] = [0, 0]
    for i in range(1, N):
        p = parents[i]
        pos_local[i] = pos_local[p] + edge_vecs[i-1]
        
    # Apply global rotation
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c, -s], [s, c]])
    pos_global = pos_local @ rot_mat.T + np.array([tx, ty])
    
    # Apply angles
    # The angle of each hexagon is its local angle + global rotation
    angles_global = np.radians(angles_local) + rot
    
    # Check overlaps
    # For speed, we can do a quick check first
    # But we need a smooth penalty for optimization.
    # Let's use a soft penalty for overlaps.
    
    # Calculate min distance between centers for all pairs
    # If dist < MIN_SEP, there is potential overlap.
    # We need to check actual polygon overlap for accuracy, but that's not differentiable.
    # We will use a heuristic penalty: sum of max(0, MIN_SEP - dist)^2
    # This is not exact, but it guides the optimizer away from overlaps.
    # Then we do a final check.
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = pos_global[i, 0] - pos_global[j, 0]
            dy = pos_global[i, 1] - pos_global[j, 1]
            dist = math.hypot(dx, dy)
            if dist < MIN_SEP:
                # Penalize overlap
                # Use a steeper penalty
                penalty += (MIN_SEP - dist) ** 2 * 100.0
                
    # Calculate outer side length
    # We can fix outer center to (0,0) and angle 0 for the objective, 
    # as the global transform handles the positioning.
    # Wait, the global transform moves the cluster. 
    # The outer hexagon should be centered at the centroid of the cluster?
    # Or we can let the outer hexagon center be a variable?
    # The task says "outer hexagon center (x, y), side length and rotation".
    # To minimize side length, the outer hexagon should be centered at the centroid of the inner cluster.
    # And its rotation should be optimized.
    # Let's add outer rotation as a variable? 
    # Or we can just try a few outer rotations.
    # For simplicity in the objective, let's assume outer center is centroid, 
    # and we optimize outer rotation too?
    # Let's add outer_rot to params.
    
    # Actually, let's just compute the min side for the current config, 
    # assuming outer center is centroid and outer angle is 0.
    # We can refine this later.
    
    centroid = np.mean(pos_global, axis=0)
    side = _min_outer_side(pos_global, np.degrees(angles_global), centroid, 0.0)
    
    return side + penalty

def construct():
    global START_TIME, DEADLINE
    
    parents = _build_tree_topology()
    
    # Initial configuration based on standard hex lattice
    sqrt3 = math.sqrt(3)
    
    # Standard positions
    init_pos = np.zeros((N, 2))
    init_ang = np.zeros(N)
    
    # 0: Center
    init_pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        init_pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2 (4 hexagons)
    # Let's place them at specific spots
    # 7: near 1
    # 8: near 2
    # 9: near 3
    # 10: near 4
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        init_pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    # Build initial params
    # Edge vectors: pos[i] - pos[parent[i]]
    edge_vecs_init = np.zeros((10, 2))
    for i in range(1, N):
        p = parents[i]
        edge_vecs_init[i-1] = init_pos[i] - init_pos[p]
        
    angles_init = np.zeros(11) # All 0
    
    # Global transform: Identity
    tx, ty, rot = 0.0, 0.0, 0.0
    
    params_init = np.concatenate([
        edge_vecs_init.ravel(), 
        angles_init, 
        [tx, ty, rot]
    ])
    
    best_params = params_init.copy()
    best_side = _min_outer_side(init_pos, init_ang, (0,0), 0.0)
    best_pos = init_pos.copy()
    best_ang = init_ang.copy()
    
    # Optimize using Nelder-Mead
    # We will run multiple restarts with slight perturbations
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        # Perturb initial params
        noise = 0.1 if r == 0 else 0.5
        params = params_init + np.random.normal(0, noise, size=params_init.shape)
        
        # Ensure edge lengths are not too small (optional, penalty handles it)
        
        res = minimize(_objective, params, args=(parents,), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        # Evaluate result
        if res.success or res.fun < best_side:
            # Extract positions from best params
            edge_vecs = res.x[0:20].reshape(10, 2)
            angles_local = res.x[20:31]
            tx, ty, rot = res.x[31:34]
            
            pos_local = np.zeros((N, 2))
            pos_local[0] = [0, 0]
            for i in range(1, N):
                p = parents[i]
                pos_local[i] = pos_local[p] + edge_vecs[i-1]
                
            c, s = np.cos(rot), np.sin(rot)
            rot_mat = np.array([[c, -s], [s, c]])
            pos_global = pos_local @ rot_mat.T + np.array([tx, ty])
            
            angles_global = np.radians(angles_local) + rot
            
            # Check validity
            if not _check_overlap_fast(pos_global, np.degrees(angles_global)):
                centroid = np.mean(pos_global, axis=0)
                side = _min_outer_side(pos_global, np.degrees(angles_global), centroid, 0.0)
                
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
                    best_pos = pos_global.copy()
                    best_ang = np.degrees(angles_global).copy()
                    
    # Final refinement: Try to optimize outer rotation
    # We can fix inner positions and optimize outer rotation and center
    # But the current side calculation assumes outer center is centroid and angle 0.
    # Let's do a quick grid search for outer rotation to see if it helps.
    
    final_pos = best_pos
    final_ang = best_ang
    centroid = np.mean(final_pos, axis=0)
    
    best_final_side = best_side
    best_outer_rot = 0.0
    
    for rot_deg in np.linspace(0, 60, 13):
        side = _min_outer_side(final_pos, final_ang, centroid, rot_deg)
        if side < best_final_side:
            best_final_side = side
            best_outer_rot = rot_deg
            
    # Construct output
    inner_list = [[float(final_pos[i,0]), float(final_pos[i,1]), float(final_ang[i])] for i in range(N)]
    
    return inner_list, (float(centroid[0]), float(centroid[1])), float(best_final_side), float(best_outer_rot)

if __name__ == "__main__":
    construct()
