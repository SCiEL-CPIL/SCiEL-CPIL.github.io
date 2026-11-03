"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (flat-top orientation at 0 deg)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _get_all_vertices(positions):
    """
    positions: (N, 3) array of (x, y, angle_deg)
    returns: (N, 6, 2) array of vertices
    """
    verts = np.empty((len(positions), 6, 2))
    for i, (x, y, ang) in enumerate(positions):
        verts[i] = _hex_vertices(x, y, ang, 1.0)
    return verts

def _min_enclosing_side(verts):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, 
    fixed orientation) that contains all vertices in 'verts'.
    verts: (N, 6, 2)
    """
    # Flatten vertices
    flat_verts = verts.reshape(-1, 2)
    
    # Normals to the sides of the outer hexagon
    # The outer hexagon sides are perpendicular to these normals.
    # Inradius R_in = S * sqrt(3) / 2.
    # Condition: dot(v, n) <= R_in for all v.
    # So S >= 2 * dot(v, n) / sqrt(3).
    
    normals = []
    for k in range(6):
        theta = math.pi / 6.0 + k * math.pi / 3.0
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    normals = np.array(normals) # (6, 2)
    
    # Project all vertices onto all normals
    # flat_verts: (M, 2), normals: (6, 2)
    # We want max over all v and n of (v . n)
    # Use matrix multiplication: (M, 2) @ (2, 6) -> (M, 6)
    projections = flat_verts @ normals.T
    max_proj = np.max(projections)
    
    S = 2.0 * max_proj / math.sqrt(3.0)
    return S

def _sat_overlap_depth(v1, v2):
    """
    Check if two convex polygons overlap using SAT.
    Returns (overlap_depth, axis) if they overlap, else (0, None).
    overlap_depth is the minimum distance to separate them.
    """
    min_overlap = float('inf')
    min_axis = None
    
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            # Project both polygons onto this normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Check for overlap
            if max1 < min2 or max2 < min1:
                # No overlap on this axis
                return 0, None
            
            # Calculate overlap depth on this axis
            # The overlap is the intersection of the two intervals
            # [min1, max1] and [min2, max2]
            # Intersection is [max(min1, min2), min(max1, max2)]
            # Length of intersection is the overlap depth?
            # No, we want the minimum shift to separate.
            # Shift = max1 - min2 (if poly1 is to the left) or max2 - min1 (if poly2 is to the left)
            # We want the minimum shift.
            # If max1 < min2, no overlap.
            # Overlap exists.
            # Depth = min(max1 - min2, max2 - min1) ?
            # Actually, if they overlap, the distance to separate is the smaller of the two "gaps" to the edges.
            # Gap 1: max1 - min2 (push poly1 left or poly2 right)
            # Gap 2: max2 - min1 (push poly2 left or poly1 right)
            # The minimum translation vector magnitude is min(Gap1, Gap2)?
            # Wait, if they are deeply embedded, max1 > min2 and max2 > min1.
            # The "overlap depth" in the sense of penetration is not simply the interval intersection length.
            # For SAT, if we want to push them apart, we project onto the axis.
            # The amount we need to move is the smaller of the distances to the nearest separating boundary.
            # dist1 = max1 - min2 (distance from max1 to min2? No, if max1 > min2, they overlap).
            # Let's define the separation distance.
            # If max1 < min2, sep = min2 - max1 > 0.
            # If max2 < min1, sep = min1 - max2 > 0.
            # If overlap, sep is negative.
            # sep = min(min2 - max1, min1 - max2)
            # We want the axis with the maximum sep (closest to 0, i.e., least penetration).
            
            sep1 = min2 - max1
            sep2 = min1 - max2
            sep = min(sep1, sep2)
            
            if sep > 1e-9:
                return 0, None # Separated
            
            # sep is negative (overlap)
            # We want the axis with the LARGEST sep (closest to 0)
            if sep > min_overlap:
                min_overlap = sep
                # Determine direction of axis
                # We want to push poly1 away from poly2.
                # If sep1 < sep2, it's easier to push poly1 left (along -normal) or poly2 right.
                # The normal points from edge i to j.
                # Let's just return the axis and depth. The solver will decide direction based on centers.
                min_axis = normal
                
    if min_overlap < -1e-9:
        # Overlap depth is -min_overlap
        return -min_overlap, min_axis
    return 0, None

def _resolve_overlaps_and_minimize(positions, steps=1000, step_size=0.01):
    """
    Optimize positions to minimize enclosing side while resolving overlaps.
    """
    best_S = float('inf')
    best_pos = positions.copy()
    
    # Calculate initial S
    verts = _get_all_vertices(positions)
    S = _min_enclosing_side(verts)
    # Check overlaps
    has_overlap = False
    for i in range(N):
        for j in range(i+1, N):
            depth, axis = _sat_overlap_depth(verts[i], verts[j])
            if depth > 1e-6:
                has_overlap = True
                break
        if has_overlap:
            break
    
    if not has_overlap and S < best_S:
        best_S = S
        best_pos = positions.copy()
        
    for step in range(steps):
        # Calculate gradients / forces
        forces = np.zeros_like(positions)
        
        # 1. Overlap resolution
        verts = _get_all_vertices(positions)
        for i in range(N):
            for j in range(i+1, N):
                depth, axis = _sat_overlap_depth(verts[i], verts[j])
                if depth > 1e-6:
                    # Push apart
                    # Determine direction
                    # Vector from center j to center i
                    c_i = positions[i, :2]
                    c_j = positions[j, :2]
                    d_vec = c_i - c_j
                    # Project d_vec onto axis
                    # If dot(d_vec, axis) > 0, axis points from j to i roughly?
                    # axis is normal to an edge.
                    # We want to move i along axis and j along -axis if they are overlapping.
                    # But we need to ensure we move them in the direction that separates them.
                    # The axis from SAT is the normal of an edge.
                    # If we project centers onto axis, the one with larger projection is "outside" in that direction.
                    # Let's just use the axis as is, and apply force proportional to depth.
                    # To be safe, check which side of the other's center they are.
                    dot_i = np.dot(c_i, axis)
                    dot_j = np.dot(c_j, axis)
                    
                    if dot_i > dot_j:
                        # i is "ahead" along axis. Push i further along axis, j back.
                        forces[i, :2] += depth * axis
                        forces[j, :2] -= depth * axis
                    else:
                        forces[i, :2] -= depth * axis
                        forces[j, :2] += depth * axis
                    
                    # Angular force? Ignore for now, linear only.
        
        # 2. Minimize enclosing side (Attract to center)
        # The vertex that defines the max projection is the one to pull.
        # For simplicity, pull all hexagons towards the origin with a force proportional to their distance.
        # This compresses the cluster.
        # We can make this force stronger if S is large.
        for i in range(N):
            # Force towards origin
            forces[i, :2] -= 0.05 * positions[i, :2]
            
        # Update positions
        positions += step_size * forces
        
        # Clamp angles? No, let them be free? 
        # The problem says "each inner hexagon may translate and rotate".
        # But my optimizer only moves x,y. It doesn't change angles.
        # Should I optimize angles too?
        # Yes, rotation can significantly reduce the bounding box.
        # Let's add a small random perturbation to angles or optimize them.
        # For now, let's fix angles to 0, 30, 60 etc? 
        # Or, let's optimize angles as well.
        # Gradient for angle is harder. 
        # Let's try randomizing angles slightly in each step? 
        # Or, just keep angles fixed at 0 for now and see if it works. 
        # Actually, the optimal packing of hexagons usually has them aligned (angle 0) or rotated by 30.
        # Let's assume angles are 0 for all hexagons to simplify the search space.
        # If this yields a good score, we can refine.
        
        # Calculate new S
        verts = _get_all_vertices(positions)
        S = _min_enclosing_side(verts)
        
        # Check overlaps
        has_overlap = False
        for i in range(N):
            for j in range(i+1, N):
                depth, axis = _sat_overlap_depth(verts[i], verts[j])
                if depth > 1e-6:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if not has_overlap:
            if S < best_S:
                best_S = S
                best_pos = positions.copy()
                
        # Decay step size?
        # step_size = max(0.001, step_size * 0.99)
        
    return best_S, best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_S = float('inf')
    best_positions = None
    
    # Random restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize positions in a random cluster
        # Start with a hexagonal grid pattern
        # Centers of touching hexagons are at distance sqrt(3)
        dist = math.sqrt(3.0)
        
        # Generate a set of 12 points in a triangular lattice
        points = []
        # Center
        points.append((0, 0))
        # Ring 1
        for i in range(6):
            ang = i * math.pi / 3.0
            points.append((dist * math.cos(ang), dist * math.sin(ang)))
        # Need 5 more. Ring 2?
        # Add some random points nearby
        for _ in range(5):
            # Random angle and distance
            ang = np.random.uniform(0, 2*math.pi)
            rad = np.random.uniform(dist, dist * 2)
            points.append((rad * math.cos(ang), rad * math.sin(ang)))
            
        positions = np.zeros((N, 3))
        for i in range(N):
            positions[i, 0] = points[i][0] + np.random.uniform(-0.1, 0.1)
            positions[i, 1] = points[i][1] + np.random.uniform(-0.1, 0.1)
            positions[i, 2] = 0.0  # Angle
            
        S, pos = _resolve_overlaps_and_minimize(positions, steps=2000, step_size=0.02)
        
        if S < best_S:
            best_S = S
            best_positions = pos.copy()
            
    # Final check and return
    if best_positions is None:
        # Fallback
        positions = np.zeros((N, 3))
        for i in range(N):
            positions[i, 0] = 3.0 * i
        best_S = 20.0
        best_positions = positions
        
    # Construct return value
    # inner: list of [x, y, angle_degrees]
    inner = []
    for p in best_positions:
        inner.append([p[0], p[1], p[2]])
        
    # Outer hexagon: center (0,0), side best_S, angle 0
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
