import numpy as np
import time
import random

N = 11

def _hex_vertices(center, angle, size=1.0):
    """
    Returns vertices of a regular hexagon with given center, angle (radians), and size.
    """
    angles = angle + np.arange(6) * np.pi / 3
    x = center[0] + size * np.cos(angles)
    y = center[1] + size * np.sin(angles)
    return np.column_stack((x, y))

def _min_width_hexagon(vertices, num_angles=18):
    """
    Calculate the minimum side length of a hexagon enclosing the given vertices.
    vertices: (M, 2) array of points.
    Returns min side length.
    """
    min_side = float('inf')
    # The hexagon orientation is periodic with pi/3.
    # We scan phi in [0, pi/3)
    for i in range(num_angles):
        phi = (i / num_angles) * (np.pi / 3)
        # Normals to the 3 pairs of parallel sides
        # Side normals are at phi, phi + pi/3, phi + 2pi/3
        # Wait, the normals to the sides of a hexagon with one vertex at angle phi?
        # If vertex is at phi, side normal is at phi + pi/6.
        # Let's define orientation by the angle of the first vertex.
        # If we rotate the container by phi, the normals are at phi + pi/6 + k*pi/3.
        
        # Actually, the width in a direction theta is invariant to the specific vertex alignment,
        # as long as we check the 3 independent directions of the hexagon's symmetry.
        # The 3 directions are separated by pi/3.
        # Let's just use phi, phi+pi/3, phi+2pi/3 as the normal directions.
        
        u0 = np.array([np.cos(phi), np.sin(phi)])
        u1 = np.array([np.cos(phi + np.pi/3), np.sin(phi + np.pi/3)])
        u2 = np.array([np.cos(phi + 2*np.pi/3), np.sin(phi + 2*np.pi/3)])
        
        # Project vertices onto these directions
        p0 = vertices @ u0
        p1 = vertices @ u1
        p2 = vertices @ u2
        
        w0 = np.max(p0) - np.min(p0)
        w1 = np.max(p1) - np.min(p1)
        w2 = np.max(p2) - np.min(p2)
        
        max_w = max(w0, w1, w2)
        # Side length S relates to width W by W = S * sqrt(3)
        side = max_w / np.sqrt(3)
        
        if side < min_side:
            min_side = side
            
    return min_side

def _overlap_penalty(positions, angles):
    """
    Calculate overlap penalty between all pairs of hexagons.
    positions: (N, 2)
    angles: (N,)
    """
    penalty = 0.0
    n = len(positions)
    # Precompute vertices for all hexagons
    verts = []
    for i in range(n):
        verts.append(_hex_vertices(positions[i], angles[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            # Simple check: distance between centers
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist > 2.5: # Max distance for overlap is 2*sqrt(3) approx, but 2.5 is safe for unit hexes
                continue
            
            # SAT check
            # Get edges of hex i and hex j
            vi = verts[i]
            vj = verts[j]
            
            # Axes to test: normals to edges of vi and vj
            axes = []
            for k in range(6):
                # Edge from vi[k] to vi[(k+1)%6]
                edge = vi[(k+1)%6] - vi[k]
                # Normal
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    axes.append(normal / norm)
            for k in range(6):
                edge = vj[(k+1)%6] - vj[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    axes.append(normal / norm)
            
            overlap = False
            for axis in axes:
                proj_i = vi @ axis
                proj_j = vj @ axis
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j or max_j < min_i:
                    overlap = False
                    break
                else:
                    # Calculate overlap depth
                    overlap_depth = min(max_i, max_j) - max(min_i, min_j)
                    # We just want to know if they overlap.
                    # If they don't separate, they overlap.
                    pass
            
            # If no separating axis found, they overlap.
            # We need a penalty value.
            # Approximate penalty: if they overlap, add 1.0
            # To make it smoother, we could use the minimum overlap depth, but SAT gives boolean.
            # Let's just add 1.0 for any overlap.
            if not (max_i < min_j or max_j < min_i): # This logic is flawed in the loop above.
                 # The loop breaks if separated. If it finishes without breaking, they overlap.
                 # Let's re-implement cleanly.
                 pass
            
            # Cleaner SAT implementation
            separated = False
            min_overlap = float('inf')
            for axis in axes:
                proj_i = vi @ axis
                proj_j = vj @ axis
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j or max_j < min_i:
                    separated = True
                    break
                else:
                    overlap_depth = min(max_i, max_j) - max(min_i, min_j)
                    if overlap_depth < min_overlap:
                        min_overlap = overlap_depth
            
            if not separated:
                # Penalty proportional to overlap depth
                penalty += min_overlap
            # else:
                # No overlap
    
    return penalty

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Initial layout: 7-hex flower + 4 extensions
    # Centers of 7-hex flower
    d = np.sqrt(3)
    centers = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [-d, 0.0],
        [d/2, d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [-d/2, -d*np.sqrt(3)/2],
        # Add 4 more at distance 2d in directions 0, pi/3, 2pi/3, pi
        [2*d, 0.0],
        [d, d*np.sqrt(3)], # 2d * (cos(pi/3), sin(pi/3)) = (d, d*sqrt(3))
        [-d, d*np.sqrt(3)], # 2d * (cos(2pi/3), sin(2pi/3)) = (-d, d*sqrt(3))
        [0.0, 2*d*np.sqrt(3)] # This is not a corner. 
        # Let's place them at the "corners" of the ring.
        # The ring hexagons are at angles 0, 60, 120, 180, 240, 300.
        # The gaps are at 30, 90, 150, 210, 270, 330.
        # Placing a hex in a gap touches 2 ring hexagons.
        # Distance from center to gap center?
        # Gap center is at distance d * sqrt(3) ? No.
        # Let's just put them at 2d in directions 30, 90, 150, 210 degrees?
    ])
    
    # Let's refine the initial positions for the last 4.
    # Directions: 30, 90, 150, 270 (to break symmetry? or 30, 150, 210, 330?)
    # Let's try 30, 150, 210, 330 (symmetric)
    angles_init = np.zeros(N)
    
    # Re-define centers for the last 4
    d = np.sqrt(3)
    r = 2 * d # Distance from center
    ang1 = np.deg2rad(30)
    ang2 = np.deg2rad(150)
    ang3 = np.deg2rad(210)
    ang4 = np.deg2rad(330)
    
    centers[7] = [r * np.cos(ang1), r * np.sin(ang1)]
    centers[8] = [r * np.cos(ang2), r * np.sin(ang2)]
    centers[9] = [r * np.cos(ang3), r * np.sin(ang3)]
    centers[10] = [r * np.cos(ang4), r * np.sin(ang4)]
    
    # SA Parameters
    T = 1.0
    T_min = 1e-4
    alpha = 0.995
    max_iter = 5000
    best_positions = centers.copy()
    best_angles = angles_init.copy()
    best_score = float('inf')
    
    current_positions = centers.copy()
    current_angles = angles_init.copy()
    
    # Calculate initial score
    # Score = Overlap Penalty + Min Hex Side Length
    # We use a weighted sum. Overlap is critical.
    
    def evaluate(pos, ang):
        if time.time() > deadline:
            return float('inf')
        op = _overlap_penalty(pos, ang)
        if op > 1e-3:
            # High overlap, add penalty to size
            # Get all vertices
            verts_list = []
            for i in range(N):
                verts_list.append(_hex_vertices(pos[i], ang[i]))
            verts = np.vstack(verts_list)
            side = _min_width_hexagon(verts, num_angles=9) # Fewer angles for speed
            return op * 100.0 + side
        else:
            verts_list = []
            for i in range(N):
                verts_list.append(_hex_vertices(pos[i], ang[i]))
            verts = np.vstack(verts_list)
            side = _min_width_hexagon(verts, num_angles=36)
            return side

    current_score = evaluate(current_positions, current_angles)
    best_score = current_score
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Perturb
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        idx = random.randint(0, N-1)
        delta_pos = np.random.normal(0, 0.5, 2)
        delta_ang = np.random.normal(0, 0.2)
        
        new_positions[idx] += delta_pos
        new_angles[idx] += delta_ang
        
        new_score = evaluate(new_positions, new_angles)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
            current_positions = new_positions
            current_angles = new_angles
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_positions = current_positions.copy()
                best_angles = current_angles.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Final evaluation with high precision
    # If best_score was inf, try a simpler layout
    if best_score > 100:
        # Fallback to linear layout
        best_positions = np.array([[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)])
        best_angles = np.zeros(N)
        best_score = 40.0
    
    # Calculate final side length and center
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(best_positions[i], best_angles[i]))
    verts = np.vstack(verts_list)
    
    # Find min side and orientation
    min_side = float('inf')
    best_phi = 0.0
    for i in range(180):
        phi = (i / 180.0) * (np.pi / 3)
        u0 = np.array([np.cos(phi), np.sin(phi)])
        u1 = np.array([np.cos(phi + np.pi/3), np.sin(phi + np.pi/3)])
        u2 = np.array([np.cos(phi + 2*np.pi/3), np.sin(phi + 2*np.pi/3)])
        
        p0 = verts @ u0
        p1 = verts @ u1
        p2 = verts @ u2
        
        w0 = np.max(p0) - np.min(p0)
        w1 = np.max(p1) - np.min(p1)
        w2 = np.max(p2) - np.min(p2)
        
        max_w = max(w0, w1, w2)
        side = max_w / np.sqrt(3)
        
        if side < min_side:
            min_side = side
            best_phi = phi
            
    # The center of the hexagon is the centroid of the vertices?
    # For a minimal enclosing hexagon, the center is the midpoint of the bounding box in each direction.
    # We can approximate it by the centroid of all vertices.
    center = np.mean(verts, axis=0)
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], np.rad2deg(best_angles[i])])
    
    # Return format: inner, center, side, angle
    # angle is the rotation of the outer hexagon
    return inner, (center[0], center[1]), min_side, np.rad2deg(best_phi)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
