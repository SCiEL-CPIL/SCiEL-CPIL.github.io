import numpy as np
from scipy.optimize import linprog
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Helper to get vertices of a unit regular hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = np.radians(angle_deg)
        # Vertices at angles 0, 60, ..., 300 relative to center
        # Side length 1 -> Circumradius 1
        angles = np.arange(6) * np.pi / 3 + angle_rad
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack((xs, ys))

    # Helper to find minimum side length of enclosing regular hexagon
    def min_enclosing_hexagon_side(vertices):
        # vertices: Nx2 array
        min_side = float('inf')
        # Try container orientations from 0 to 60 degrees
        # Period is 60 degrees for regular hexagon
        for phi_deg in np.arange(0, 60, 1.0):
            phi_rad = np.radians(phi_deg)
            # Normals for the 6 sides of the container hexagon
            # Angles: phi, phi+60, ..., phi+300
            normal_angles = phi_rad + np.arange(6) * np.pi / 3
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            
            # Project vertices onto normals
            projections = vertices @ normals.T  # N x 6
            M = np.max(projections, axis=0)  # 6
            m = np.min(projections, axis=0)  # 6
            
            # Setup LP
            # Minimize A
            # x = [cx, cy, A]
            c_lp = np.array([0.0, 0.0, 1.0])
            
            A_lp = []
            b_lp = []
            
            for i in range(6):
                nx, ny = normals[i]
                # Constraint 1: -nx*cx - ny*cy - A <= -M[i]
                A_lp.append([-nx, -ny, -1.0])
                b_lp.append(-M[i])
                
                # Constraint 2: nx*cx + ny*cy - A <= m[i]
                A_lp.append([nx, ny, -1.0])
                b_lp.append(m[i])
                
            A_lp = np.array(A_lp)
            b_lp = np.array(b_lp)
            
            res = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, method='highs')
            
            if res.status == 0:
                A_val = res.x[2]
                # Side length S = A * 2 / sqrt(3)
                side = A_val * 2.0 / np.sqrt(3.0)
                if side < min_side:
                    min_side = side
                    best_phi = phi_deg
                    best_center = (res.x[0], res.x[1])
            else:
                # Fallback if LP fails (should not happen)
                pass
                
        return min_side, best_phi, best_center

    # Generate candidate packings
    # Axial coordinates to cartesian:
    # x = size * (q + r/2)
    # y = size * (r * sqrt(3)/2)
    # Here size=1 (distance between centers is sqrt(3) in cartesian? No.)
    # Standard hex grid with side 1 hexagons:
    # Distance between centers is sqrt(3).
    # Let's use a coordinate system where center is (0,0).
    # Neighbors are at distance sqrt(3).
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    # Base 7-hexagon cluster
    base_centers = [(0,0)]
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for q,r in ring1:
        base_centers.append(axial_to_cartesian(q,r))
        
    # Ring 2
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q+r) == 2: # This is not the right condition for ring 2
                pass
                
    # Correct ring generation
    def get_ring(n):
        if n == 0: return [(0,0)]
        # Start at (n, 0)
        # Directions: (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0)
        dirs = [(1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0)]
        q, r = n, 0
        res = []
        for i in range(6):
            for j in range(n):
                res.append((q, r))
                q += dirs[i][0]
                r += dirs[i][1]
        return res

    ring1_axial = get_ring(1)
    ring2_axial = get_ring(2)
    
    # Convert base to cartesian
    base_cart = [axial_to_cartesian(q,r) for q,r in [(0,0)] + ring1_axial]
    
    # Candidates for the 4 extra hexagons from ring2
    # We want to pick 4 indices from 0..11
    # Try symmetric distributions
    candidate_indices = [
        [0, 3, 6, 9],
        [0, 1, 6, 7],
        [0, 2, 5, 7],
        [0, 4, 6, 10],
        [0, 1, 2, 5],
        [0, 3, 5, 8]
    ]
    
    best_result = None
    best_score = float('inf')
    
    for idxs in candidate_indices:
        if time.time() > deadline:
            break
            
        centers = list(base_cart)
        for i in idxs:
            q, r = ring2_axial[i]
            centers.append(axial_to_cartesian(q,r))
            
        # Ensure we have exactly 11
        if len(centers) != 11:
            continue
            
        # Generate vertices for all hexagons (angle 0)
        all_vertices = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy, 0.0)
            all_vertices.append(verts)
            
        all_vertices = np.vstack(all_vertices)
        
        side, phi, center = min_enclosing_hexagon_side(all_vertices)
        
        if side < best_score:
            best_score = side
            # Construct the inner list
            # The inner hexagons are at 'centers' with angle 0
            # The outer hexagon is at 'center' with side 'side' and angle 'phi'
            inner = [[cx, cy, 0.0] for cx, cy in centers]
            best_result = (inner, center, side, phi)

    # If no result found, return a safe default (though unlikely)
    if best_result is None:
        # Fallback: simple grid
        inner = [[i * np.sqrt(3), 0, 0] for i in range(11)]
        return inner, (0,0), 50.0, 0.0
        
    return best_result
# EVOLVE-BLOCK-END
