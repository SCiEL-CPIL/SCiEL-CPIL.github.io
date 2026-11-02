# EVOLVE-BLOCK-START
import math
import time
import numpy as np

N = 11

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    # Helper: Calculate minimal enclosing regular hexagon side length
    # for a set of 2D points, given a rotation angle theta for the hexagon.
    # Returns the side length s.
    def min_hex_side(points, theta):
        # points: (M, 2) array
        # theta: rotation of the enclosing hexagon
        # Normals for the enclosing hexagon
        # A regular hexagon with side s has apothem a = s * sqrt(3) / 2
        # The vertices of the inner hexagons are the 'points' we must enclose.
        # Wait, we must enclose the *inner hexagons*, not just their centers.
        # So we need the vertices of all inner hexagons.
        
        pass # This function will be called with vertices

    def get_vertices(inner_list):
        verts = []
        for x, y, ang_deg in inner_list:
            ang = math.radians(ang_deg)
            for j in range(6):
                va = ang + j * math.pi / 3
                vx = x + math.cos(va)
                vy = y + math.sin(va)
                verts.append([vx, vy])
        return np.array(verts)

    def evaluate_layout(inner_list):
        """
        Returns the minimal side length of the enclosing regular hexagon.
        We try 12 discrete rotations for the outer hexagon to find the best fit
        for the given inner layout.
        """
        verts = get_vertices(inner_list)
        if len(verts) == 0:
            return 0.0
        
        best_s = float('inf')
        
        # Try 12 rotations (every 30 degrees) to cover all symmetries efficiently
        # Actually, due to symmetry, 0 to 60 is enough, but 30 deg steps are safe.
        for theta_deg in range(0, 60, 30):
            theta = math.radians(theta_deg)
            # 6 normal directions
            normals = np.array([[math.cos(theta + k * math.pi / 3), math.sin(theta + k * math.pi / 3)] for k in range(6)])
            
            # Project vertices onto normals
            # projections[i, k] = dot(verts[i], normals[k])
            projections = verts @ normals.T
            
            # For each pair of opposite normals (k, k+3), calculate the required half-width
            # The center of the enclosing hexagon is not fixed at 0,0.
            # We need to find the center (cx, cy) that minimizes the max projection distance.
            # For a fixed theta, the minimal enclosing hexagon center is the Chebyshev center of the projections?
            # Actually, for a fixed theta, the problem reduces to finding the smallest square (in the projected space of 3 axes) 
            # that contains the points, but constrained to be a regular hexagon?
            
            # Let's use the property: 
            # The minimal s is determined by the max over the 3 axes of the range of projections?
            # No.
            
            # Robust method:
            # The enclosing hexagon is defined by 6 half-planes.
            # We want to minimize s.
            # For a given theta, the optimal center (cx, cy) is the centroid of the points? 
            # No, it's the point that minimizes the max distance to the boundaries.
            
            # Let's approximate by trying a few centers:
            # 1. Centroid of vertices
            # 2. Centroid of inner hexagon centers
            
            centers = [
                verts.mean(axis=0),
                np.array([inner_list[i][0] for i in range(len(inner_list))], dtype=float).mean(axis=0),
                np.array([0.0, 0.0]) # Origin
            ]
            
            for c in centers:
                # Shift projections by center
                # proj_shifted[i, k] = dot(verts[i] - c, normals[k])
                # = dot(verts[i], normals[k]) - dot(c, normals[k])
                c_proj = c @ normals.T
                shifted = projections - c_proj
                
                # For each direction k, we need max(shifted[:, k]) <= s*sqrt(3)/2 
                # and max(shifted[:, k+3]) <= s*sqrt(3)/2
                # Note normals[k+3] = -normals[k]
                # So shifted[:, k+3] = -shifted[:, k]
                # So we need max(|shifted[:, k]|) <= s*sqrt(3)/2 for k=0,1,2
                # Thus s >= 2/sqrt(3) * max(|shifted[:, k]|)
                
                # Calculate max abs projection for the 3 unique directions
                max_abs_proj = 0.0
                for k in range(3):
                    max_abs_proj = max(max_abs_proj, np.max(np.abs(shifted[:, k])))
                
                s_candidate = (2.0 / math.sqrt(3.0)) * max_abs_proj
                if s_candidate < best_s:
                    best_s = s_candidate
        
        return best_s

    # Initial Layout: Hexagonal grid
    # Center
    positions = [[0.0, 0.0]]
    # Ring 1
    for i in range(6):
        angle = i * math.pi / 3
        positions.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
    # Ring 2: Add 4 more
    # To make it compact, place them at the "midpoints" of the outer ring edges?
    # Or just extend the grid.
    # A standard hexagonal packing of 19 (side 3) has 1+6+12.
    # We have 11. 1+6+4.
    # Let's place the 4 at angles 0, 60, 120, 180 of ring 2 (distance 4).
    # This creates a "diamond" shape.
    for i in range(4):
        angle = i * math.pi / 3
        positions.append([4.0 * math.cos(angle), 4.0 * math.sin(angle)])
    
    angles = [0.0] * N
    inner_init = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    
    best_score = evaluate_layout(inner_init)
    best_inner = [row[:] for row in inner_init]
    
    # Local Search / Hill Climbing
    # We will perturb positions and angles slightly to see if we can reduce the score.
    current_inner = [row[:] for row in inner_init]
    current_score = best_score
    
    step_size = 0.5
    num_iterations = 5000
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random hexagon to move
        idx = np.random.randint(N)
        
        # Save old state
        old_x, old_y, old_a = current_inner[idx]
        
        # Perturb
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        da = np.random.uniform(-15, 15) # degrees
        
        new_x = old_x + dx
        new_y = old_y + dy
        new_a = old_a + da
        
        # Apply move
        current_inner[idx] = [new_x, new_y, new_a]
        
        new_score = evaluate_layout(current_inner)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_inner = [row[:] for row in current_inner]
        else:
            # Revert
            current_inner[idx] = [old_x, old_y, old_a]
        
        # Occasionally reduce step size (simulated annealing style cooling)
        if i % 500 == 0:
            step_size *= 0.9

    # Determine outer hexagon parameters for the best inner layout
    # We need to return: inner, center, side, angle
    # We already have best_side. We need center and angle.
    # Let's re-evaluate to find the specific center and angle that gave best_side.
    
    verts = get_vertices(best_inner)
    best_s_final = float('inf')
    best_c = (0,0)
    best_theta = 0.0
    
    for theta_deg in range(0, 60, 30):
        theta = math.radians(theta_deg)
        normals = np.array([[math.cos(theta + k * math.pi / 3), math.sin(theta + k * math.pi / 3)] for k in range(6)])
        projections = verts @ normals.T
        
        centers = [
            verts.mean(axis=0),
            np.array([best_inner[i][0] for i in range(len(best_inner))], dtype=float).mean(axis=0),
            np.array([0.0, 0.0])
        ]
        
        for c in centers:
            c_proj = c @ normals.T
            shifted = projections - c_proj
            max_abs_proj = 0.0
            for k in range(3):
                max_abs_proj = max(max_abs_proj, np.max(np.abs(shifted[:, k])))
            
            s_candidate = (2.0 / math.sqrt(3.0)) * max_abs_proj
            if s_candidate < best_s_final:
                best_s_final = s_candidate
                best_c = tuple(c)
                best_theta = theta

    return best_inner, best_c, best_s_final, math.degrees(best_theta)
# EVOLVE-BLOCK-END
