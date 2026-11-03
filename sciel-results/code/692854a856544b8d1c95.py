"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True
    
    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        positions: list of (x, y) for each inner hexagon
        angles: list of rotation angles for each inner hexagon
        outer_angle_deg: rotation of outer hexagon
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        # Get all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        
        # For a regular hexagon with side s and rotation theta, the constraint is:
        # For each vertex (x, y), it must satisfy:
        # x*cos(theta) + y*sin(theta) <= s*sqrt(3)/2  (and similar for other 3 axes)
        # Actually, for a regular hexagon centered at origin with side s and rotation theta:
        # The support function in direction (cos phi, sin phi) is:
        # h(phi) = s * max over k of cos(phi - theta - k*pi/3)
        # For a regular hexagon, the constraint is that for each of the 6 normals:
        # dot(vertex, normal_k) <= s * sqrt(3)/2
        # where normal_k are the outward normals at angles theta + k*pi/3
        
        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        
        # For a regular hexagon with side s, the apothem (inradius) is s*sqrt(3)/2
        # So we need: max_proj <= s * sqrt(3)/2
        # Therefore: s >= max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
        side = 2 * max_proj / math.sqrt(3)
        return side
    
    # Objective function: minimize the outer hexagon side length
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_x, outer_y, outer_angle]
        positions = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            positions.append((x, y))
            angles.append(a)
        
        outer_x = params[3*N]
        outer_y = params[3*N + 1]
        outer_angle = params[3*N + 2]
        
        # Center the inner hexagons relative to outer center
        # Translate so that the centroid of inner hexagons is at (0,0)
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(x - cx + outer_x, y - cy + outer_y) for x, y in positions]
        
        # Check for overlaps
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return 1e10 + 0.1 * (i + j)  # Penalize overlaps
        
        # Compute required outer side length
        side = compute_min_side(positions, angles, outer_angle)
        return side
    
    # Initial guess: honeycomb-like arrangement
    # 4 rows of 3 hexagons in a honeycomb pattern
    # Row offsets for honeycomb: 0, 0.5*sqrt(3), 0, 0.5*sqrt(3)
    # Actually for hexagons with side 1, the distance between centers in honeycomb:
    # Horizontal neighbors: 2*sqrt(3)/2 = sqrt(3) ≈ 1.732 (wait, that's for flat-top)
    # Let me think: for pointy-top hexagons with side 1:
    # Width = 2, height = sqrt(3)
    # Horizontal spacing: 2*sqrt(3)/2 * ... 
    # Actually, let's just use a simple initial guess and let the optimizer work.
    
    # Simple initial: 3x4 grid with hexagon spacing
    # For hexagons of side 1, minimum center-to-center distance to avoid overlap is 2 (touching)
    # But with rotation, we can do better. Let's start with a reasonable guess.
    
    # Honeycomb: 4 columns of 3, or 3 rows of 4
    # Let's try 4 rows of 3
    init_params = []
    for row in range(4):
        for col in range(3):
            # Hexagon center spacing: horizontal = 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464 for touching
            # Actually for two hexagons of side 1 touching, center distance = 2
            # Let's use 2.1 to have a small gap
            x = col * 2.1 * math.sqrt(3) / 2  # ≈ 1.819
            y = row * 2.0 + (col % 2) * 1.0  # Offset odd columns
            a = 0.0
            init_params.extend([x, y, a])
    
    # Outer hexagon initial: center at centroid, angle 0
    cx = sum(init_params[3*i] for i in range(N)) / N
    cy = sum(init_params[3*i + 1] for i in range(N)) / N
    init_params.extend([cx, cy, 0.0])
    
    init_params = np.array(init_params)
    
    # Optimize using Nelder-Mead
    best_result = None
    best_side = 1e10
    
    # Run multiple restarts with different initial guesses
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            params0 = init_params.copy()
        else:
            # Perturb initial guess
            params0 = init_params + np.random.randn(len(init_params)) * 0.5
        
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if result.fun < best_side and result.fun < 1e9:
                best_side = result.fun
                best_result = result.x.copy()
        except Exception:
            continue
        
        if time.time() > deadline - 20:
            break
    
    if best_result is None:
        best_result = init_params
    
    # Extract final positions
    positions = []
    angles = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i + 1]
        a = best_result[3*i + 2]
        positions.append((x, y))
        angles.append(a)
    
    outer_x = best_result[3*N]
    outer_y = best_result[3*N + 1]
    outer_angle = best_result[3*N + 2]
    
    # Build inner list
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    
    # Compute the actual side length
    side = compute_min_side(positions, angles, outer_angle)
    
    return inner, (outer_x, outer_y), side, outer_angle
# EVOLVE-BLOCK-END
