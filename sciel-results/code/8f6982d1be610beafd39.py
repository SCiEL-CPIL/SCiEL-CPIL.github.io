import numpy as np
import time
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    # Vertices of a unit hexagon at origin
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    all_verts = []
    for p in positions:
        all_verts.append(unit_hex_verts + p)
    all_verts = np.vstack(all_verts)
    
    # Outer hexagon edges normals (inward pointing)
    # Outer hexagon vertices: (s cos(a + 60i), s sin(a + 60i))
    # Edge i connects vertex i and i+1.
    # Normal to edge i points inward.
    # For a regular hexagon, the distance from center to edge (apothem) is s * sqrt(3)/2.
    # The normal vector for edge i (between vertex i and i+1) is in the direction of the angle bisector of the center and the edge midpoint.
    # Angle of vertex i is a + 60i. Angle of vertex i+1 is a + 60(i+1).
    # Midpoint angle is a + 60i + 30.
    # Inward normal direction is a + 60i + 30 + 180? No, inward is towards center.
    # Let's use the property: point (x,y) is inside hexagon of side s, angle a, center (0,0)
    # if for all i, dot((x,y), n_i) <= s * sqrt(3)/2, where n_i is unit inward normal.
    
    a_rad = np.deg2rad(angle_deg)
    # Inward normals
    # Vertex i is at angle a + 60i.
    # Edge i is between V_i and V_{i+1}.
    # The normal to edge i points towards the center.
    # The angle of the normal is a + 60i + 30 + 180? 
    # Let's check: i=0. V0 at a. V1 at a+60. Midpoint at a+30.
    # Vector from midpoint to center is opposite to midpoint direction.
    # So normal direction is a+30+180 = a+210.
    # Wait, standard hexagon with a=0:
    # V0=(1,0), V1=(0.5, sqrt(3)/2). Edge 0. Midpoint (0.75, sqrt(3)/4).
    # Normal towards origin is (-0.75, -sqrt(3)/4) normalized -> (-sqrt(3)/2, -1/2).
    # Angle of (-sqrt(3)/2, -1/2) is 210 degrees.
    # Formula: a + 60i + 210? For i=0, 0+210=210. Correct.
    # So n_i angle = a + 60i + 210 degrees? Or a + 60i - 150?
    # Let's just compute normals directly.
    
    normals = []
    for i in range(6):
        v1 = np.array([np.cos(a_rad + i*np.pi/3), np.sin(a_rad + i*np.pi/3)])
        v2 = np.array([np.cos(a_rad + (i+1)*np.pi/3), np.sin(a_rad + (i+1)*np.pi/3)])
        e = v2 - v1
        # Inward normal is perpendicular to e, pointing to origin.
        # Rotate e by -90 deg? 
        # e for i=0 (a=0): (0.5, sqrt(3)/2) - (1,0) = (-0.5, sqrt(3)/2).
        # Perp: (-sqrt(3)/2, -0.5) or (sqrt(3)/2, 0.5).
        # Dot with origin (0,0) - midpoint?
        # Midpoint m = (v1+v2)/2.
        # We want n such that dot(n, m) < 0 (if origin is center).
        # Let's use cross product in 2D or just rotate.
        # Rotate e by -90 degrees: (e_y, -e_x).
        n = np.array([e[1], -e[0]])
        # Check direction
        if np.dot(n, v1) > 0: # v1 is on boundary, origin is 0. dot(n, v1) should be positive if n is outward?
            # Actually, for a convex polygon containing origin, inward normal n satisfies dot(n, v) > 0? No.
            # Let's just project.
            pass
        # Simpler: The distance from origin to the line containing the edge is the apothem.
        # The line equation is dot(n_unit, x) = apothem.
        # n_unit should point from origin to the edge? No, from origin to outside.
        # Let's define n_unit as outward normal.
        # Then constraint is dot(n_unit, x) <= s * sqrt(3)/2 for all x inside.
        # For edge i, outward normal direction is opposite to inward.
        # Inward normal angle was a + 60i + 210. Outward is a + 60i + 30.
        # Let's verify. i=0, a=0. Outward angle 30.
        # Normal (cos30, sin30) = (sqrt(3)/2, 0.5).
        # Dot with V0 (1,0) is sqrt(3)/2.
        # Dot with V1 (0.5, sqrt(3)/2) is 3/4 + sqrt(3)/4. Not equal.
        # Wait, the line passes through V0 and V1.
        # Distance from origin to line is apothem.
        # The normal vector from origin to the line is in the direction of the angle bisector of V0 and V1.
        # Angle bisector of 0 and 60 is 30.
        # So outward normal is indeed at angle a + 60i + 30.
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
        
    normals = np.array(normals)
    
    # For each normal, find the maximum projection of all inner vertices.
    # max_proj = max( dot(x, n) for x in all_verts )
    # This max_proj must be <= s * sqrt(3)/2.
    # So s >= max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3).
    # We need the max over all 6 normals.
    
    projections = all_verts @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    # Initial guess: Hexagonal lattice
    # Centers:
    # (0,0)
    # (2.5, 0), (-2.5, 0)
    # (1.25, 2.165), (-1.25, 2.165), (1.25, -2.165), (-1.25, -2.165)
    # (3.75, 2.165), (-3.75, 2.165), (3.75, -2.165), (-3.75, -2.165), (0, 4.33)
    # Let's refine these coordinates.
    # Distance between centers in hex packing is 2 * apothem = sqrt(3) * side? No.
    # Side 1. Apothem sqrt(3)/2.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait, if they touch, distance is 2 * (distance from center to edge) = 2 * sqrt(3)/2 = sqrt(3).
    # Let's check the seed program.
    # Seed: [-2.5, 0], [2.5, 0]. Distance 5.
    # My lattice:
    # Row 0: (0,0)
    # Row 1: (sqrt(3)/2 * 1? No.
    # Let's use standard hex grid coordinates.
    # Center (0,0).
    # Neighbors at distance sqrt(3).
    # Angles 0, 60, 120, 180, 240, 300.
    # (sqrt(3), 0)
    # (sqrt(3)/2, 1.5)
    # (-sqrt(3)/2, 1.5)
    # (-sqrt(3), 0)
    # (-sqrt(3)/2, -1.5)
    # (sqrt(3)/2, -1.5)
    
    # Let's build a 12-hexagon cluster.
    # Center + 6 neighbors + 5 more?
    # A hexagon of hexagons with side 2 has 1 + 6 + 12 = 19. Too many.
    # Side 1 (center + 6) = 7.
    # We need 12.
    # Maybe a 2x3 rectangle of hexagons?
    # Or just let the optimizer figure it out.
    # Initial positions:
    # 12 points roughly in a hexagon shape.
    
    h = np.sqrt(3)/2
    init_pos = np.array([
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ])
    
    # Flatten positions and add angle
    # Variables: x1, y1, ..., x12, y12, angle
    x0 = np.concatenate([init_pos.flatten(), [0]])
    
    def objective(params):
        pos = params[:24].reshape(12, 2)
        angle = params[24]
        
        # Check time
        if time.time() > deadline:
            return 1e6
            
        s = _min_outer_side(pos, angle)
        
        # Verify validity quickly? 
        # The problem states "Invalid: ... two inner hexagons not separated".
        # Our objective minimizes s. If they overlap, the "packing" is invalid.
        # Does _min_outer_side care about overlap? No.
        # We must ensure non-overlap.
        # If they overlap, the configuration is invalid.
        # We can add a penalty for overlap.
        
        # Separating Axis Test for overlap
        # For each pair, check if they overlap.
        # If they overlap, return a large penalty.
        
        # To be efficient, we can just check a few pairs or use a simple distance check?
        # Distance between centers < 2 * apothem means they MIGHT overlap, but not necessarily.
        # For regular hexagons, if distance > 2*apothem, they don't overlap?
        # No, if rotated, they can overlap even if centers are far?
        # Actually, for convex polygons, if distance between centers is greater than the sum of their circumradii, they don't overlap.
        # Circumradius = 1. Sum = 2.
        # If dist > 2, no overlap.
        # If dist < sqrt(3) (2*apothem), they definitely overlap if aligned?
        # Let's use a simple check: if any pair of centers is closer than sqrt(3) - epsilon, assume potential overlap and do SAT?
        # Or just do SAT for all pairs. 12*11/2 = 66 pairs. 6 SAT checks each. 396 checks.
        # Each check involves 6 vertices of A and 6 of B.
        # This is fast in NumPy.
        
        # Let's implement a fast overlap check.
        # Precompute vertices for all hexagons?
        # Vertices depend on position and angle.
        # Angle is global in our parameterization? 
        # Wait, the task says "inner a (12, 3) array of rows (x, y, angle_deg)".
        # So each inner hexagon can have a DIFFERENT angle.
        # My current parameterization has a single global angle for all inner hexagons?
        # No, I only passed `angle` to `_min_outer_side` for the OUTER hexagon.
        # The inner hexagons in `_min_outer_side` are assumed to be at angle 0?
        # `unit_hex_verts = _hex_vertices(0, 0, 1, 0)`.
        # So I am assuming all inner hexagons are axis-aligned (angle 0).
        # Is this a valid restriction?
        # The optimal packing might involve rotated inner hexagons.
        # However, for regular hexagons, it's often optimal to keep them aligned.
        # Let's stick to aligned inner hexagons for now to reduce dimensions.
        # If we need to, we can add angles later.
        
        # Check overlap for aligned hexagons.
        # Vertices of unit hexagon at origin:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        
        verts = _hex_vertices(0, 0, 1, 0)
        
        all_verts = pos[:, np.newaxis, :] + verts[np.newaxis, :, :] # (12, 6, 2)
        
        # SAT
        for i in range(12):
            for j in range(i+1, 12):
                # Check axes from i and j
                # Edges of i
                for k in range(6):
                    v1 = all_verts[i, k]
                    v2 = all_verts[i, (k+1)%6]
                    e = v2 - v1
                    n = np.array([-e[1], e[0]])
                    if np.linalg.norm(n) < 1e-10: continue
                    n = n / np.linalg.norm(n)
                    
                    proj_i = np.dot(all_verts[i], n)
                    proj_j = np.dot(all_verts[j], n)
                    
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        continue
                    # If we get here, they might overlap on this axis.
                    # We need to check all axes. If ANY axis separates, they don't overlap.
                    # So if we find a separating axis, break.
                    # If we check all axes and none separate, they overlap.
                    
                # Optimization: If we find a separating axis, we can break early.
                # Let's restructure.
                
        # Let's write a helper for SAT overlap.
        def is_overlap(verts_a, verts_b):
            # Check axes from A
            for k in range(6):
                v1 = verts_a[k]
                v2 = verts_a[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-10: continue
                n = n / norm
                proj_a = np.dot(verts_a, n)
                proj_b = np.dot(verts_b, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    return False
            # Check axes from B
            for k in range(6):
                v1 = verts_b[k]
                v2 = verts_b[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-10: continue
                n = n / norm
                proj_a = np.dot(verts_a, n)
                proj_b = np.dot(verts_b, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    return False
            return True

        # Check all pairs
        for i in range(12):
            for j in range(i+1, 12):
                if is_overlap(all_verts[i], all_verts[j]):
                    return 1e6 + s # Penalty + objective
        
        return s

    # Bounds
    # Positions can be anywhere, but let's bound them to a reasonable box.
    # Outer hexagon side s. If s is around 4-5, positions are within ~5.
    bounds = [(-5, 5)] * 24 + [(0, 180)]
    
    # Run optimization
    result = differential_evolution(objective, bounds, maxiter=100, popsize=15, tol=1e-4, seed=42, polish=True)
    
    if result.success or result.fun < 1e5:
        best_params = result.x
        best_pos = best_params[:24].reshape(12, 2)
        best_angle = best_params[24]
        
        # Construct return values
        inner_hex_data = np.zeros((12, 3))
        inner_hex_data[:, :2] = best_pos
        inner_hex_data[:, 2] = 0 # All inner hexagons angle 0
        
        outer_hex_data = np.array([0, 0, best_angle])
        outer_hex_side_length = _min_outer_side(best_pos, best_angle)
        
        # Final validation and adjustment
        # Ensure s is at least sqrt(12)
        if outer_hex_side_length < np.sqrt(12):
            outer_hex_side_length = np.sqrt(12)
            
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    else:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
