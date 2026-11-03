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
        """Check if two hexagons overlap using Separating Axis Theorem"""
        axes = []
        for i in range(3):
            # Edge normals for hexagon 1
            edge = v1[i+1] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            axes.append(normal)
            # Edge normals for hexagon 2
            edge = v2[i+1] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return False
        return True
    
    # Helper: compute minimum outer side length needed to contain all inner hexagons
    def compute_min_side(inner_hexes, outer_center, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons"""
        outer_center = np.array(outer_center)
        outer_angle_rad = math.radians(outer_angle_deg)
        
        # Support function: max projection onto each of the 6 outer hexagon normals
        max_proj = 0.0
        for i in range(6):
            # Normal direction for outer hexagon face i
            normal_angle = outer_angle_rad + i * math.pi / 3
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # For each inner hexagon, find max projection of its vertices
            for cx, cy, angle_deg in inner_hexes:
                verts = hex_vertices(cx, cy, angle_deg)
                # Project vertices relative to outer center
                rel_verts = verts - outer_center
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        # The outer hexagon with side length S has apothem (inradius) = S * sqrt(3)/2
        # So S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)
    
    # Helper: check if all hexagons are inside the outer hexagon
    def is_inside_outer(inner_hexes, outer_center, outer_side, outer_angle_deg):
        outer_center = np.array(outer_center)
        outer_angle_rad = math.radians(outer_angle_deg)
        apothem = outer_side * math.sqrt(3) / 2.0
        
        for i in range(6):
            normal_angle = outer_angle_rad + i * math.pi / 3
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            for cx, cy, angle_deg in inner_hexes:
                verts = hex_vertices(cx, cy, angle_deg)
                rel_verts = verts - outer_center
                proj = np.dot(rel_verts, normal)
                if np.max(proj) > apothem + 1e-6:
                    return False
        return True
    
    # Helper: minimum distance between two hexagons for non-overlap
    # For two unit hexagons, the minimum center-to-center distance depends on relative rotation
    def min_center_distance(angle1_deg, angle2_deg):
        """
        Compute the minimum center-to-center distance for two unit hexagons
        with given angles to be non-overlapping.
        For identical orientation, this is 2*sqrt(3)/sqrt(3) = 2 (flat-to-flat)
        Actually for hexagons, the minimum distance for non-overlap when aligned is 2*apothem = 2*sqrt(3)/2 * 2 = 2*sqrt(3)
        Wait: apothem of unit hexagon = sqrt(3)/2. Two hexagons touching flat-to-flat: distance = 2*apothem = sqrt(3)
        But they can also touch vertex-to-vertex.
        
        For two identical hexagons (same rotation), the minimum distance for non-overlap is:
        - Flat-to-flat: 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
        - Vertex-to-vertex: 2 * 1 = 2
        
        So minimum is sqrt(3) for aligned hexagons.
        
        For different rotations, it's more complex. We'll use a conservative bound.
        """
        # Conservative: use sqrt(3) for aligned, increase for misaligned
        # Actually, let's just use a fixed minimum of sqrt(3) - 0.01 for optimization
        # The SAT check will handle the exact constraint
        return math.sqrt(3) - 0.01
    
    # Build honeycomb initial configuration
    # Honeycomb lattice vectors for unit hexagons:
    # a1 = (3/2, sqrt(3)/2), a2 = (0, sqrt(3))
    # Or equivalently, neighbors at distance sqrt(3) in 6 directions
    
    sqrt3 = math.sqrt(3)
    
    # Honeycomb positions: use axial coordinates (q, r) -> (x, y)
    # x = 3/2 * q + sqrt(3)/2 * r
    # y = sqrt(3) * r
    # But for unit hexagons, the spacing needs to be sqrt(3)
    
    # Let's use a simpler approach: place hexagons in a honeycomb pattern
    # Row 0: 4 hexagons at x = -1.5, -0.5, 0.5, 1.5 (spacing 1.0 in x, but need sqrt(3) vertical)
    # Actually, for unit hexagons in honeycomb:
    # Horizontal spacing: 3/2 * 1 = 1.5 (for pointy-top) or 2*sqrt(3)/2 = sqrt(3) (for flat-top)
    
    # Let's use flat-top hexagons (angle 0 means flat side on top)
    # For flat-top hexagons with side 1:
    # Width (vertex to vertex horizontally) = 2
    # Height (flat to flat vertically) = sqrt(3)
    # Honeycomb: 
    #   Row spacing: sqrt(3)
    #   Column spacing: 3/2 (horizontal offset by 3/2, vertical offset by sqrt(3)/2)
    
    # Let me reconsider. For a regular hexagon with side 1 and angle 0 (flat top):
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # So vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Width = 2, Height = sqrt(3)
    
    # For honeycomb packing of flat-top hexagons:
    # Adjacent hexagons in same row: horizontal distance = 3/2 (they interlock)
    # Wait, that's not right. Let me think again.
    
    # Two flat-top hexagons side by side: their flat sides touch.
    # The flat side is vertical, at x = ±0.5. So if centers are at (0,0) and (3, 0), they touch? No.
    # Actually, the rightmost vertex of left hex is at x=1, leftmost vertex of right hex is at x=center_x-1.
    # For them to touch: center_x_right - 1 = 1, so center_x_right = 2.
    # But that's vertex-to-vertex. For flat-to-flat, they'd need to be offset vertically.
    
    # Let's just use a known honeycomb layout:
    # Axial coordinates for hex grid, converted to pixel coordinates
    # For pointy-top hexagons: x = size * (sqrt(3) * q + sqrt(3)/2 * r), y = size * (3/2 * r)
    # For flat-top: x = size * (3/2 * q), y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    
    # Let's use flat-top with size=1:
    # x = 3/2 * q, y = sqrt(3)/2 * q + sqrt(3) * r
    
    # Generate 12 hexagons in a compact honeycomb shape
    # A hexagonal number: 1 + 6 + 12 = 19 for radius 2. Too many.
    # For 12, let's use a rectangular-ish honeycomb block
    
    # Axial coordinates for a 3x4 honeycomb block:
    # q from 0 to 2, r from 0 to 3, but with proper hex offset
    
    # Simpler: just place them in a grid with honeycomb spacing
    positions = []
    
    # 3 columns, 4 rows, with honeycomb offset
    # Column x positions: 0, 1.5, 3.0 (spacing 1.5)
    # Row y positions: 0, sqrt(3), 2*sqrt(3), 3*sqrt(3)
    # Odd columns offset by sqrt(3)/2
    
    for col in range(3):
        for row in range(4):
            x = col * 1.5
            y = row * sqrt3 + (sqrt3 / 2 if col % 2 == 1 else 0)
            positions.append((x, y, 0.0))
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    # Re-center
    positions = [(x - cx, y - cy, ang) for x, y, ang in positions]
    
    # Verify no overlaps in initial config
    def check_no_overlaps(hexes):
        for i in range(len(hexes)):
            for j in range(i+1, len(hexes)):
                v1 = hex_vertices(*hexes[i])
                v2 = hex_vertices(*hexes[j])
                if hexagons_overlap(v1, v2):
                    return False
        return True
    
    if not check_no_overlaps(positions):
        # Fallback to wider spacing
        positions = []
        for col in range(3):
            for row in range(4):
                x = col * 2.0
                y = row * sqrt3
                positions.append((x, y, 0.0))
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        positions = [(x - cx, y - cy, ang) for x, y, ang in positions]
    
    # Objective function for SLSQP
    def objective(params):
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            inner_hexes.append((x, y, angle))
        
        outer_cx = params[N*3]
        outer_cy = params[N*3+1]
        outer_angle = params[N*3+2]
        
        side = compute_min_side(inner_hexes, (outer_cx, outer_cy), outer_angle)
        return side
    
    # Non-overlap constraints: for each pair, distance between centers >= min_dist
    def make_overlap_constraint(i, j):
        def constraint(params):
            xi = params[i*3]
            yi = params[i*3+1]
            ai = params[i*3+2]
            xj = params[j*3]
            yj = params[j*3+1]
            aj = params[j*3+2]
            
            # Compute actual minimum distance using SAT
            v1 = hex_vertices(xi, yi, ai)
            v2 = hex_vertices(xj, yj, aj)
            
            # Find minimum separation distance
            min_sep = float('inf')
            axes = []
            for k in range(3):
                edge1 = v1[k+1] - v1[k]
                n1 = np.array([-edge1[1], edge1[0]])
                n1 = n1 / (np.linalg.norm(n1) + 1e-12)
                axes.append(n1)
                
                edge2 = v2[k+1] - v2[k]
                n2 = np.array([-edge2[1], edge2[0]])
                n2 = n2 / (np.linalg.norm(n2) + 1e-12)
                axes.append(n2)
            
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Separation along this axis
                if max1 <= min2:
                    sep = min2 - max1
                elif max2 <= min1:
                    sep = min1 - max2
                else:
                    sep = 0.0  # Overlapping along this axis
                
                min_sep = min(min_sep, sep)
            
            # Constraint: min_sep >= 0 (no overlap)
            return min_sep
        
        constraint.name = f'no_overlap_{i}_{j}'
        return constraint
    
    # Build constraints
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': make_overlap_constraint(i, j),
                'jac': None  # Let SLSQP approximate
            })
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
        bounds.append((0, 60))  # angle (hexagon has 60-degree symmetry)
    bounds.append((-3, 3))  # outer_cx
    bounds.append((-3, 3))  # outer_cy
    bounds.append((0, 60))  # outer_angle
    
    # Initial parameters
    initial_params = []
    for px, py, angle in positions:
        initial_params.extend([px, py, angle])
    initial_params.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple starting points
    start_configs = [initial_params]
    
    # Add perturbed versions
    np.random.seed(42)
    for _ in range(3):
        perturbed = list(initial_params)
        for i in range(N):
            perturbed[i*3] += np.random.uniform(-0.1, 0.1)
            perturbed[i*3+1] += np.random.uniform(-0.1, 0.1)
        start_configs.append(perturbed)
    
    for start_idx, start_params in enumerate(start_configs):
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                objective,
                start_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 200,
                    'ftol': 1e-8,
                    'disp': False
                }
            )
            
            if result.success and result.fun < best_side:
                # Verify no overlaps
                final_hexes = []
                for i in range(N):
                    final_hexes.append((result.x[i*3], result.x[i*3+1], result.x[i*3+2]))
                
                if check_no_overlaps(final_hexes):
                    best_result = result.x
                    best_side = result.fun
        except Exception as e:
            pass
    
    # If optimization failed, use initial honeycomb
    if best_result is None:
        best_result = np.array(initial_params)
    
    # Extract final configuration
    final_inner_hexes = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        angle = best_result[i*3+2]
        final_inner_hexes.append((x, y, angle))
    
    outer_cx = best_result[N*3]
    outer_cy = best_result[N*3+1]
    outer_angle = best_result[N*3+2]
    
    # Final overlap check
    if not check_no_overlaps(final_inner_hexes):
        # Revert to initial honeycomb
        final_inner_hexes = positions
        outer_cx, outer_cy, outer_angle = 0.0, 0.0, 0.0
    
    inner = []
    for x, y, angle in final_inner_hexes:
        inner.append([float(x), float(y), float(angle)])
    
    final_side = compute_min_side(final_inner_hexes, (outer_cx, outer_cy), outer_angle)
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle)
# EVOLVE-BLOCK-END
