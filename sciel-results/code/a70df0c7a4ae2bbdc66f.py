"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Precompute hexagon vertices for a unit hexagon (side=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    hex_vertices = np.stack([np.cos(hex_angles), np.sin(hex_angles)], axis=1)
    
    def get_hex_vertices(center, angle_deg, side=1.0):
        """Get vertices of a regular hexagon with given center, rotation, and side."""
        a = angle_deg * np.pi / 180.0
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return center + side * (R @ hex_vertices.T).T
    
    def hexagon_contains_point(vertices, point):
        """Check if point is inside a convex polygon (hexagon)."""
        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            cross = (v2[0] - v1[0]) * (point[1] - v1[1]) - (v2[1] - v1[1]) * (point[0] - v1[0])
            if cross < -1e-9:
                return False
        return True
    
    def hexagons_disjoint(inner_hexes, outer_vertices, tol=1e-6):
        """Check if all inner hexagons are disjoint from each other and inside outer hexagon."""
        # Check inner hexagons are inside outer
        for i, (center, angle) in enumerate(inner_hexes):
            verts = get_hex_vertices(center, angle)
            for v in verts:
                if not hexagon_contains_point(outer_vertices, v):
                    return False
            # Also check center is inside
            if not hexagon_contains_point(outer_vertices, center):
                return False
        
        # Check inner hexagons are disjoint from each other
        for i in range(len(inner_hexes)):
            for j in range(i + 1, len(inner_hexes)):
                vi = get_hex_vertices(inner_hexes[i][0], inner_hexes[i][1])
                vj = get_hex_vertices(inner_hexes[j][0], inner_hexes[j][1])
                # Check if any edge of i intersects any edge of j
                # Simple check: if any vertex of i is inside j or vice versa
                for v in vi:
                    if hexagon_contains_point(vj, v):
                        return False
                for v in vj:
                    if hexagon_contains_point(vi, v):
                        return False
        return True
    
    def outer_hex_vertices(center, side, angle_deg):
        """Get vertices of outer hexagon."""
        a = angle_deg * np.pi / 180.0
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return center + side * (R @ hex_vertices.T).T
    
    def objective(params):
        """Optimize: minimize outer side length. Params: [cx, cy, side, outer_angle, x1, y1, a1, x2, y2, a2, ...]"""
        cx, cy = params[0], params[1]
        side = params[2]
        outer_angle = params[3]
        
        inner = []
        for k in range(N):
            x = params[4 + 3*k]
            y = params[5 + 3*k]
            ang = params[6 + 3*k]
            inner.append((np.array([x, y]), ang))
        
        outer_verts = outer_hex_vertices(np.array([cx, cy]), side, outer_angle)
        
        # Penalty if not all fit
        if not hexagons_disjoint(inner, outer_verts):
            return side + 100.0  # large penalty
        
        return side
    
    # Initial structured packing: centered hexagonal arrangement
    # For N=11, use a 3-3-3-2 arrangement or similar
    # Let's use a hexagonal number pattern: center + 6 around + 4 more
    # Or simply a 4-3-4 grid pattern
    
    # Better: use known compact packings for 11 hexagons
    # A 3x3 grid with corners removed = 5, not enough
    # Let's use: row0: 3, row1: 3, row2: 3, row3: 2 (offset)
    
    # Hexagonal lattice spacing for unit hexagons: sqrt(3)
    s = np.sqrt(3)
    
    # Define a compact arrangement for 11 hexagons
    # Center at origin, use hexagonal coordinates
    positions = []
    # Row 0 (top): 3 hexagons
    for i in range(3):
        positions.append((i * s, 0))
    # Row 1: 3 hexagons, offset
    for i in range(3):
        positions.append((i * s - s/2, s * np.sqrt(3)/2))
    # Row 2: 3 hexagons
    for i in range(3):
        positions.append((i * s, s * np.sqrt(3)))
    # Row 3: 2 hexagons
    for i in range(2):
        positions.append((i * s - s/2, s * np.sqrt(3) * 3/2))
    
    # Center the arrangement
    pos_array = np.array(positions)
    pos_array -= pos_array.mean(axis=0)
    
    # Initial guess for outer hexagon
    # Estimate needed side length
    max_dist = np.max(np.linalg.norm(pos_array, axis=1))
    initial_side = max_dist + 1.5  # some margin
    
    # Initial params
    init_params = [0.0, 0.0, initial_side, 0.0]
    for x, y in pos_array:
        init_params.extend([x, y, 0.0])
    
    # Use differential evolution for global search
    bounds = [(-5, 5), (-5, 5), (3, 10), (0, 60)]
    for k in range(N):
        bounds.extend([(-5, 5), (-5, 5), (0, 60)])
    
    best_result = None
    best_score = float('inf')
    
    # Try differential evolution
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=50, popsize=10, seed=42,
            tol=1e-4, init='sobol'
        )
        if time.time() < deadline:
            score = result.fun
            if score < best_score:
                best_score = score
                best_result = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        # Not enough time for polish, return best
        if best_result is not None:
            cx, cy, side, outer_angle = best_result[0], best_result[1], best_result[2], best_result[3]
            inner = []
            for k in range(N):
                x = best_result[4 + 3*k]
                y = best_result[5 + 3*k]
                ang = best_result[6 + 3*k]
                inner.append([x, y, ang])
            return inner, (cx, cy), side, outer_angle
        else:
            # Fallback
            inner = [[pos_array[k][0], pos_array[k][1], 0.0] for k in range(N)]
            return inner, (0.0, 0.0), initial_side, 0.0
    
    # Polish with local optimization
    if best_result is not None:
        try:
            polish_result = minimize(
                objective, best_result, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_result = polish_result.x
        except Exception:
            pass
    
    if best_result is not None:
        cx, cy, side, outer_angle = best_result[0], best_result[1], best_result[2], best_result[3]
        inner = []
        for k in range(N):
            x = best_result[4 + 3*k]
            y = best_result[5 + 3*k]
            ang = best_result[6 + 3*k]
            inner.append([x, y, ang])
        return inner, (cx, cy), side, outer_angle
    else:
        inner = [[pos_array[k][0], pos_array[k][1], 0.0] for k in range(N)]
        return inner, (0.0, 0.0), initial_side, 0.0

# EVOLVE-BLOCK-END
