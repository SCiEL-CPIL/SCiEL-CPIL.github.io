"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(poly1, poly2, tol=1e-9):
        """Check if two convex polygons are disjoint using SAT."""
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True
        return False

    def min_separation_distance(hex1_verts, hex2_verts):
        """
        Compute the minimum distance between two convex polygons.
        Returns negative if overlapping (approximate via SAT gap).
        Actually, let's compute the maximum gap found in SAT.
        If all gaps are negative, they overlap.
        """
        max_gap = -np.inf
        for poly in [hex1_verts, hex2_verts]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                proj1 = np.dot(hex1_verts, normal)
                proj2 = np.dot(hex2_verts, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Gap is min2 - max1 or min1 - max2
                gap1 = min2 - max1
                gap2 = min1 - max2
                gap = max(gap1, gap2)
                
                if gap > max_gap:
                    max_gap = gap
        
        return max_gap

    def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_required = 0.0
        theta = outer_angle_rad
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for k in range(6):
                    edge_angle = theta + np.radians(k * 60)
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def objective(params):
        """
        Objective: minimize outer side length + penalty for overlaps.
        params: [x0, y0, a0, ..., x11, y11, a11]
        Outer hexagon is centered at (0,0) with angle 0 for now, 
        but we can optimize center/angle separately or fix them.
        Let's fix outer center at (0,0) and optimize inner positions.
        Actually, let's include outer center and angle in params for flexibility.
        params: [x0, y0, a0, ..., x11, y11, a11, cx, cy, ctheta]
        """
        if time.time() > deadline:
            return 1e6
        
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        
        # Compute overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = min_separation_distance(inner_hexes[i], inner_hexes[j])
                if gap < 0:
                    # Overlapping, penalize quadratically
                    penalty += 10.0 * (-gap) ** 2
        
        # Compute outer side
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        
        return side + penalty

    # Initial configuration: Honeycomb cluster
    sqrt3 = np.sqrt(3)
    # 3 rows: 4, 4, 4? Or 5, 4, 3?
    # Let's try a compact cluster.
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # Total 12.
    
    init_x = []
    init_y = []
    init_a = []
    
    # Honeycomb spacing
    # Horizontal spacing: 1.5 (for pointy top) or sqrt(3) (for flat top)
    # Let's use flat top (angle 0).
    # Vertices at 0, 60, 120...
    # Width is 2, Height is sqrt(3).
    # To pack flat top hexagons:
    # Row y = 0: x = 0, 2, 4, 6 (spacing 2) -> wait, side 1, width 2.
    # Adjacent flat top hexagons in a row touch at edges? No, they touch at vertices if rotated.
    # For flat top, horizontal neighbors touch at vertical edges. Distance between centers = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) * 2? No.
    # Apothem = sqrt(3)/2. Width = 2.
    # Two flat top hexagons side by side: distance between centers = 2 * apothem = sqrt(3).
    # Wait, apothem is distance from center to edge.
    # If they touch edges, center distance = 2 * apothem = sqrt(3).
    # Vertical spacing: Row offset.
    # For flat top, rows are offset by 0.5 * width = 1.0 horizontally?
    # And vertically by 3/4 * height? Height = sqrt(3).
    # Let's just use a grid and let optimizer fix it.
    
    # Grid: 4 cols, 3 rows
    for r in range(3):
        for c in range(4):
            x = c * 1.8  # slightly larger than sqrt(3) ~ 1.732 to avoid initial overlap
            y = r * 1.6  # slightly larger than 1.5
            if r % 2 == 1:
                x += 0.9
            init_x.append(x)
            init_y.append(y)
            init_a.append(0.0)
    
    # Center the cluster
    init_x = np.array(init_x)
    init_y = np.array(init_y)
    init_x -= np.mean(init_x)
    init_y -= np.mean(init_y)
    
    init_params = np.array(init_x + init_y + init_a + [0.0, 0.0, 0.0])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((-np.pi, np.pi))
    bounds.append((-5, 5))
    bounds.append((-5, 5))
    bounds.append((-np.pi, np.pi))
    
    best_params = init_params.copy()
    best_obj = objective(init_params)
    
    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 60:
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=30, popsize=15,
                tol=1e-4, mutation=(0.5, 1.0),
                recombination=0.7, seed=42,
                polish=False  # Skip polish, we'll do it later
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract solution
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    outer_cx = params[36]
    outer_cy = params[37]
    outer_a = params[38]
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    outer_side = best_obj  # This includes penalty, but if penalty is 0, it's the side.
    
    # Verify validity
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not polygons_disjoint_sat(inner_hexes[i], inner_hexes[j], 1e-7):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Recompute exact side length without penalty
    exact_side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
    
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
