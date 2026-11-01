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
    
    # Define multiple initial configurations (Structural Change: Multi-Start Local Search)
    configs = []
    
    # Config 1: Honeycomb (Flat Top)
    hx, hy, ha = [], [], []
    sqrt3 = np.sqrt(3)
    for r in range(3):
        for c in range(4):
            x = c * 1.74 # ~sqrt(3)
            y = r * 1.5
            if r % 2 == 1:
                x += 0.87
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))
    
    # Config 2: Square Grid
    sx, sy, sa = [], [], []
    for r in range(3):
        for c in range(4):
            sx.append(c * 2.1)
            sy.append(r * 2.1)
            sa.append(0.0)
    sx = np.array(sx); sy = np.array(sy); sa = np.array(sa)
    sx -= np.mean(sx); sy -= np.mean(sy)
    configs.append(np.concatenate([sx, sy, sa, [0.0, 0.0, 0.0]]))
    
    # Config 3: Random Cluster
    rx, ry, ra = [], [], []
    for i in range(N):
        rx.append(np.random.uniform(-3, 3))
        ry.append(np.random.uniform(-3, 3))
        ra.append(np.random.uniform(-np.pi, np.pi))
    rx = np.array(rx); ry = np.array(ry); ra = np.array(ra)
    rx -= np.mean(rx); ry -= np.mean(ry)
    configs.append(np.concatenate([rx, ry, ra, [0.0, 0.0, 0.0]]))

    best_params = configs[0].copy()
    best_obj = objective(best_params)
    
    # Multi-Start Nelder-Mead
    for cfg in configs:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(
                objective, cfg, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
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
