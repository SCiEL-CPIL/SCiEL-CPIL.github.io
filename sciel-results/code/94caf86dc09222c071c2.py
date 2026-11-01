"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def rotate_points(points, angle_rad):
        """Rotate points around origin by angle_rad."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return points @ rot.T

    def required_side_for_cluster(cluster_positions, cluster_rot, outer_rot):
        """Compute the minimum outer hexagon side length to contain all hexagons
        in the cluster after rotating the cluster by cluster_rot and the outer by outer_rot.
        cluster_positions: (12, 2) array of hex centers in cluster frame
        """
        # Rotate cluster positions
        rotated_centers = rotate_points(cluster_positions, cluster_rot)
        
        # For each hex, get its vertices in world frame
        max_required = 0.0
        for i in range(N):
            cx, cy = rotated_centers[i]
            # Hex vertices relative to its center (unit hex, orientation 0 in cluster frame)
            local_angles = np.radians(np.arange(0, 360, 60))
            local_verts = np.column_stack([np.cos(local_angles), np.sin(local_angles)])
            # Rotate local vertices by cluster rotation
            world_verts = local_verts @ np.array([[np.cos(cluster_rot), -np.sin(cluster_rot)],
                                                   [np.sin(cluster_rot), np.cos(cluster_rot)]]).T
            world_verts = world_verts + np.array([cx, cy])
            
            # Check against outer hexagon constraints
            for k in range(6):
                edge_angle = outer_rot + np.radians(k * 60)
                nx = np.cos(edge_angle)
                ny = np.sin(edge_angle)
                dists = world_verts[:, 0] * nx + world_verts[:, 1] * ny
                max_dist = np.max(dists)
                if max_dist > 0:
                    required_S = max_dist * 2.0 / np.sqrt(3.0)
                    if required_S > max_required:
                        max_required = required_S
        return max_required

    def generate_cluster_configs():
        """Generate several dense cluster configurations for 12 unit hexagons."""
        configs = []
        sqrt3 = np.sqrt(3)
        
        # Config 1: 3x4 honeycomb (all aligned, orientation 0)
        hx, hy = [], []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                hx.append(x); hy.append(y)
        pos = np.column_stack([hx, hy])
        pos -= pos.mean(axis=0)
        configs.append(('honeycomb_3x4', pos, np.zeros(N)))
        
        # Config 2: 4x3 honeycomb
        hx, hy = [], []
        for r in range(4):
            for c in range(3):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                hx.append(x); hy.append(y)
        pos = np.column_stack([hx, hy])
        pos -= pos.mean(axis=0)
        configs.append(('honeycomb_4x3', pos, np.zeros(N)))
        
        # Config 3: 2x6 honeycomb
        hx, hy = [], []
        for r in range(2):
            for c in range(6):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                hx.append(x); hy.append(y)
        pos = np.column_stack([hx, hy])
        pos -= pos.mean(axis=0)
        configs.append(('honeycomb_2x6', pos, np.zeros(N)))
        
        # Config 4: Center + ring of 11 (approximate)
        # Use hexagonal packing: center + 6 + 5 more
        hx, hy = [0.0], [0.0]
        # Ring of 6 at distance sqrt(3)
        for k in range(6):
            ang = np.radians(k * 60)
            hx.append(sqrt3 * np.cos(ang))
            hy.append(sqrt3 * np.sin(ang))
        # Add 5 more in second ring (select 5 from 12 positions)
        for k in range(12):
            if len(hx) >= N:
                break
            # Second ring positions
            base_ang = np.radians(k * 30 + 30)
            r2 = 2 * sqrt3
            x2 = r2 * np.cos(base_ang)
            y2 = r2 * np.sin(base_ang)
            # Check if not too close to existing
            too_close = False
            for i in range(len(hx)):
                dx = x2 - hx[i]
                dy = y2 - hy[i]
                if dx*dx + dy*dy < (sqrt3 - 0.1)**2:
                    too_close = True
                    break
            if not too_close:
                hx.append(x2)
                hy.append(y2)
        # Fill remaining if needed
        while len(hx) < N:
            ang = np.radians(15 + 30 * len(hx))
            r = 2.5 * sqrt3
            hx.append(r * np.cos(ang))
            hy.append(r * np.sin(ang))
        pos = np.column_stack([hx[:N], hy[:N]])
        pos -= pos.mean(axis=0)
        configs.append(('center_ring', pos, np.zeros(N)))
        
        # Config 5: Honeycomb with 30° rotated hexagons
        hx, hy = [], []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3 * 1.02
                y = r * 1.5 * 1.02
                if r % 2 == 1:
                    x += sqrt3 * 1.02 / 2
                hx.append(x); hy.append(y)
        pos = np.column_stack([hx, hy])
        pos -= pos.mean(axis=0)
        configs.append(('honeycomb_3x4_rot', pos, np.full(N, np.pi/6)))
        
        # Config 6: Alternating orientations in honeycomb
        hx, hy = [], []
        orients = []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3 * 1.01
                y = r * 1.5 * 1.01
                if r % 2 == 1:
                    x += sqrt3 * 1.01 / 2
                hx.append(x); hy.append(y)
                orients.append(0.0 if (r + c) % 2 == 0 else np.pi/6)
        pos = np.column_stack([hx, hy])
        pos -= pos.mean(axis=0)
        configs.append(('honeycomb_alt', pos, np.array(orients)))
        
        # Config 7: Compact diamond shape
        hx, hy = [], []
        positions_diamond = [
            (0, 0), (sqrt3, 0), (-sqrt3, 0),
            (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
            (3*sqrt3/2, 1.5), (-3*sqrt3/2, 1.5),
            (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
            (3*sqrt3/2, -1.5), (-3*sqrt3/2, -1.5),
            (sqrt3, 3.0), (-sqrt3, 3.0),
        ]
        for p in positions_diamond[:N]:
            hx.append(p[0]); hy.append(p[1])
        pos = np.column_stack([hx, hy])
        pos -= pos.mean(axis=0)
        configs.append(('diamond', pos, np.zeros(N)))
        
        return configs

    def optimize_2d(cluster_pos, cluster_orients, n_grid=72):
        """Optimize (cluster_rotation, outer_rotation) in 2D for a given cluster."""
        best_side = np.inf
        best_cr = 0.0
        best_or = 0.0
        
        # Grid search over cluster rotation and outer rotation
        # Both have 60° periodicity, so search [0, 60°)
        cr_vals = np.linspace(0, np.pi/3, n_grid, endpoint=False)
        or_vals = np.linspace(0, np.pi/3, n_grid, endpoint=False)
        
        for cr in cr_vals:
            for orr in or_vals:
                side = required_side_for_cluster(cluster_pos, cr, orr)
                if side < best_side:
                    best_side = side
                    best_cr = cr
                    best_or = orr
        
        # Local refinement using minimize
        def obj_2d(params):
            cr, orr = params[0], params[1]
            return required_side_for_cluster(cluster_pos, cr, orr)
        
        try:
            result = minimize(
                obj_2d, [best_cr, best_or], method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_side:
                best_side = result.fun
                best_cr = result.x[0]
                best_or = result.x[1]
        except Exception:
            pass
        
        return best_side, best_cr, best_or

    # Generate all cluster configs
    cluster_configs = generate_cluster_configs()
    
    best_overall_side = np.inf
    best_cluster_pos = None
    best_cluster_orients = None
    best_cluster_rot = 0.0
    best_outer_rot = 0.0
    best_cluster_name = ''
    
    # Optimize each cluster config
    for name, cpos, corients in cluster_configs:
        if time.time() > deadline - 10:
            break
        
        side, cr, orr = optimize_2d(cpos, corients)
        if side < best_overall_side:
            best_overall_side = side
            best_cluster_pos = cpos.copy()
            best_cluster_orients = corients.copy()
            best_cluster_rot = cr
            best_outer_rot = orr
            best_cluster_name = name
    
    # Fine-tune: try more grid points around the best solution
    if time.time() < deadline - 5:
        # Refined grid around best
        cr_center = best_cluster_rot
        or_center = best_outer_rot
        cr_fine = np.linspace(cr_center - 0.05, cr_center + 0.05, 30)
        or_fine = np.linspace(or_center - 0.05, or_center + 0.05, 30)
        
        for cr in cr_fine:
            for orr in or_fine:
                side = required_side_for_cluster(best_cluster_pos, cr, orr)
                if side < best_overall_side:
                    best_overall_side = side
                    best_cluster_rot = cr
                    best_outer_rot = orr
    
    # Final local optimization
    if time.time() < deadline - 3:
        def obj_final(params):
            cr, orr = params
            return required_side_for_cluster(best_cluster_pos, cr, orr)
        
        try:
            result = minimize(
                obj_final, [best_cluster_rot, best_outer_rot], method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_overall_side:
                best_overall_side = result.fun
                best_cluster_rot = result.x[0]
                best_outer_rot = result.x[1]
        except Exception:
            pass
    
    # Compute final positions
    rotated_centers = rotate_points(best_cluster_pos, best_cluster_rot)
    
    # Apply per-hex orientation (cluster orientation + cluster rotation)
    inner = []
    for i in range(N):
        cx, cy = rotated_centers[i]
        # The hex orientation in world frame = cluster_orient + cluster_rot
        world_orient = best_cluster_orients[i] + best_cluster_rot
        inner.append([cx, cy, np.degrees(world_orient)])
    
    # Verify validity
    def min_separation(cx1, cy1, a1, cx2, cy2, a2):
        """Compute minimum separation between two unit hexagons."""
        verts1 = hexagon_vertices(cx1, cy1, 1.0, a1)
        verts2 = hexagon_vertices(cx2, cy2, 1.0, a2)
        edge_angles1 = a1 + np.radians(np.arange(0, 360, 60))
        edge_angles2 = a2 + np.radians(np.arange(0, 360, 60))
        normals1 = np.column_stack([np.cos(edge_angles1 + np.pi/2), np.sin(edge_angles1 + np.pi/2)])
        normals2 = np.column_stack([np.cos(edge_angles2 + np.pi/2), np.sin(edge_angles2 + np.pi/2)])
        all_normals = np.vstack([normals1, normals2])
        min_gap = np.inf
        for normal in all_normals:
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            d_center = np.dot(np.array([cx1, cy1]) - np.array([cx2, cy2]), normal)
            if d_center >= 0:
                gap = np.min(proj2) - np.max(proj1)
            else:
                gap = np.min(proj1) - np.max(proj2)
            if gap < min_gap:
                min_gap = gap
        return min_gap
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            sep = min_separation(
                inner[i][0], inner[i][1], np.radians(inner[i][2]),
                inner[j][0], inner[j][1], np.radians(inner[j][2])
            )
            if sep < -1e-4:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    return inner, (0.0, 0.0), best_overall_side, np.degrees(best_outer_rot)
# EVOLVE-BLOCK-END
