import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hexagon_vertices(cx, cy, side, rotation_deg=0.0):
        rot = math.radians(rotation_deg)
        verts = []
        for i in range(6):
            ang = rot + i * math.pi / 3.0
            verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, side, rotation_deg=0.0):
        rot = math.radians(rotation_deg)
        # Transform point to hexagon local frame
        dx, dy = px - cx, py - cy
        # Rotate point by -rot
        cos_r, sin_r = math.cos(-rot), math.sin(-rot)
        lx = dx * cos_r - dy * sin_r
        ly = dx * sin_r + dy * cos_r
        # For a regular hexagon with flat-top orientation (vertex at angle 0),
        # the six half-plane inequalities are:
        # |x| <= side, |x/2 + y*sqrt(3)/2| <= side, |-x/2 + y*sqrt(3)/2| <= side
        # But our hexagon has vertices at angles rot + i*pi/3
        # So in local frame, vertices are at angles i*pi/3
        # The apothem is side*sqrt(3)/2
        # Six normals at angles pi/6 + i*pi/3
        max_proj = 0.0
        for i in range(6):
            ang = math.pi / 6.0 + i * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            proj = lx * nx + ly * ny
            max_proj = max(max_proj, abs(proj))
        return max_proj <= side * APOTHEM + 1e-9

    def hexagon_hexagon_disjoint(h1_center, h1_rot, h2_center, h2_rot, side=1.0):
        """Check if two unit hexagons are disjoint (non-overlapping).
        We check if any vertex of one is inside the other, and vice versa.
        Also check edge intersections for robustness."""
        h1_verts = hexagon_vertices(h1_center[0], h1_center[1], side, h1_rot)
        h2_verts = hexagon_vertices(h2_center[0], h2_center[1], side, h2_rot)
        
        # Check if any vertex of h1 is inside h2
        for vx, vy in h1_verts:
            if point_in_hexagon(vx, vy, h2_center[0], h2_center[1], side, h2_rot):
                return False
        # Check if any vertex of h2 is inside h1
        for vx, vy in h2_verts:
            if point_in_hexagon(vx, vy, h1_center[0], h1_center[1], side, h1_rot):
                return False
        
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def orient(a, b, c):
                val = (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
                if abs(val) < 1e-12:
                    return 0
                return 1 if val > 0 else -1
            o1 = orient(p1, p2, p3)
            o2 = orient(p1, p2, p4)
            o3 = orient(p3, p4, p1)
            o4 = orient(p3, p4, p2)
            if o1 != o2 and o3 != o4:
                return True
            return False
        
        for i in range(6):
            p1 = h1_verts[i]
            p2 = h1_verts[(i+1) % 6]
            for j in range(6):
                p3 = h2_verts[j]
                p4 = h2_verts[(j+1) % 6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def compute_min_enclosing_hexagon(pts, angle_deg):
        """Compute the minimal side length of a regular hexagon (with given rotation)
        that encloses all points. The hexagon center is also optimized."""
        rot = math.radians(angle_deg)
        
        def objective(params):
            cx, cy = params
            max_proj = 0.0
            for i in range(6):
                ang = rot + math.pi / 6.0 + i * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            S = (max_proj + APOTHEM) / APOTHEM
            return S
        
        # Initial guess: centroid of points
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        
        # Use Nelder-Mead for center optimization
        try:
            res = minimize(objective, [cx0, cy0], method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < 1e10:
                return res.fun, (res.x[0], res.x[1])
        except:
            pass
        return objective([cx0, cy0]), (cx0, cy0)

    def compute_min_enclosing_hexagon_full(pts):
        """Find the minimal enclosing hexagon over all rotations and centers."""
        best_S = float('inf')
        best_angle = 0.0
        best_center = (0, 0)
        
        # Coarse search over angles
        for i in range(37):
            ang = i * (math.pi / 6) / 36
            S, center = compute_min_enclosing_hexagon(pts, math.degrees(ang))
            if S < best_S:
                best_S = S
                best_angle = ang
                best_center = center
        
        # Refine with differential evolution over angle
        try:
            def full_obj(params):
                ang = params[0] % (math.pi / 6)
                S, _ = compute_min_enclosing_hexagon(pts, math.degrees(ang))
                return S
            
            res = differential_evolution(full_obj, bounds=[(0, math.pi/6)], 
                                         maxiter=50, popsize=10, seed=42,
                                         tol=1e-10, polish=True)
            if res.fun < best_S:
                best_angle = res.x[0] % (math.pi / 6)
                best_S, best_center = compute_min_enclosing_hexagon(pts, math.degrees(best_angle))
        except:
            pass
        
        return best_S, best_angle, best_center

    def force_directed_relaxation(initial_pts, num_iter=2000, dt=0.01, 
                                  repulse_k=2.0, attract_k=0.5, 
                                  compact_k=0.3, min_dist=2.0, 
                                  target_dist=2.1):
        """Force-directed relaxation to find a dense packing of hexagons.
        Forces:
        1. Repulsion when hexagons are too close (to prevent overlap)
        2. Attraction to maintain target distance (for density)
        3. Compactness force pulling toward centroid
        """
        pts = initial_pts.copy().astype(float)
        n = len(pts)
        
        for it in range(num_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            forces = np.zeros_like(pts)
            centroid = np.mean(pts, axis=0)
            
            # Pairwise forces
            for i in range(n):
                for j in range(i+1, n):
                    d = pts[i] - pts[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-10:
                        d = np.array([1e-10, 1e-10])
                        dist = np.linalg.norm(d)
                    d_norm = d / dist
                    
                    # Repulsive force when too close
                    if dist < min_dist:
                        f_rep = repulse_k * (min_dist - dist)
                        forces[i] += f_rep * d_norm
                        forces[j] -= f_rep * d_norm
                    # Attractive force to target distance
                    elif dist > target_dist and dist < target_dist * 1.5:
                        f_att = attract_k * (dist - target_dist)
                        forces[i] -= f_att * d_norm
                        forces[j] += f_att * d_norm
            
            # Compactness: pull toward centroid
            for i in range(n):
                forces[i] += compact_k * (centroid - pts[i])
            
            # Update positions with damping
            damping = 1.0 - 0.01 * (it / num_iter)
            pts += dt * forces * damping
            
            # Keep points within reasonable bounds
            max_coord = 10.0
            pts = np.clip(pts, -max_coord, max_coord)
        
        return pts

    def evaluate_packing(pts):
        """Evaluate a packing: check disjointness and compute enclosing hexagon size."""
        n = len(pts)
        # All hexagons have the same rotation (0 degrees for simplicity in relaxation)
        # Check pairwise disjointness
        for i in range(n):
            for j in range(i+1, n):
                if not hexagon_hexagon_disjoint(pts[i], 0.0, pts[j], 0.0):
                    return None  # Overlap
        S, angle, center = compute_min_enclosing_hexagon_full(pts)
        return S, angle, center

    def generate_initial_configurations():
        """Generate diverse initial configurations for relaxation."""
        configs = []
        
        # Configuration 1: Hexagonal lattice arrangement
        axial_1 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        configs.append(axial_to_cartesian(axial_1))
        
        # Configuration 2: Linear arrangement
        axial_2 = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (1, 1), (2, 1), (3, 1),
            (2, 2), (3, 2), (4, 2)
        ]
        configs.append(axial_to_cartesian(axial_2))
        
        # Configuration 3: Clustered
        axial_3 = [
            (0, 0), (1, 0), (-1, 0),
            (0, 1), (1, 1), (-1, 1),
            (0, -1), (1, -1), (-1, -1),
            (2, 0), (-2, 0)
        ]
        configs.append(axial_to_cartesian(axial_3))
        
        # Configuration 4: Random cluster
        np.random.seed(42)
        random_pts = np.random.randn(N, 2) * 1.5
        configs.append(random_pts)
        
        # Configuration 5: Ring + center
        ring_pts = []
        for i in range(6):
            ang = i * math.pi / 3
            ring_pts.append([2.0 * math.cos(ang), 2.0 * math.sin(ang)])
        ring_pts.append([0, 0])
        ring_pts.append([3.5, 0])
        ring_pts.append([0, 3.5])
        ring_pts.append([0, -3.5])
        ring_pts.append([-3.5, 0])
        configs.append(np.array(ring_pts))
        
        # Configuration 6: Staggered rows
        axial_6 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2)
        ]
        configs.append(axial_to_cartesian(axial_6))
        
        return configs

    best_global_S = float('inf')
    best_global_pts = None
    best_global_angle = 0.0
    best_global_center = None

    initial_configs = generate_initial_configurations()
    
    for config in initial_configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Try multiple relaxation parameters
        param_sets = [
            {'repulse_k': 2.0, 'attract_k': 0.5, 'compact_k': 0.3, 'min_dist': 2.0, 'target_dist': 2.1, 'num_iter': 3000},
            {'repulse_k': 3.0, 'attract_k': 0.8, 'compact_k': 0.5, 'min_dist': 2.0, 'target_dist': 2.2, 'num_iter': 3000},
            {'repulse_k': 1.5, 'attract_k': 0.3, 'compact_k': 0.2, 'min_dist': 1.95, 'target_dist': 2.05, 'num_iter': 4000},
            {'repulse_k': 4.0, 'attract_k': 1.0, 'compact_k': 0.6, 'min_dist': 2.0, 'target_dist': 2.3, 'num_iter': 2000},
        ]
        
        for params in param_sets:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            relaxed = force_directed_relaxation(
                config, 
                num_iter=params['num_iter'],
                repulse_k=params['repulse_k'],
                attract_k=params['attract_k'],
                compact_k=params['compact_k'],
                min_dist=params['min_dist'],
                target_dist=params['target_dist']
            )
            
            # Evaluate the relaxed configuration
            result = evaluate_packing(relaxed)
            if result is not None:
                S, angle, center = result
                if S < best_global_S:
                    best_global_S = S
                    best_global_pts = relaxed.copy()
                    best_global_angle = angle
                    best_global_center = center
            
            # Also try a local optimization on the relaxed positions
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            def local_obj(params_flat):
                pts_local = params_flat.reshape(N, 2)
                # Check disjointness
                for i in range(N):
                    for j in range(i+1, N):
                        d = pts_local[i] - pts_local[j]
                        dist = np.linalg.norm(d)
                        if dist < 1.99:  # Allow slight tolerance
                            return 1000.0 + (1.99 - dist) * 100
                S, _, _ = compute_min_enclosing_hexagon_full(pts_local)
                return S
            
            try:
                res = minimize(local_obj, relaxed.ravel(), method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < 1000:
                    pts_opt = res.x.reshape(N, 2)
                    result = evaluate_packing(pts_opt)
                    if result is not None:
                        S, angle, center = result
                        if S < best_global_S:
                            best_global_S = S
                            best_global_pts = pts_opt.copy()
                            best_global_angle = angle
                            best_global_center = center
            except:
                pass

    # If no valid packing found, use a fallback
    if best_global_pts is None or best_global_S == float('inf'):
        axial_fallback = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        best_global_pts = axial_to_cartesian(axial_fallback)
        best_global_angle = 0.0
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_S, _, _ = compute_min_enclosing_hexagon_full(best_global_pts)

    # Final evaluation with the best configuration
    final_S, final_angle, final_center = compute_min_enclosing_hexagon_full(best_global_pts)
    if final_S < best_global_S:
        best_global_S = final_S
        best_global_angle = final_angle
        best_global_center = final_center

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_angle)
    
    # Build output
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
