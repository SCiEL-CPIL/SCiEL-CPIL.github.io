import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3.0)

def construct():
    t0 = time.time()
    deadline = 145.0

    # Unit hexagon vertices (flat-top, vertex at angle 0)
    unit_hex = np.array([
        [1.0, 0.0],
        [0.5, S3/2],
        [-0.5, S3/2],
        [-1.0, 0.0],
        [-0.5, -S3/2],
        [0.5, -S3/2]
    ])

    # Outer hexagon normals (flat-top)
    normals = np.array([
        [1.0, 0.0],
        [0.5, S3/2],
        [-0.5, S3/2],
        [-1.0, 0.0],
        [-0.5, -S3/2],
        [0.5, -S3/2]
    ])

    def rot_mat(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def get_verts(center, angle_deg):
        th = np.deg2rad(angle_deg)
        R = rot_mat(th)
        return unit_hex @ R.T + np.array(center)

    def outer_side(centers, angles):
        """Minimum outer hexagon side length (flat-top, centered at origin)."""
        max_ap = 0.0
        for k in range(len(centers)):
            v = get_verts(centers[k], angles[k])
            projs = v @ normals.T
            max_ap = max(max_ap, np.max(projs))
        return 2.0 * max_ap / S3

    def separation_violation(centers):
        """Sum of squared overlaps using center-distance heuristic (min dist = sqrt(3))."""
        min_d = S3 * 0.9999
        viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_d:
                    viol += (min_d - d)**2
        return viol

    def exact_nonoverlap(centers, angles, tol=1e-9):
        """Check pairwise non-overlap via vertex-edge tests."""
        verts_list = [get_verts(centers[k], angles[k]) for k in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if _hex_overlap(verts_list[i], verts_list[j], tol):
                    return False
        return True

    def _hex_overlap(v1, v2, tol=1e-9):
        """Return True if two convex hexagons overlap (interior intersection)."""
        # Check if any vertex of v1 is strictly inside v2 or vice versa
        if _point_in_convex(v1[0], v2, tol) or _point_in_convex(v2[0], v1, tol):
            return True
        # Check edge intersections
        for i in range(6):
            a1, a2 = v1[i], v1[(i+1)%6]
            for j in range(6):
                b1, b2 = v2[j], v2[(j+1)%6]
                if _segments_intersect(a1, a2, b1, b2, tol):
                    return True
        return False

    def _point_in_convex(p, poly, tol=1e-9):
        """Check if point p is inside convex polygon poly (strictly interior)."""
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i+1)%n]
            cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
            if cross < -tol:
                return False
        return True

    def _segments_intersect(p1, p2, p3, p4, tol=1e-9):
        """Check if segments p1-p2 and p3-p4 intersect (proper or touching)."""
        d1 = _orient(p3, p4, p1)
        d2 = _orient(p3, p4, p2)
        d3 = _orient(p1, p2, p3)
        d4 = _orient(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        if abs(d1) <= tol and _on_segment(p3, p4, p1, tol): return True
        if abs(d2) <= tol and _on_segment(p3, p4, p2, tol): return True
        if abs(d3) <= tol and _on_segment(p1, p2, p3, tol): return True
        if abs(d4) <= tol and _on_segment(p1, p2, p4, tol): return True
        return False

    def _orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def _on_segment(a, b, c, tol=1e-9):
        return (min(a[0],b[0])-tol <= c[0] <= max(a[0],b[0])+tol and
                min(a[1],b[1])-tol <= c[1] <= max(a[1],b[1])+tol)

    def inside_outer(centers, angles, side):
        """Check all vertices are inside outer hexagon of given side length."""
        apothem = side * S3 / 2.0
        for k in range(N):
            v = get_verts(centers[k], angles[k])
            projs = v @ normals.T
            if np.any(projs > apothem + 1e-9):
                return False
        return True

    def valid(centers, angles, side):
        return exact_nonoverlap(centers, angles) and inside_outer(centers, angles, side)

    # --- Rigid cluster definitions ---
    # Each cluster is defined by N relative positions (from centroid) with fixed angles.
    # We optimize: cluster rotation theta, translation (tx, ty)
    # The outer hexagon is centered at origin, flat-top.

    d = S3  # adjacent center distance

    def make_clusters():
        """Generate several rigid cluster configurations."""
        clusters = []

        # Cluster A: 7-hex flower (hexagonal close-packed)
        # Center + 6 around
        pos_a = np.array([
            [0, 0],
            [d, 0], [d/2, 1.5], [-d/2, 1.5],
            [-d, 0], [-d/2, -1.5], [d/2, -1.5]
        ])
        # Add 4 more at second ring positions (choose 4 of 12)
        # Second ring: at distance 2d in 6 directions, and at distance sqrt(3)*d in 6 diagonal directions
        # Let's add 4 hexagons to make a compact shape
        # Positions for 11: use a "thickened" hexagon
        # Row-based: 3-4-4 or similar
        # Actually let's do: center row 5, upper row 3, lower row 3
        pos_a2 = []
        # Bottom row: 3 hexes
        for i in range(3):
            pos_a2.append([i*d - d, -1.5])
        # Middle row: 5 hexes
        for i in range(5):
            pos_a2.append([i*d - 2*d, 0])
        # Top row: 3 hexes
        for i in range(3):
            pos_a2.append([i*d - d, 1.5])
        pos_a2 = np.array(pos_a2[:N])
        clusters.append(("row_353", pos_a2, np.zeros(N)))

        # Cluster B: 4-3-4 arrangement
        pos_b = []
        for i in range(4):
            pos_b.append([i*d - 1.5*d, 0])
        for i in range(3):
            pos_b.append([i*d - d + d/2, 1.5])
        for i in range(4):
            pos_b.append([i*d - 1.5*d, 3.0])
        pos_b = np.array(pos_b[:N])
        clusters.append(("row_434", pos_b, np.zeros(N)))

        # Cluster C: Compact triangular (3-4-4)
        pos_c = []
        for i in range(3):
            pos_c.append([i*d - d, 0])
        for i in range(4):
            pos_c.append([i*d - 1.5*d + d/2, 1.5])
        for i in range(4):
            pos_c.append([i*d - 1.5*d + d/2, 3.0])
        pos_c = np.array(pos_c[:N])
        clusters.append(("tri_344", pos_c, np.zeros(N)))

        # Cluster D: 5-3-3 (wider bottom)
        pos_d = []
        for i in range(5):
            pos_d.append([i*d - 2*d, 0])
        for i in range(3):
            pos_d.append([i*d - d + d/2, 1.5])
        for i in range(3):
            pos_d.append([i*d - d + d/2, 3.0])
        pos_d = np.array(pos_d[:N])
        clusters.append(("row_533", pos_d, np.zeros(N)))

        # Cluster E: 2-3-3-3 (tall)
        pos_e = []
        for i in range(2):
            pos_e.append([i*d - 0.5*d, 0])
        for i in range(3):
            pos_e.append([i*d - d + d/2, 1.5])
        for i in range(3):
            pos_e.append([i*d - d + d/2, 3.0])
        for i in range(3):
            pos_e.append([i*d - d + d/2, 4.5])
        pos_e = np.array(pos_e[:N])
        clusters.append(("tall_2333", pos_e, np.zeros(N)))

        # Cluster F: Hexagonal flower + 4 satellites (7+4)
        pos_f = np.array([
            [0, 0],
            [d, 0], [d/2, 1.5], [-d/2, 1.5],
            [-d, 0], [-d/2, -1.5], [d/2, -1.5],
            [2*d, 0], [1.5*d, 1.5*1.5], [-1.5*d, 1.5*1.5],
            [-2*d, 0]
        ])
        clusters.append(("flower_74", pos_f, np.zeros(N)))

        # Cluster G: Staggered rows 4-4-3
        pos_g = []
        for i in range(4):
            pos_g.append([i*d - 1.5*d, 0])
        for i in range(4):
            pos_g.append([i*d - 1.5*d + d/2, 1.5])
        for i in range(3):
            pos_g.append([i*d - d + d/2, 3.0])
        pos_g = np.array(pos_g[:N])
        clusters.append(("stagger_443", pos_g, np.zeros(N)))

        return clusters

    def cluster_objective(params, rel_pos, rel_angles):
        """Objective for rigid cluster: theta, tx, ty.
        All hexagons rotate by theta, translate by (tx,ty)."""
        theta, tx, ty = params
        R = rot_mat(theta)
        centers = rel_pos @ R.T + np.array([tx, ty])
        angles = rel_angles + np.degrees(theta)
        
        # Compute outer side
        side = outer_side(centers, angles)
        
        # Check overlap (use fast distance check first)
        sv = separation_violation(centers)
        if sv > 1e-6:
            return side + 1e6 * sv
        
        return side

    def optimize_cluster(rel_pos, rel_angles, time_budget=20.0):
        """Optimize rotation and translation for a rigid cluster."""
        t_start = time.time()
        
        # Coarse grid search over theta
        best_result = None
        best_obj = float('inf')
        
        # Try many rotation angles
        theta_range = np.linspace(0, np.pi/3, 13)  # 0 to 60 degrees (hex symmetry)
        
        for th0 in theta_range:
            if time.time() - t_start > time_budget * 0.5:
                break
            # For each theta, optimize translation
            def trans_obj(params):
                tx, ty = params
                obj = cluster_objective([th0, tx, ty], rel_pos, rel_angles)
                return obj
            
            # Initial guess: center of mass at origin
            com = np.mean(rel_pos, axis=0)
            x0 = [-com[0]*np.cos(th0) + com[1]*np.sin(th0),
                  -com[0]*np.sin(th0) - com[1]*np.cos(th0)]
            
            res = minimize(trans_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = (th0, res.x[0], res.x[1])
        
        # Refine with Nelder-Mead on all 3 params
        if best_result is not None:
            x0 = np.array(best_result)
            res = minimize(lambda p: cluster_objective(p, rel_pos, rel_angles), x0,
                          method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        
        return best_result, best_obj

    def get_config_from_params(params, rel_pos, rel_angles):
        """Get centers and angles from cluster params."""
        theta, tx, ty = params
        R = rot_mat(theta)
        centers = rel_pos @ R.T + np.array([tx, ty])
        angles = rel_angles + np.degrees(theta)
        return centers, angles

    # Generate clusters
    clusters = make_clusters()
    
    best_side = float('inf')
    best_centers = None
    best_angles = None
    
    # Optimize each cluster
    for name, rel_pos, rel_angles in clusters:
        if time.time() > deadline - 15:
            break
        
        t_cluster_start = time.time()
        remaining = deadline - 15 - time.time()
        budget = min(20.0, remaining / max(1, len(clusters)))
        
        params, obj = optimize_cluster(rel_pos, rel_angles, time_budget=budget)
        
        if params is not None:
            centers, angles = get_config_from_params(params, rel_pos, rel_angles)
            
            # Verify with exact overlap check
            if exact_nonoverlap(centers, angles):
                side = outer_side(centers, angles)
                if inside_outer(centers, angles, side):
                    if side < best_side:
                        best_side = side
                        best_centers = centers.copy()
                        best_angles = angles.copy()
    
    # If no valid cluster found, try free optimization with scipy
    if best_centers is None and time.time() < deadline - 10:
        # Use differential evolution on free positions
        def free_obj(params):
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            side = outer_side(centers, angles)
            sv = separation_violation(centers)
            if sv > 1e-6:
                return side + 1e6 * sv
            return side
        
        # Initial bounds
        bounds = []
        for i in range(N):
            bounds.append((-4, 4))  # x
            bounds.append((-4, 4))  # y
        for i in range(N):
            bounds.append((-30, 30))  # angle
        
        # Use a compact initial guess
        x0 = np.zeros(3*N)
        d = S3
        for i in range(N):
            row = i // 4
            col = i % 4
            x0[2*i] = col*d - 1.5*d
            x0[2*i+1] = row*1.5
            x0[2*N+i] = 0
        
        # Nelder-Mead on free params (subset for speed)
        res = minimize(free_obj, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
        
        centers = res.x[:2*N].reshape(N, 2)
        angles = res.x[2*N:]
        
        if exact_nonoverlap(centers, angles):
            side = outer_side(centers, angles)
            if inside_outer(centers, angles, side):
                if side < best_side:
                    best_side = side
                    best_centers = centers.copy()
                    best_angles = angles.copy()
    
    # Final polish: if we have a valid config, do fine-tuning
    if best_centers is not None and time.time() < deadline - 5:
        # Try small perturbations
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_side = best_side
        
        step = 0.01
        for it in range(5000):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step)
            dy = np.random.normal(0, step)
            da = np.random.normal(0, step * 3)
            
            new_c = current_centers.copy()
            new_a = current_angles.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
            new_a[idx] += da
            
            if separation_violation(new_c) > 1e-6:
                step *= 0.98
                continue
            
            new_side = outer_side(new_c, new_a)
            if new_side < current_side:
                current_centers = new_c
                current_angles = new_a
                current_side = new_side
                step = min(step * 1.05, 0.05)
            else:
                step *= 0.98
        
        if current_side < best_side:
            best_side = current_side
            best_centers = current_centers
            best_angles = current_angles
    
    # Output
    if best_centers is None:
        # Fallback: spread out
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final validation
    if not exact_nonoverlap(best_centers, best_angles):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Recompute exact side
    best_side = outer_side(best_centers, best_angles)
    
    inner = []
    for i in range(N):
        x, y = float(best_centers[i, 0]), float(best_centers[i, 1])
        ang = float(best_angles[i])
        inner.append([x, y, ang])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
