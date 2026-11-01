import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    N = 12

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        # Angles: 0, 60, 120, 180, 240, 300 degrees
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygon_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for v in [v1, v2]:
            for i in range(len(v)):
                j = (i + 1) % len(v)
                edge = v[j] - v[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal /= norm
                    axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
            
        return False

    def hexagons_disjoint_fast(positions, angles):
        """Check if all hexagons are pairwise disjoint. Returns True if disjoint."""
        n = len(positions)
        verts_list = []
        for i in range(n):
            verts_list.append(hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
        
        for i in range(n):
            for j in range(i + 1, n):
                if not polygon_disjoint_sat(verts_list[i], verts_list[j]):
                    return False
        return True

    def calc_min_outer_side_fast(all_verts, theta):
        """Calculate minimum outer hexagon side length for a given orientation theta."""
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def find_best_outer_orientation_fast(all_verts):
        """Find orientation and side length for minimum enclosing hexagon."""
        # Coarse search
        thetas_coarse = np.linspace(0, np.pi/3, 180)
        sides = np.array([calc_min_outer_side_fast(all_verts, th) for th in thetas_coarse])
        best_idx = np.argmin(sides)
        
        # Fine search around best
        step = (np.pi/3) / 179
        lo = thetas_coarse[max(0, best_idx - 2)]
        hi = thetas_coarse[min(len(thetas_coarse)-1, best_idx + 2)]
        
        # Use minimize_scalar on a narrow window
        res = minimize_scalar(lambda th: calc_min_outer_side_fast(all_verts, th), 
                              bounds=(lo, hi), method='bounded', 
                              options={'xatol': 1e-6})
        return res.fun, res.x

    def objective(params):
        """Objective: outer hexagon side length, with penalty for overlap."""
        if time.time() > deadline:
            return 1e10
            
        n = len(params) // 3
        positions = np.array([[params[i*3], params[i*3+1]] for i in range(n)])
        angles = np.array([params[i*3+2] for i in range(n)])
        
        # Check disjointness
        if not hexagons_disjoint_fast(positions, angles):
            # Return a large penalty, but based on how bad the overlap is?
            # For simplicity, just return a large number to steer away.
            return 1e10
        
        all_verts = []
        for i in range(n):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        side, _ = find_best_outer_orientation_fast(all_verts)
        return side

    # Generate initial configuration: Honeycomb
    def make_honeycomb_config(d=2.1):
        pos = []
        ang = []
        # Center
        pos.append([0.0, 0.0])
        ang.append(0.0)
        # Ring 1: 6 hexagons
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d * np.cos(a), d * np.sin(a)])
            ang.append(0.0)
        # Ring 2: 5 more hexagons (to make 12 total, skipping one to save space or just take 5)
        # Let's take the first 5 of the next ring
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([2 * d * np.cos(a), 2 * d * np.sin(a)])
            ang.append(0.0)
        return np.array(pos), np.array(ang)

    # Ensure initial config is valid
    pos0, ang0 = make_honeycomb_config(2.1)
    if not hexagons_disjoint_fast(pos0, ang0):
        # Scale up until valid
        scale = 1.0
        while not hexagons_disjoint_fast(pos0 * scale, ang0) and scale < 5.0:
            scale += 0.1
        pos0 = pos0 * scale

    params0 = np.concatenate([pos0.flatten(), ang0])
    
    # Bounds for optimization
    bounds = []
    for i in range(12):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, np.pi))

    best_obj = objective(params0)
    best_params = params0.copy()

    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                polish=False
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: Local Refinement with L-BFGS-B
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Final Verification and Extraction
    best_positions = np.array([[best_params[i*3], best_params[i*3+1]] for i in range(12)])
    best_angles = np.array([best_params[i*3+2] for i in range(12)])
    
    # If the optimized result is invalid (overlap), fall back to the initial valid config
    if not hexagons_disjoint_fast(best_positions, best_angles):
        best_positions = pos0
        best_angles = ang0

    all_verts = []
    for i in range(12):
        v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    outer_side, outer_angle_rad = find_best_outer_orientation_fast(all_verts)
    
    # Calculate outer center
    angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = all_verts @ normals.T
    mins = dots.min(axis=0)
    maxs = dots.max(axis=0)
    mids = (mins + maxs) / 2.0
    c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
    outer_center = tuple(c)
    outer_angle_deg = np.degrees(outer_angle_rad)

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(12)]
    
    return inner, outer_center, outer_side, outer_angle_deg
    # EVOLVE-BLOCK-END
