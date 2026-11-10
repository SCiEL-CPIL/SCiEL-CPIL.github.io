import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    def get_hex_verts(cx, cy, d, angle_deg):
        a = np.radians(angle_deg)
        angles = a + np.radians(60) * np.arange(6)
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])
    
    def min_separation(v1, v2):
        """Minimum distance between two convex hexagons. Negative if overlapping."""
        # SAT: check all 12 edge normals
        min_dist = np.inf
        for poly in [v1, v2]:
            for i in range(6):
                p1 = poly[i]
                p2 = poly[(i+1) % 6]
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-12:
                    continue
                normal = np.array([-edge[1], edge[0]]) / edge_len
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(min1 - max2, min2 - max1)
                if gap > 0:
                    return -gap  # Separated by gap
                # Track the minimum overlap
                # Actually for SAT, if any axis separates, they don't overlap
                # For distance, we need the minimum gap over all axes
                min_dist = min(min_dist, -gap)
        # If no separating axis, they overlap
        # min_dist should be negative
        return min_dist if min_dist < 0 else 0
    
    def check_valid(inner_data, outer_data, s):
        """Check if the packing is valid. Returns (valid, violation_amount)."""
        outer_verts = get_hex_verts(outer_data[0], outer_data[1], s, outer_data[2])
        
        inner_verts_list = []
        for i in range(12):
            vx, vy, va = inner_data[i]
            inner_verts_list.append(get_hex_verts(vx, vy, 1.0, va))
        
        # Check non-overlap
        max_violation = 0
        for i in range(12):
            for j in range(i+1, 12):
                sep = min_separation(inner_verts_list[i], inner_verts_list[j])
                if sep < 1e-6:
                    max_violation = max(max_violation, 1e-6 - sep)
        
        # Check containment
        for iv in inner_verts_list:
            for v in iv:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i+1) % 6]
                    cross = np.cross(p2 - p1, v - p1)
                    if cross < -1e-6:
                        max_violation = max(max_violation, -1e-6 - cross)
        
        return max_violation
    
    def objective(x):
        s = x[0]
        outer_angle = x[1]
        inner_data = x[2:].reshape(12, 3)
        
        violation = check_valid(inner_data, np.array([0, 0, outer_angle]), s)
        if violation > 0:
            return s + 100 * violation
        return s
    
    # Initial guess: compact arrangement
    # 1 center + 6 ring + 5 outer
    sqrt3 = np.sqrt(3)
    centers = [(0, 0)]
    # Ring of 6
    for k in range(6):
        angle = np.radians(60 * k + 30)
        centers.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    # 5 more in the second ring
    second_ring_angles = [0, 60, 120, 180, 240]
    for k in range(5):
        angle = np.radians(second_ring_angles[k])
        centers.append((2*sqrt3 * np.cos(angle), 2*sqrt3 * np.sin(angle)))
    
    init = [5.0, 0.0]  # s, outer_angle
    for cx, cy in centers:
        init.extend([cx, cy, 0.0])
    
    best_s = np.inf
    best_x = None
    
    rng = np.random.default_rng(123)
    n_trials = 30
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x0 = np.array(init)
        elif trial == 1:
            x0 = np.array(init)
            x0[0] = 4.0
        else:
            x0 = np.array(init) + rng.normal(0, 0.5, size=len(init))
            x0[0] = 4.0 + rng.uniform(-1, 1)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if result.fun < best_s and result.fun < 10:
                best_s = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if best_x is None:
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
        return inner_hex_data, outer_hex_data, 8.0
    
    s = best_x[0]
    outer_angle = best_x[1]
    inner_data = best_x[2:].reshape(12, 3)
    
    inner_hex_data = inner_data
    outer_hex_data = np.array([0, 0, outer_angle])
    
    return inner_hex_data, outer_hex_data, s
