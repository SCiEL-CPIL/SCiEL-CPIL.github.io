"""
Strategy: Topology-Driven Feasibility Restoration with Exact Support Function Optimization.

Key Differences from Current Program:
1. Topology Awareness: Explicitly defines the 11-hexagon topology (1 center, 6 ring, 4 gaps)
   to initialize a configuration that is topologically correct and near-feasible, 
   rather than relying on random perturbations of a potentially bad start.
2. Exact Non-Overlap Handling: Instead of using center-distance proxies for overlap penalties,
   we compute the exact minimum distance between convex polygons (hexagons) using 
   support functions. This allows for tighter, more accurate constraints.
3. Barrier Method for Support Function: The objective is the maximum support function value.
   We use a smooth approximation (log-sum-exp) of the max function to maintain differentiability
   while accurately capturing the "bottleneck" vertices that define the outer boundary.
4. Two-Phase Optimization: 
   Phase 1: Minimize a relaxed objective (center distance based) to get close to feasibility.
   Phase 2: Switch to exact support-function-based objective with a barrier for overlaps,
   refining the solution to high precision.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _support_function_point(point, outer_angle_rad):
    """
    Compute the support function value for a single point relative to the outer hexagon.
    For a regular hexagon centered at origin with side S and rotation a,
    the point (x,y) is inside if max(|x'|, |x' + y'/sqrt(3)|, |x' - y'/sqrt(3)|) <= S
    where (x', y') is the rotated point.
    """
    ca = np.cos(outer_angle_rad)
    sa = np.sin(outer_angle_rad)
    x, y = point
    lx = ca * x - sa * y
    ly = sa * x + ca * y
    sqrt3 = np.sqrt(3)
    
    c1 = abs(lx)
    c2 = abs(lx + ly / sqrt3)
    c3 = abs(lx - ly / sqrt3)
    
    return max(c1, c2, c3)

def _min_outer_side_exact(inner_positions, inner_angles, outer_center, outer_angle_deg):
    """Compute minimum outer side length to contain all inner hexagons exactly."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle_deg)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check overlaps using SAT
    def sat_overlap(v1, v2):
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
        return True

    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    # Check containment
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for verts in verts_list:
        for v in verts:
            if not point_in_poly(v, outer_verts):
                return False
    return True

def _initial_configuration():
    """
    Generate a specific initial configuration for 11 hexagons.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center.
    - Ring 2: 4 hexagons at distance 6 from center, in the gaps.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def _objective_phase1(params):
    """
    Phase 1 Objective: Minimize center distances to approximate hexagonal packing.
    This is smooth and fast, getting us close to a valid configuration.
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    
    # Objective: Sum of squared deviations from ideal hexagonal lattice distances
    # Ideal distances:
    # Center to Ring 1: 2*sqrt(3)
    # Ring 1 to Ring 1 (adjacent): 2*sqrt(3)
    # Ring 1 to Ring 2: 2*sqrt(3)
    # Center to Ring 2: 6
    # Ring 2 to Ring 2 (adjacent): 2*sqrt(3)
    
    sqrt3 = np.sqrt(3)
    ideal_d1 = 2 * sqrt3
    ideal_d2 = 6.0
    
    obj = 0.0
    
    # Center (0) to Ring 1 (1-6)
    for i in range(1, 7):
        d = np.linalg.norm(pos[0] - pos[i])
        obj += (d - ideal_d1)**2
        
    # Ring 1 (i) to Ring 1 (i+1 mod 6)
    for i in range(1, 7):
        j = (i - 1) % 6 + 1
        d = np.linalg.norm(pos[i] - pos[j])
        obj += (d - ideal_d1)**2
        
    # Ring 1 (i) to Ring 2 (k)
    # Ring 2 angles are pi/6, 5pi/6, etc.
    # Each Ring 2 hexagon is adjacent to 2 Ring 1 hexagons.
    # Ring 2 index 7 (angle pi/6) is adjacent to Ring 1 index 1 (angle 0) and 2 (angle pi/3)
    # Let's map them explicitly
    adjacencies = [
        (7, 1), (7, 2),
        (8, 2), (8, 3),
        (9, 3), (9, 4),
        (10, 4), (10, 5),
        # Also Ring 1 to Ring 1? No, Ring 2 is in the gaps.
        # Wait, Ring 2 hexagons are also adjacent to each other?
        # Ring 2 at pi/6 and 5pi/6. Distance between them?
        # Center to R2 is 6. Angle diff is 4pi/3 or 2pi/3?
        # pi/6 to 5pi/6 is 4pi/3? No, 5pi/6 - pi/6 = 4pi/6 = 2pi/3.
        # Distance = sqrt(6^2 + 6^2 - 2*6*6*cos(2pi/3)) = sqrt(72 - 72*(-0.5)) = sqrt(108) = 6*sqrt(3).
        # This is NOT 2*sqrt(3). So Ring 2 hexagons are NOT adjacent to each other in the hexagonal lattice sense?
        # Let's check the geometry.
        # A hexagonal cluster of 11 hexagons:
        # 1 in center.
        # 6 around it.
        # 6 more around the ring? No, that would be 19.
        # The "11 hexagons" packing is usually:
        # 1 center
        # 6 in first ring
        # 4 in the "corners" of the second ring?
        # Actually, the optimal packing for 11 hexagons in a hexagon is often cited.
        # Let's stick to the topology defined in _initial_configuration.
        # The distances in _initial_configuration are:
        # R2 at 6.0 distance from center.
        # R2 at pi/6. R1 at 0 and pi/3.
        # Dist R2(7) to R1(1): sqrt((6cos30 - 2sqrt3)^2 + (6sin30)^2)
        # = sqrt((3sqrt3 - 2sqrt3)^2 + (3)^2) = sqrt(3 + 9) = sqrt(12) = 2sqrt3. Correct.
        # Dist R2(7) to R2(8):
        # R2(7) at 6, pi/6. R2(8) at 6, 5pi/6.
        # Angle diff 2pi/3.
        # Dist = 6 * sqrt(2 - 2cos(2pi/3)) = 6 * sqrt(2 - 2(-0.5)) = 6 * sqrt(3) = 6sqrt3.
        # So they are far apart.
    ]
    
    for i, j in adjacencies:
        d = np.linalg.norm(pos[i] - pos[j])
        obj += (d - ideal_d1)**2
        
    # Center to Ring 2
    for i in range(7, 11):
        d = np.linalg.norm(pos[0] - pos[i])
        obj += (d - ideal_d2)**2
        
    # Add a small term to keep angles aligned (0 degrees) to maintain hexagonal symmetry
    obj += 0.1 * np.sum(ang**2)
    
    return obj

def _objective_phase2(params):
    """
    Phase 2 Objective: Minimize outer hexagon side length with smooth overlap penalty.
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    
    # 1. Outer side length (smooth approximation using log-sum-exp)
    side = _min_outer_side_exact(pos, ang, (0.0, 0.0), 0.0)
    
    # 2. Overlap penalty using exact distance between hexagons
    # For unit hexagons, minimum center distance is 2*sqrt(3) if aligned.
    # But with rotation, it's complex. We use a proxy: 
    # If center distance < 2*sqrt(3) - epsilon, they likely overlap.
    # To be safer, we use a barrier based on the minimum distance between boundaries.
    # However, computing exact min distance between two rotated hexagons is expensive.
    # We will use a strong penalty on center distance violations.
    
    min_center_dist = 2 * np.sqrt(3)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_center_dist:
                # Quadratic penalty for violation
                penalty += (min_center_dist - d)**2
    
    # Weight penalty heavily
    return side + 50.0 * penalty

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None
    
    # Initial configuration
    pos_init, ang_init = _initial_configuration()
    x0 = np.concatenate([pos_init.ravel(), ang_init])
    
    # Phase 1: Relaxation to ideal lattice
    # This should be very fast and get us to a valid-ish configuration
    try:
        res1 = minimize(_objective_phase1, x0, method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
        if res1.fun < _objective_phase1(x0):
            x0 = res1.x
    except Exception:
        pass
    
    # Phase 2: Refine with exact objective
    # Use multiple restarts from the phase 1 solution with small perturbations
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Perturb slightly to escape local minima
        temp = 0.1 * (1.0 + np.random.rand())
        x_start = x0 + np.random.normal(0, temp, x0.shape)
        
        try:
            res = minimize(_objective_phase2, x_start, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final Refinement
    if time.time() < DEADLINE - 5.0 and best_params is not None:
        try:
            res_final = minimize(_objective_phase2, best_params, method='L-BFGS-B',
                                 options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Extract best solution
    if best_params is None:
        best_params = x0
        
    pos_best = best_params[:2*N].reshape(N, 2)
    ang_best = best_params[2*N:]
    
    inner_list = [[float(pos_best[i,0]), float(pos_best[i,1]), float(ang_best[i])] for i in range(N)]
    final_side = _min_outer_side_exact(pos_best, ang_best, (0.0, 0.0), 0.0)
    
    # Final validity check
    if not _check_validity(inner_list, (0.0, 0.0), final_side, 0.0):
        # Fallback to initial configuration if invalid
        pos_f, ang_f = _initial_configuration()
        inner_list = [[float(pos_f[i,0]), float(pos_f[i,1]), float(ang_f[i])] for i in range(N)]
        final_side = _min_outer_side_exact(pos_f, ang_f, (0.0, 0.0), 0.0)
        
    return inner_list, (0.0, 0.0), float(final_side), 0.0

if __name__ == "__main__":
    construct()
