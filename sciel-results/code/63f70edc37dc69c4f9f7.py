import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute base unit hexagon vertices (pointy-top, vertex at 0 deg)
    base_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = base_verts @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def point_in_hexagon(pt, verts):
        """
        Check if point pt is inside or on the boundary of convex hexagon verts.
        Returns True if inside/on boundary.
        Uses cross product sign consistency for CCW polygon.
        """
        # verts are CCW
        # Cross product of edge (v[i+1]-v[i]) and vector (pt - v[i])
        # If all cross products have the same sign (or zero), point is inside.
        cross_products = []
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            edge = v2 - v1
            vec = pt - v1
            cross = edge[0] * vec[1] - edge[1] * vec[0]
            cross_products.append(cross)
        
        # Check if all are non-negative or all non-positive (allowing small tolerance)
        tol = 1e-9
        all_pos = all(c >= -tol for c in cross_products)
        all_neg = all(c <= tol for c in cross_products)
        return all_pos or all_neg

    def check_disjoint(pos, ang, other_pos, other_ang, eps=1e-5):
        """
        Returns True if hexagons are disjoint (with margin eps).
        If they overlap, returns False.
        """
        v1 = get_hex_verts(pos[0], pos[1], 1.0, ang)
        v2 = get_hex_verts(other_pos[0], other_pos[1], 1.0, other_ang)
        
        # Check if any vertex of v1 is inside v2
        for pt in v1:
            if point_in_hexagon(pt, v2):
                return False
        # Check if any vertex of v2 is inside v1
        for pt in v2:
            if point_in_hexagon(pt, v1):
                return False
        return True

    def calc_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at origin
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals of the outer hexagon edges
        # For a regular hexagon, the outward normals are at angles theta + 30, theta + 90, ...
        # Actually, for a regular hexagon with vertex at angle 0, the normal to the edge between vertex 0 and 1 is at 30 deg.
        # Let's define the 6 support directions.
        normals = np.array([
            [math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] 
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # The distance from center to edge is max_proj.
        # For a regular hexagon, apothem = side * sqrt(3) / 2.
        # So side = 2 * apothem / sqrt(3) = 2 * max_proj / sqrt(3).
        return max_proj * 2.0 / math.sqrt(3)

    def objective(params):
        """
        Objective function: Outer side length + Penalty for overlaps.
        params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, outer_angle]
        """
        positions = params[:3*N].reshape(N, 3)[:, :2]
        angles = params[:3*N].reshape(N, 3)[:, 2]
        outer_angle = params[3*N]
        
        base_cost = calc_outer_side(positions, angles, outer_angle)
        
        penalty = 0.0
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(positions[i], angles[i], positions[j], angles[j]):
                    # Simple penalty: count overlap
                    penalty += 10.0
        
        return base_cost + penalty

    def objective_grad(params):
        """
        Numerical gradient for SLSQP.
        """
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    # --- Initial Configuration Generation ---
    
    sqrt3 = math.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    # Generate a compact hexagonal cluster
    # Center + 6 neighbors + 5 from ring 2
    init_positions = []
    init_positions.append(np.array([0.0, 0.0]))
    for k in range(6):
        ang = k * math.pi / 3
        pos = sqrt3 * np.array([math.cos(ang), math.sin(ang)])
        init_positions.append(pos)
    
    # Ring 2 points
    ring2_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d = max(abs(i), abs(j), abs(i+j))
            if d == 2:
                pos = i * v1 + j * v2
                ring2_pts.append(pos)
    
    # Sort by angle to pick a contiguous arc
    ring2_sorted = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
    for k in range(5):
        init_positions.append(ring2_sorted[k])
    
    init_positions = np.array(init_positions)
    # Center it
    center = np.mean(init_positions, axis=0)
    init_positions = init_positions - center
    
    # Initial parameters: x, y, angle for each, then outer angle
    x0 = np.zeros(3*N + 1)
    for i in range(N):
        x0[3*i] = init_positions[i, 0]
        x0[3*i+1] = init_positions[i, 1]
        x0[3*i+2] = 0.0 # Initial angle 0
    x0[3*N] = 0.0 # Initial outer angle 0

    # --- Optimization ---
    
    best_x = x0.copy()
    best_score = objective(best_x)
    
    # Bounds: positions in [-6, 6], angles in [0, 60] (symmetry), outer angle in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-6, 6))
        bounds.append((-6, 6))
        bounds.append((0, 60))
    bounds.append((0, 60))
    
    # Try multiple runs with slight perturbations if time permits
    num_runs = 3
    
    for run in range(num_runs):
        if time.time() > deadline - 20:
            break
            
        # Perturb initial solution for subsequent runs
        if run > 0:
            x0_perturbed = best_x.copy()
            noise = np.random.normal(0, 0.1, len(x0_perturbed))
            # Don't perturb outer angle too much
            noise[-1] = np.random.normal(0, 0.05)
            x0_perturbed += noise
            # Ensure angles are within bounds roughly
            for i in range(N):
                x0_perturbed[3*i+2] = np.clip(x0_perturbed[3*i+2], 0, 60)
            x0_perturbed[-1] = np.clip(x0_perturbed[-1], 0, 60)
        else:
            x0_perturbed = x0.copy()
            
        try:
            res = minimize(
                objective,
                x0_perturbed,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.fun < best_score:
                # Verify validity before accepting
                pos = res.x[:3*N].reshape(N, 3)[:, :2]
                ang = res.x[:3*N].reshape(N, 3)[:, 2]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(pos[i], ang[i], pos[j], ang[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_score = res.fun
                    best_x = res.x.copy()
                    
        except Exception as e:
            pass

    # --- Final Result Extraction ---
    
    final_pos = best_x[:3*N].reshape(N, 3)[:, :2]
    final_ang = best_x[:3*N].reshape(N, 3)[:, 2]
    final_outer_angle = best_x[3*N]
    
    # Recalculate exact side length
    final_side = calc_outer_side(final_pos, final_ang, final_outer_angle)
    
    # Construct output
    inner = []
    for i in range(N):
        inner.append([float(final_pos[i, 0]), float(final_pos[i, 1]), float(final_ang[i])])
    
    # The outer hexagon is centered at (0,0)
    # Return side length and rotation
    return inner, (0.0, 0.0), float(final_side), float(final_outer_angle)

# Fallback if optimization fails completely (should not happen with the above)
if __name__ == "__main__":
    # Basic test
    res = construct()
    print(res)
