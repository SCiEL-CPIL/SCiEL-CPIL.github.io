import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_hex_verts(cx, cy, angle_deg):
        """Get vertices of unit hexagon at center (cx,cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        # Unit hexagon vertices at angles 0,60,120,180,240,300
        base = np.array([[math.cos(math.radians(k*60)), math.sin(math.radians(k*60))] for k in range(6)])
        rot = np.array([[ca, -sa], [sa, ca]])
        return base @ rot.T + np.array([cx, cy])

    def poly_dist(p1, p2, eps=1e-10):
        """Minimum distance between two convex hexagons using support functions.
        Returns negative if overlapping (penetration depth), positive if separated."""
        min_dist = float('inf')
        # Check all 12 edge normals (6 from each polygon)
        for poly, other in [(p1, p2), (p2, p1)]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i+1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                # Project both polygons onto this normal
                proj1 = p1 @ normal
                proj2 = other @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                # Gap: positive means separated
                gap = max(min1 - max2, min2 - max1)
                if gap < 0:
                    min_dist = min(min_dist, gap)
                else:
                    # Found a separating axis - polygons are disjoint
                    return gap
        return min_dist

    def calc_outer_side(centers, angles, outer_angle=0.0):
        """Calculate minimum side length of regular hexagon containing all inner hexagons."""
        theta = math.radians(outer_angle)
        max_proj = 0.0
        for k in range(6):
            na = theta + math.pi/6 + k * math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            for i in range(N):
                cx, cy = centers[i]
                a = math.radians(angles[i])
                ca, sa = math.cos(a), math.sin(a)
                # Max projection of hex i onto direction (nx, ny)
                # = cx*nx + cy*ny + max over vertices of (vx*nx + vy*ny)
                # For unit hex rotated by a, max proj onto unit dir d is:
                # max_k (cos(a + k*pi/3)*d_x + sin(a + k*pi/3)*d_y)
                best = 0.0
                for k in range(6):
                    va = a + k * math.pi/3
                    proj = math.cos(va)*nx + math.sin(va)*ny
                    if proj > best:
                        best = proj
                proj_val = cx*nx + cy*ny + best
                if proj_val > max_proj:
                    max_proj = proj_val
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(centers, angles, tol=1e-6):
        """Check if all hexagons are pairwise disjoint."""
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if poly_dist(polys[i], polys[j]) < -tol:
                    return False
        return True

    # Phase 1: Generate good initial configurations from honeycomb lattice
    sqrt3 = math.sqrt(3)
    best_config = None
    best_score = float('inf')

    # Standard honeycomb: centers at triangular lattice points
    # For unit hexagons, touching distance is 2*sqrt(3)/2 * 2? 
    # Inradius = sqrt(3)/2, so center-to-center for touching = 2*sqrt(3)
    # Actually for aligned hexagons, the distance between parallel sides is sqrt(3)
    # So centers need to be sqrt(3) apart in the normal direction
    # Triangular lattice: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    # This gives center distance = sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25+0.75) = sqrt(3)
    # Wait that's the distance between adjacent centers in the lattice
    
    # Let me reconsider. For flat-top hexagons with side 1:
    # Width (flat to flat) = sqrt(3), height (point to point) = 2
    # For hexagons to touch side-to-side, centers must be sqrt(3) apart perpendicular to sides
    # Triangular lattice vectors: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    # |a1| = sqrt(2.25 + 0.75) = sqrt(3) ≈ 1.732
    # This should give touching hexagons

    a1 = np.array([1.5, sqrt3/2])
    a2 = np.array([0.0, sqrt3])

    # Generate a large grid of lattice points
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            lattice_points.append(p)
    
    # Sort by distance from origin, take closest 12
    lattice_points.sort(key=lambda p: np.dot(p, p))
    init_points = np.array(lattice_points[:12])
    
    # Center the cluster
    centroid = np.mean(init_points, axis=0)
    init_points -= centroid

    # Try different global rotations and angle assignments
    for rot_deg in [0, 30, 60, 90, 120, 150]:
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                           [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated = init_points @ rot_mat.T
        angles = np.full(N, rot_deg)
        
        if is_valid(rotated, angles, tol=1e-5):
            score = calc_outer_side(rotated, angles, 0.0)
            if score < best_score:
                best_score = score
                best_config = (rotated.copy(), angles.copy())

    # If no valid config found, use a looser packing
    if best_config is None:
        # Scale up the lattice slightly to ensure validity
        for scale in [1.01, 1.02, 1.05, 1.1]:
            scaled = init_points * scale
            for rot_deg in [0, 30]:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                                   [math.sin(rot_rad), math.cos(rot_rad)]])
                rotated = scaled @ rot_mat.T
                angles = np.full(N, rot_deg)
                if is_valid(rotated, angles, tol=1e-5):
                    score = calc_outer_side(rotated, angles, 0.0)
                    if score < best_score:
                        best_score = score
                        best_config = (rotated.copy(), angles.copy())
                    break
            if best_config is not None:
                break

    if best_config is None:
        # Fallback: very loose packing
        init_points = np.array([[i*4, j*4] for i in range(-2,3) for j in range(-2,3)][:12])
        init_points -= np.mean(init_points, axis=0)
        best_config = (init_points, np.zeros(N))
        best_score = calc_outer_side(init_points, np.zeros(N), 0.0)

    init_centers, init_angles = best_config
    x0 = np.concatenate([init_centers.flatten(), init_angles, [0.0]])

    # Phase 2: SLSQP optimization with hard constraints
    def objective(params):
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle = params[3*N]
        return calc_outer_side(centers, angles, outer_angle)

    def make_constraint(i, j):
        def constraint(params):
            ci = params[2*i:2*i+2]
            cj = params[2*j:2*j+2]
            ai = params[2*N + i]
            aj = params[2*N + j]
            pi = get_hex_verts(ci[0], ci[1], ai)
            pj = get_hex_verts(cj[0], cj[1], aj)
            return poly_dist(pi, pj)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    bounds = [(-10, 10), (-10, 10)] * N + [(0, 360)] * N + [(0, 60)]
    
    best_result = x0.copy()
    best_val = best_score

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
        if res.fun < best_val:
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:3*N]
            oang = res.x[3*N]
            if is_valid(pos, ang, tol=1e-5):
                best_val = res.fun
                best_result = res.x.copy()
    except Exception:
        pass

    # Phase 3: Nelder-Mead polish with smooth penalty
    if time.time() < deadline - 15:
        def smooth_penalty(params):
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = poly_dist(polys[i], polys[j])
                    if d < 0:
                        pen += 100.0 * d * d
            return pen

        def obj_pen(params):
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            oang = params[3*N]
            return calc_outer_side(centers, angles, oang) + smooth_penalty(params)

        try:
            res2 = minimize(obj_pen, best_result, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:3*N]
            oang = res2.x[3*N]
            if is_valid(pos, ang, tol=1e-5):
                val = calc_outer_side(pos, ang, oang)
                if val < best_val:
                    best_val = val
                    best_result = res2.x.copy()
        except Exception:
            pass

    # Final: extract result
    final_centers = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:3*N]
    final_outer_angle = best_result[3*N]
    final_side = calc_outer_side(final_centers, final_angles, final_outer_angle)

    inner = [[final_centers[i][0], final_centers[i][1], final_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
