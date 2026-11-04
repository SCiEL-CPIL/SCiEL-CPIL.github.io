import math
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import fsolve, brentq
import itertools

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices (flat-top, side 1)
    # Angles: 0, 60, 120, 180, 240, 300
    unit_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] 
        for a in [0, 60, 120, 180, 240, 300]
    ])
    
    # Precompute edge normals for unit hexagon (outward)
    # Edge 0: v0 to v1. Normal is perpendicular.
    # For flat-top, edge 0 is (1,0) to (0.5, sqrt(3)/2). 
    # Vector: (-0.5, sqrt(3)/2). Normal: (sqrt(3)/2, 0.5) -> angle 30 deg.
    # Let's just compute them dynamically to be safe, or hardcode.
    # Normals for flat-top unit hex (outward):
    # 30, 90, 150, 210, 270, 330 degrees
    hex_normals = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] 
        for a in [30, 90, 150, 210, 270, 330]
    ])

    def get_hex_verts(cx, cy, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        return unit_verts @ rot.T + np.array([cx, cy])

    def get_hex_normals(angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        return hex_normals @ rot.T

    def min_dist_hex(i, j, centers, angles):
        """
        Compute minimum distance between hex i and hex j.
        Returns negative if overlapping (penetration depth).
        Uses Separating Axis Theorem (SAT) on 12 axes.
        """
        v1 = get_hex_verts(centers[i,0], centers[i,1], angles[i])
        v2 = get_hex_verts(centers[j,0], centers[j,1], angles[j])
        
        # Normals of both polygons
        n1 = get_hex_normals(angles[i])
        n2 = get_hex_normals(angles[j])
        
        min_dist = float('inf')
        
        for normal in itertools.chain(n1, n2):
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            gap = max(min1 - max2, min2 - max1)
            
            if gap < 0:
                # Overlapping on this axis, update min penetration
                min_dist = min(min_dist, gap)
            else:
                # Separating axis found
                return gap
        
        return min_dist

    def check_validity(centers, angles, tol=1e-7):
        for i in range(N):
            for j in range(i+1, N):
                if min_dist_hex(i, j, centers, angles) < -tol:
                    return False
        return True

    def calc_container_size(centers, angles, container_angle=0.0):
        """
        Calculate the side length of the smallest regular hexagon (rotated by container_angle)
        that contains all inner hexagons.
        """
        theta = math.radians(container_angle)
        max_proj = 0.0
        
        # Container normals are at theta + 30 + k*60
        for k in range(6):
            na = theta + math.pi/6 + k * math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            
            for i in range(N):
                # Max projection of hex i onto (nx, ny)
                # This is the support function of the hexagon in direction (nx, ny)
                # Plus the center projection
                h = support_function(angles[i], nx, ny)
                proj_val = centers[i,0]*nx + centers[i,1]*ny + h
                if proj_val > max_proj:
                    max_proj = proj_val
        
        # For a regular hexagon with side s, the distance from center to side (inradius) is s * sqrt(3) / 2
        # max_proj is the distance from center to the furthest point in the normal direction.
        # So s * sqrt(3) / 2 = max_proj
        return 2.0 * max_proj / math.sqrt(3)

    def support_function(angle_deg, nx, ny):
        """
        Max projection of unit hexagon (rotated by angle_deg) onto unit vector (nx, ny).
        """
        # The support function of a regular polygon is the max dot product of its vertices with the direction.
        # For a hexagon, the support function is piecewise linear.
        # Vertices are at angles angle_deg + k*60.
        # proj_k = cos(angle_deg + k*60 - dir_angle)
        # We just need to find the max over 6 vertices.
        # Actually, for a regular polygon, the support function is s * cos(pi/6) * ... no, it's simpler.
        # Let's just compute dot products with the 6 vertices.
        # v_k = (cos(a+k*pi/3), sin(a+k*pi/3))
        # dot = cos(a+k*pi/3)*nx + sin(a+k*pi/3)*ny
        #      = cos(a+k*pi/3 - phi) where phi = atan2(ny, nx)
        # Max of cos(x) is 1, but x is discrete.
        
        a = math.radians(angle_deg)
        phi = math.atan2(ny, nx)
        max_proj = 0.0
        for k in range(6):
            val = math.cos(a + k*math.pi/3 - phi)
            if val > max_proj:
                max_proj = val
        return max_proj

    # Strategy:
    # 1. Generate initial honeycomb-like configurations.
    # 2. For each configuration, define a "topology" (contact graph).
    # 3. Use fsolve to find exact positions that satisfy contact constraints and container wall constraints.
    # 4. Minimize container size by adjusting the "tightness" of the packing.
    
    best_score = float('inf')
    best_result = None

    # Initial configurations: Honeycomb clusters
    # A honeycomb cluster of 12 hexagons can be:
    # - 1 center + 6 ring + 5 of second ring
    # - 2 centers + ...
    
    sqrt3 = math.sqrt(3)
    a1 = np.array([1.5, sqrt3/2])
    a2 = np.array([0.0, sqrt3])
    
    # Generate points
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            lattice_points.append(p)
            
    lattice_points.sort(key=lambda p: np.dot(p, p))
    init_points = np.array(lattice_points[:12])
    centroid = np.mean(init_points, axis=0)
    init_points -= centroid

    # Try different global rotations
    for rot_deg in [0, 30, 60]:
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                           [math.sin(rot_rad), math.cos(rot_rad)]])
        centers = init_points @ rot_mat.T
        angles = np.full(N, rot_deg)
        
        # Check validity
        if not check_validity(centers, angles):
            continue
            
        # Define a "stress" or "pressure" parameter.
        # We want to find the minimal container size.
        # Instead of optimizing positions directly, we can use the fact that
        # in a minimal packing, the graph of contacts is rigid (or close to it).
        
        # Let's try a direct optimization on the container size and positions
        # using a penalty method for collisions, but structured to find the "tight" packing.
        
        # Actually, the prompt suggests a "homotopy continuation" or "Minkowski difference" approach.
        # A simpler robust approach for this specific N=12 case:
        # Use the initial honeycomb, and use `minimize` with SLSQP to minimize the container size,
        # with constraints that ensure no overlap.
        # The current program does this. Why does it score 10.0?
        # Maybe it gets stuck in a local minimum where hexagons are not optimally rotated.
        
        # Let's try a different initial guess: "Loose" packing, then squeeze.
        # Or better: Use the "eviction" idea? No, that's for the island.
        
        # Let's try to enforce specific contact topologies.
        # For N=12, a common optimal packing is a "flower" with 1 center and 11 around?
        # Or two rows of 6?
        
        # Let's try a row packing: 2 rows of 6.
        # Hexagons in a row touch side-to-side. Distance = sqrt(3).
        # Row 1: y=0, x = -2.5*sqrt(3), -1.5*sqrt(3), ...
        # Row 2: y=sqrt(3), x = -2.0*sqrt(3), -1.0*sqrt(3), ...
        
        row_centers = []
        for k in range(6):
            row_centers.append([-2.5*sqrt3 + k*sqrt3, 0])
        for k in range(6):
            row_centers.append([-2.0*sqrt3 + k*sqrt3, sqrt3])
            
        row_centers = np.array(row_centers)
        row_centers -= np.mean(row_centers, axis=0)
        
        if not check_validity(row_centers, np.zeros(N)):
            continue
            
        # We have two candidate initial configs: Honeycomb and Row.
        # Let's run a local optimization from both.
        
        for init_c in [centers, row_centers]:
            for init_a in [angles, np.zeros(N)]:
                if time.time() > deadline - 10:
                    break
                
                x0 = np.concatenate([init_c.flatten(), init_a, [0.0]])
                
                def objective(params):
                    c = params[:2*N].reshape(N, 2)
                    a = params[2*N:3*N]
                    oa = params[3*N]
                    return calc_container_size(c, a, oa)
                
                def make_constraint(i, j):
                    def constraint(params):
                        ci = params[2*i:2*i+2]
                        cj = params[2*j:2*j+2]
                        ai = params[2*N + i]
                        aj = params[2*N + j]
                        return min_dist_hex(i, j, 
                            np.array([[ci[0], ci[1]],[cj[0], cj[1]]]), 
                            np.array([ai, aj]))
                    return constraint
                
                constraints = []
                for i in range(N):
                    for j in range(i+1, N):
                        constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
                
                bounds = [(-10, 10), (-10, 10)] * N + [(0, 360)] * N + [(0, 60)]
                
                try:
                    from scipy.optimize import minimize
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                                  constraints=constraints,
                                  options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
                    
                    if res.success or res.fun < best_score:
                        pos = res.x[:2*N].reshape(N, 2)
                        ang = res.x[2*N:3*N]
                        oang = res.x[3*N]
                        
                        if check_validity(pos, ang, tol=1e-5):
                            val = calc_container_size(pos, ang, oang)
                            if val < best_score:
                                best_score = val
                                best_result = res.x.copy()
                except Exception:
                    pass

    # If no result found, return a valid but loose packing
    if best_result is None:
        centers = init_points
        angles = np.zeros(N)
        best_score = calc_container_size(centers, angles, 0.0)
        best_result = np.concatenate([centers.flatten(), angles, [0.0]])

    final_centers = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:3*N]
    final_outer_angle = best_result[3*N]
    final_side = calc_container_size(final_centers, final_angles, final_outer_angle)

    inner = [[final_centers[i][0], final_centers[i][1], final_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
