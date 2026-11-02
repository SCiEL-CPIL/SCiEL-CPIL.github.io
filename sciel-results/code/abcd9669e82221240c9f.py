import numpy as np
from scipy.optimize import minimize, differential_evolution, nelder_mead
import time

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # --- Geometry helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        if np.min(overlaps) <= 1e-9:
            return 0.0
            
        return np.min(overlaps)

    def min_side_given_verts(verts, outer_center, outer_angle_rad):
        """
        Minimum side length of a regular hexagon (centre, angle) that contains
        all points in verts.
        verts: (M, 2)
        outer_center: (2,)
        outer_angle_rad: float
        Returns: float (side length)
        """
        n = np.column_stack([
            np.cos(outer_angle_rad + np.radians(np.arange(0, 360, 60))),
            np.sin(outer_angle_rad + np.radians(np.arange(0, 360, 60)))
        ])  # (6, 2)
        
        rel = verts - np.array(outer_center)  # (M, 2)
        proj = rel @ n.T  # (M, 6)
        
        max_abs = np.max(np.abs(proj))
        apothem = max_abs
        return 2.0 * apothem / SQRT3

    def optimal_outer_rotation(verts, outer_center):
        """
        Find the outer rotation angle that minimizes the required side length.
        This is a 1D optimization over the rotation angle.
        """
        def obj(theta):
            return min_side_given_verts(verts, outer_center, theta)
        
        # Try multiple starting points and pick the best
        best_val = np.inf
        best_theta = 0.0
        for theta0 in np.linspace(0, np.pi/3, 13):
            result = minimize(obj, [theta0], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_theta = result.x[0]
        
        return best_val, best_theta

    # --- Cluster Generation ---

    def make_honeycomb_cluster():
        """
        1 centre + 6 ring-1 + 5 ring-2.
        Returns array of shape (N, 3): [x, y, angle_deg]
        """
        gap = 0.05
        d1 = SQRT3 + gap
        d2 = 2 * SQRT3 + gap
        
        centers = []
        centers.append([0.0, 0.0, 0.0])
        for k in range(6):
            a = np.radians(60 * k)
            centers.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
        for k in range(5):
            a = np.radians(60 * k)
            centers.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
        
        return np.array(centers)

    def make_circular_cluster():
        """
        12 hexagons on a circle.
        """
        centers = []
        r = 2.2
        for i in range(N):
            a = 2 * np.pi * i / N
            centers.append([r * np.cos(a), r * np.sin(a), np.degrees(a)])
        return np.array(centers)

    def make_random_cluster(seed):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(-3, 3, (N, 2))
        angles = rng.uniform(0, 60, N)
        return np.column_stack([centers, angles])

    def cluster_valid(base):
        """Check that the base cluster has no overlapping hexagons."""
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(base[i, 0], base[i, 1], 1.0, np.radians(base[i, 2]))
                vj = hex_vertices(base[j, 0], base[j, 1], 1.0, np.radians(base[j, 2]))
                if penetration_depth(vi, vj) > 1e-6:
                    return False
        return True

    def transform_cluster(base, cx, cy, rot_deg):
        """
        Transform base cluster (N,3) by translation (cx, cy) and rotation rot_deg.
        Returns (N, 3) new cluster positions.
        """
        rot = np.radians(rot_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        pos = base[:, :2].copy()
        pos = pos @ R.T + np.array([cx, cy])
        angles = base[:, 2] + rot_deg
        
        return np.column_stack([pos, angles])

    def get_cluster_verts(cluster):
        """
        Get all vertices of all hexagons in the cluster.
        Returns (N*6, 2)
        """
        verts_list = []
        for i in range(N):
            v = hex_vertices(cluster[i, 0], cluster[i, 1], 1.0, np.radians(cluster[i, 2]))
            verts_list.append(v)
        return np.vstack(verts_list)

    # --- Objective Functions ---

    def objective_nested(params, base_cluster):
        """
        params: [cx, cy, rot_deg]
        Uses nested optimization to find optimal outer rotation.
        Returns outer side length.
        """
        cx, cy, rot = params
        
        # Transform cluster
        cluster = transform_cluster(base_cluster, cx, cy, rot)
        verts = get_cluster_verts(cluster)
        
        # Find optimal outer rotation (outer center at (0,0) for symmetry)
        side, _ = optimal_outer_rotation(verts, (0.0, 0.0))
        return side

    # --- Main Search ---

    best_s = np.inf
    best_params = None
    best_base = None
    best_outer_theta = 0.0

    # Generate candidate clusters
    clusters = []
    for base in [make_honeycomb_cluster(), make_circular_cluster(),
                 make_random_cluster(42), make_random_cluster(123),
                 make_random_cluster(999), make_random_cluster(777),
                 make_random_cluster(555)]:
        if cluster_valid(base):
            clusters.append(base)

    if not clusters:
        # Fallback: very spread out
        base = np.column_stack([
            np.array([3.5 * np.cos(2 * np.pi * i / N) for i in range(N)]),
            np.array([3.5 * np.sin(2 * np.pi * i / N) for i in range(N)]),
            np.zeros(N)
        ])
        clusters = [base]

    # Bounds for global search (3D: cx, cy, rot)
    bounds_3d = [
        (-5, 5),    # cx
        (-5, 5),    # cy
        (0, 60),    # cluster rotation (degrees)
    ]

    for base in clusters:
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break

        # Stage 1: Nelder-Mead on 3D cluster parameters
        # The nested optimization handles the outer rotation
        x0 = np.array([0.0, 0.0, 0.0])
        
        try:
            res_nm = minimize(
                objective_nested,
                x0,
                args=(base,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_nm.fun < best_s:
                best_s = res_nm.fun
                best_params = res_nm.x.copy()
                best_base = base.copy()
                
                # Find the optimal outer rotation for this best configuration
                cluster = transform_cluster(best_base, best_params[0], best_params[1], best_params[2])
                verts = get_cluster_verts(cluster)
                _, best_outer_theta = optimal_outer_rotation(verts, (0.0, 0.0))
                
        except Exception as e:
            pass

        # Stage 2: Try multiple starting points for Nelder-Mead
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            for x0_try in [np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 15.0]), np.array([0.0, 0.0, 45.0])]:
                try:
                    res_nm2 = minimize(
                        objective_nested,
                        x0_try,
                        args=(base,),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    
                    if res_nm2.fun < best_s:
                        best_s = res_nm2.fun
                        best_params = res_nm2.x.copy()
                        best_base = base.copy()
                        
                        # Find the optimal outer rotation for this best configuration
                        cluster = transform_cluster(best_base, best_params[0], best_params[1], best_params[2])
                        verts = get_cluster_verts(cluster)
                        _, best_outer_theta = optimal_outer_rotation(verts, (0.0, 0.0))
                        
                except Exception as e:
                    pass

        # Stage 3: Final polish if time permits
        elapsed = time.time() - start_time
        if elapsed < deadline - 5 and best_params is not None:
            try:
                res_polish = minimize(
                    objective_nested,
                    best_params,
                    args=(best_base,),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_polish.fun < best_s:
                    best_s = res_polish.fun
                    best_params = res_polish.x.copy()
                    
                    # Find the optimal outer rotation for this best configuration
                    cluster = transform_cluster(best_base, best_params[0], best_params[1], best_params[2])
                    verts = get_cluster_verts(cluster)
                    _, best_outer_theta = optimal_outer_rotation(verts, (0.0, 0.0))
                    
            except Exception as e:
                pass

    # --- Construct Final Result ---

    if best_params is None:
        # Fallback
        base = make_honeycomb_cluster()
        cx, cy, rot = 0, 0, 0
        s = 5.0
        best_outer_theta = 0.0
    else:
        cx, cy, rot = best_params
        s = best_s

    # Apply final transformation to get inner hexagon positions
    final_cluster = transform_cluster(best_base, cx, cy, rot)
    
    # The outer hexagon center is (0, 0) and rotation is best_outer_theta.
    inner = []
    for i in range(N):
        inner.append([final_cluster[i, 0], final_cluster[i, 1], final_cluster[i, 2]])
    
    outer_center = (0.0, 0.0)
    outer_side = s
    outer_angle_deg = np.degrees(best_outer_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg

# Keep the original function name
# The prompt says: "Keep the function names and signatures of the seed program."
# The seed program has `construct()` returning `inner, outer_center, outer_side, outer_angle_deg`.
# My code above does this.

if __name__ == "__main__":
    # Simple test
    inner, center, side, angle = construct()
    print(f"Side: {side}, Center: {center}, Angle: {angle}")
    for h in inner:
        print(h)
