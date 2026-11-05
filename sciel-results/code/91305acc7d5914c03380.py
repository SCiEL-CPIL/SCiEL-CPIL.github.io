import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Vertices of a unit hexagon (side 1) - flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.deg2rad(np.arange(6) * 60)
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    # Center hexagon
    center_hex = np.array([[0.0, 0.0]])

    # 6 neighbor positions at distance sqrt(3) (touching)
    r1 = np.sqrt(3)
    angles_r1 = np.deg2rad(np.arange(6) * 60)
    neighbors_r1 = np.column_stack([r1 * np.cos(angles_r1), r1 * np.sin(angles_r1)])

    # 6 outer positions at distance 3 (touching the neighbors)
    r2 = 3.0
    angles_r2 = np.deg2rad(30 + np.arange(6) * 60)
    neighbors_r2 = np.column_stack([r2 * np.cos(angles_r2), r2 * np.sin(angles_r2)])

    def compute_min_side(centers, theta_cluster, theta_inner, theta_outer):
        """
        Compute the minimum outer hexagon side length needed to enclose all inner hexagons.
        centers: array of shape (n, 2) - positions of inner hexagon centers
        theta_cluster: rotation of the entire cluster
        theta_inner: rotation of each inner hexagon
        theta_outer: rotation of the outer hexagon
        """
        c_c = np.cos(theta_cluster)
        s_c = np.sin(theta_cluster)
        rotated_centers = np.column_stack([
            centers[:, 0] * c_c - centers[:, 1] * s_c,
            centers[:, 0] * s_c + centers[:, 1] * c_c
        ])

        c_i = np.cos(theta_inner)
        s_i = np.sin(theta_inner)
        rotated_verts = np.column_stack([
            base_vertices[:, 0] * c_i - base_vertices[:, 1] * s_i,
            base_vertices[:, 0] * s_i + base_vertices[:, 1] * c_i
        ])

        # All vertices: (n, 6, 2)
        n = len(rotated_centers)
        all_vertices = rotated_verts.reshape(1, 6, 2) + rotated_centers.reshape(n, 1, 2)
        all_vertices = all_vertices.reshape(n * 6, 2)

        # Normals of the outer hexagon sides
        # For a regular hexagon with side S, the distance from center to each side is S*sqrt(3)/2
        # The normals point outward at angles 30, 90, 150, 210, 270, 330 degrees (for pointy-top)
        # or 0, 60, 120, 180, 240, 300 degrees (for flat-top)
        # Let's use the convention where side is measured such that the apothem = S*sqrt(3)/2
        outer_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_outer
        normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])

        # Project all vertices onto all normals
        dots = all_vertices @ normals.T  # (n*6, 6)
        max_proj = np.max(dots, axis=0)  # (6,)

        # Required S for each direction: S = max_proj / (sqrt(3)/2)
        S_values = max_proj / (np.sqrt(3) / 2)
        return np.max(S_values)

    def optimize_for_topology(centers):
        """
        For a given set of centers, find optimal rotations to minimize enclosing hexagon side.
        """
        best_side = float('inf')
        best_params = None

        # Try different combinations of inner and outer rotations
        # Due to symmetry, we can restrict to a fundamental domain
        # theta_outer in [0, 60), theta_cluster in [0, 60), theta_inner in [0, 60)
        
        # Strategy: Use the fact that for a given cluster orientation, 
        # we can analytically find the best outer orientation
        # and the best inner orientation
        
        # Let's do a 2D search over theta_cluster and theta_inner,
        # with theta_outer optimized analytically (or via 1D search)
        
        def objective_2d(params):
            theta_cluster = params[0]
            theta_inner = params[1]
            
            # For given cluster and inner rotations, find best outer rotation
            # by 1D search
            def obj_outer(theta_outer):
                return compute_min_side(centers, theta_cluster, theta_inner, theta_outer)
            
            res_outer = minimize_scalar(obj_outer, bounds=(0, np.pi/3), method='bounded',
                                       options={'xatol': 1e-10})
            return res_outer.fun
        
        # Multi-start 2D optimization
        start_configs = [
            [0.0, 0.0],
            [0.0, np.pi/6],
            [np.pi/6, 0.0],
            [np.pi/6, np.pi/6],
            [np.pi/12, np.pi/12],
            [np.pi/4, np.pi/4],
            [np.pi/3, 0.0],
            [0.0, np.pi/3],
            [np.pi/8, np.pi/8],
            [3*np.pi/8, 3*np.pi/8],
        ]
        
        bounds_2d = [(0, np.pi/3), (0, np.pi/3)]
        
        for x_start in start_configs:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(objective_2d, x_start, method='Nelder-Mead',
                             bounds=bounds_2d,
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                     'adaptive': True, 'disp': False})
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        
        if best_params is None:
            return best_side, None
        
        # Final polish: 3D optimization with tighter tolerances
        def objective_3d(params):
            return compute_min_side(centers, params[0], params[1], params[2])
        
        # Get the outer angle from the 2D solution
        def obj_outer_final(theta_outer):
            return compute_min_side(centers, best_params[0], best_params[1], theta_outer)
        
        res_outer_final = minimize_scalar(obj_outer_final, bounds=(0, np.pi/3), method='bounded',
                                         options={'xatol': 1e-12})
        
        initial_3d = np.array([best_params[0], best_params[1], res_outer_final.x])
        
        try:
            res3 = minimize(objective_3d, initial_3d, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12,
                                  'adaptive': True, 'disp': False})
            if res3.fun < best_side:
                best_side = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
        
        return best_side, best_params

    # Topology 1: Full flower (1 + 6 + 6 = 13 hexagons)
    centers_full = np.vstack([center_hex, neighbors_r1, neighbors_r2])
    side_full, params_full = optimize_for_topology(centers_full)

    # Topology 2: 1 + 6 + 4 (select 4 of 6 outer positions)
    best_side = side_full
    best_centers = centers_full
    best_params = params_full

    # Try all combinations of 4 outer positions
    outer_indices = list(range(6))
    for combo in combinations(outer_indices, 4):
        if time.time() > deadline - 10:
            break
        
        centers_combo = np.vstack([center_hex, neighbors_r1, neighbors_r2[list(combo)]])
        side_combo, params_combo = optimize_for_topology(centers_combo)
        
        if side_combo < best_side:
            best_side = side_combo
            best_centers = centers_combo
            best_params = params_combo

    # Topology 3: 1 + 6 + 5 (select 5 of 6 outer positions)
    for combo in combinations(outer_indices, 5):
        if time.time() > deadline - 5:
            break
        
        centers_combo = np.vstack([center_hex, neighbors_r1, neighbors_r2[list(combo)]])
        side_combo, params_combo = optimize_for_topology(centers_combo)
        
        if side_combo < best_side:
            best_side = side_combo
            best_centers = centers_combo
            best_params = params_combo

    # Topology 4: 1 + 6 only (7 hexagons)
    centers_7 = np.vstack([center_hex, neighbors_r1])
    side_7, params_7 = optimize_for_topology(centers_7)
    if side_7 < best_side:
        best_side = side_7
        best_centers = centers_7
        best_params = params_7

    # Topology 5: 1 + 5 (6 hexagons, one neighbor missing)
    for missing in range(6):
        if time.time() > deadline - 3:
            break
        idx = [i for i in range(6) if i != missing]
        centers_combo = np.vstack([center_hex, neighbors_r1[idx]])
        side_combo, params_combo = optimize_for_topology(centers_combo)
        
        if side_combo < best_side:
            best_side = side_combo
            best_centers = centers_combo
            best_params = params_combo

    # Topology 6: 1 + 4 (5 hexagons)
    for combo in combinations(range(6), 4):
        if time.time() > deadline - 2:
            break
        centers_combo = np.vstack([center_hex, neighbors_r1[list(combo)]])
        side_combo, params_combo = optimize_for_topology(centers_combo)
        
        if side_combo < best_side:
            best_side = side_combo
            best_centers = centers_combo
            best_params = params_combo

    if best_params is None:
        # Fallback: just the full flower with default rotations
        best_params = np.array([0.0, 0.0, 0.0])
        best_side = compute_min_side(centers_full, 0.0, 0.0, 0.0)

    # Extract final parameters
    theta_cluster = best_params[0]
    theta_inner = best_params[1]
    theta_outer = best_params[2]

    # Compute final centers
    c = np.cos(theta_cluster)
    s = np.sin(theta_cluster)
    final_centers = np.column_stack([
        best_centers[:, 0] * c - best_centers[:, 1] * s,
        best_centers[:, 0] * s + best_centers[:, 1] * c
    ])

    inner = []
    for i in range(len(final_centers)):
        inner.append([final_centers[i][0], final_centers[i][1], np.rad2deg(theta_inner)])

    # Compute exact side length with safety margin
    side = compute_min_side(best_centers, theta_cluster, theta_inner, theta_outer)
    side += 1e-9

    return inner, (0.0, 0.0), side, np.rad2deg(theta_outer)

if __name__ == "__main__":
    construct()
