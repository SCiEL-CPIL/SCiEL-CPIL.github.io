import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Centers of the 11 hexagons in the rigid cluster
    # 1 center
    centers = np.array([
        [0.0, 0.0]
    ])

    # 6 neighbors at dist sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    # 4 more at dist 3
    # Angles 30, 90, 150, 210
    r2 = 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers)

    # Vertices of a unit hexagon (side 1)
    # Base vertices at angle 0 (pointy right)
    # v_k = (cos(60k), sin(60k))
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])

    def get_all_vertices(theta_cluster, theta_inner):
        """
        theta_cluster: rotation of the entire cluster
        theta_inner: rotation of inner hexagons relative to cluster
        Returns: Array of shape (n_hex * 6, 2) containing all vertices
        """
        # Rotate centers by cluster rotation
        c = np.cos(theta_cluster)
        s = np.sin(theta_cluster)
        rot_cluster = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ rot_cluster.T

        # Rotate base vertices by inner rotation (relative to cluster)
        c2 = np.cos(theta_inner)
        s2 = np.sin(theta_inner)
        rot_inner = np.array([[c2, -s2], [s2, c2]])
        rotated_verts = base_vertices @ rot_inner.T

        all_vertices = []
        for i in range(n_hex):
            verts = rotated_verts + rotated_centers[i]
            all_vertices.append(verts)

        return np.vstack(all_vertices)

    def objective(params):
        """
        params: [theta_outer, theta_cluster, theta_inner]
        Minimizes the side length S of the outer hexagon.
        Uses smooth max via log-sum-exp for better optimization.
        """
        theta_outer = params[0]
        theta_cluster = params[1]
        theta_inner = params[2]

        # Get all vertices
        verts = get_all_vertices(theta_cluster, theta_inner)

        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_outer
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)  # shape (6, 2)

        # Project all vertices onto all normals
        dots = verts @ normals.T  # shape (n_hex*6, 6)

        # Use smooth max (log-sum-exp) for each normal direction
        beta = 50.0  # Temperature for soft-max
        smooth_max = np.log(np.sum(np.exp(beta * dots), axis=0)) / beta

        # Required S for each direction
        S_values = smooth_max / (np.sqrt(3) / 2)

        # Return the maximum required S (smoothed)
        beta_outer = 20.0
        S_smooth = np.log(np.sum(np.exp(beta_outer * S_values))) / beta_outer

        return S_smooth

    def exact_objective(params):
        """
        Exact (nonsmooth) objective for final polish.
        """
        theta_outer = params[0]
        theta_cluster = params[1]
        theta_inner = params[2]

        verts = get_all_vertices(theta_cluster, theta_inner)

        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_outer
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)

        dots = verts @ normals.T
        max_proj = np.max(dots)

        S = max_proj / (np.sqrt(3) / 2)
        return S

    # Initial guess
    x0 = np.array([0.0, 0.0, 0.0])

    # Bounds
    bounds = [(0, np.pi/3), (0, np.pi/3), (0, np.pi/3)]

    best_score = None
    best_params = None

    # Multi-start with different initial configurations
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break

        if i == 0:
            x_start = x0
        elif i == 1:
            x_start = np.array([0.0, 0.0, np.pi/6])
        elif i == 2:
            x_start = np.array([0.0, np.pi/6, 0.0])
        elif i == 3:
            x_start = np.array([0.0, np.pi/6, np.pi/6])
        elif i == 4:
            x_start = np.array([np.pi/6, 0.0, 0.0])
        elif i == 5:
            x_start = np.array([np.pi/6, 0.0, np.pi/6])
        elif i == 6:
            x_start = np.array([np.pi/6, np.pi/6, 0.0])
        elif i == 7:
            x_start = np.array([np.pi/6, np.pi/6, np.pi/6])
        else:
            x_start = np.random.uniform(0, np.pi/3, 3)

        try:
            # First pass with smooth objective
            res = minimize(objective, x_start, method='Powell', bounds=bounds,
                          options={'maxiter': 300, 'disp': False})

            if res.fun < (best_score if best_score is not None else float('inf')):
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish with exact objective
    if best_params is not None and time.time() < deadline - 5:
        try:
            res2 = minimize(exact_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})

            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    if best_params is None:
        return _fallback_solution()

    # Extract best solution
    theta_outer = best_params[0]
    theta_cluster = best_params[1]
    theta_inner = best_params[2]

    # Compute final centers and angles
    c = np.cos(theta_cluster)
    s = np.sin(theta_cluster)
    rot_cluster = np.array([[c, -s], [s, c]])
    final_centers = centers @ rot_cluster.T

    inner = []
    for i in range(n_hex):
        inner.append([final_centers[i][0], final_centers[i][1], np.rad2deg(theta_inner)])

    # Compute exact side length
    side = exact_objective(best_params)

    # Add tiny margin for safety
    side += 1e-9

    return inner, (0.0, 0.0), side, np.rad2deg(theta_outer)

def _fallback_solution():
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))

    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])

    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
