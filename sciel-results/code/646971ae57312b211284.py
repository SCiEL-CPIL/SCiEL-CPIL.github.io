import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """

    N = 11
    # 12 edge normals of a unit hexagon (angle 0): outward normals at 30, 90, 150, 210, 270, 330 deg
    _normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in (30, 90, 150, 210, 270, 330)
    ])
    _normals = np.vstack([_normals, -_normals])  # 12 normals
    _apothem = np.sqrt(3.0) / 2.0  # apothem of unit hexagon = s*sqrt(3)/2 = sqrt(3)/2

    def get_vertices(cx, cy, s=1.0, angle_deg=0.0):
        angles = np.deg2rad(angle_deg) + np.deg2rad(60.0 * np.arange(6))
        return np.stack([cx + s * np.cos(angles), cy + s * np.sin(angles)], axis=-1)

    def calc_required_S(points):
        """Minimal S for a hexagon centered at origin, angle 0, to contain all points."""
        x = points[:, 0]
        y = points[:, 1]
        sqrt3 = np.sqrt(3.0)
        S1 = x + y / sqrt3
        S2 = -x + y / sqrt3
        S3 = -x - y / sqrt3
        S4 = x - y / sqrt3
        S5 = 2 * y / sqrt3
        S6 = -2 * y / sqrt3
        return float(np.max(np.maximum.reduce([S1, S2, S3, S4, S5, S6])))

    def all_vertices(centers, angles_deg):
        """Compute all 66 vertices (11*6) as (66, 2)."""
        verts = []
        for i in range(N):
            verts.append(get_vertices(centers[i, 0], centers[i, 1], 1.0, angles_deg[i]))
        return np.vstack(verts)

    def overlap_penalty(centers, angles_deg):
        """
        For each pair, compute the minimum 'separation' along the 12 edge normals.
        Separation = (gap between projected intervals) where positive = separated.
        If any separation < 0, they overlap; penalty = sum of negative separations squared.
        """
        # Precompute rotated normals for each hexagon
        # For hexagon i with angle a_i, its normals are _normals rotated by a_i
        # Actually, the 12 edge normals for hexagon i are the 6 outward normals of hex i
        # and the 6 outward normals of hex j. The separation check uses all 12.
        # Simpler: for each pair (i,j), check all 12 normals (6 from i's orientation, 6 from j's).
        # But since hexagons are regular, the 12 normals are just the union of the two sets of 6.
        # For a unit hexagon with angle a, outward normals are at angles a+30+60k.
        # The 12 normals for pair (i,j) are: normals_i (6) + normals_j (6).
        # Separation along normal n:
        #   proj_i_max = max over vertices of hex i of (v - c_i) . n
        #   proj_i_min = min over vertices of hex i of (v - c_i) . n
        #   Similarly for hex j.
        #   The gap = min(proj_i_min, proj_j_min) - max(proj_i_max, proj_j_max) ... no.
        # Actually: the intervals [min_i, max_i] and [min_j, max_j] are apart if
        #   max_i < min_j or max_j < min_i.
        #   separation = min(min_j - max_i, min_i - max_j)
        #   If separation > 0, they are separated along this normal.
        # For the pair to be non-overlapping, at least one normal must have separation > 0.
        # Overlap penalty: if for ALL 12 normals, separation <= 0, they overlap.
        # But the task says: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
        # So valid if max over 12 normals of separation > 1e-6.
        # Penalty: if max_separation < 1e-6, penalize (1e-6 - max_separation)^2.
        
        # For a unit regular hexagon centered at origin with angle a, the projection onto
        # a unit normal n is: max = apothem + |n . (vertex direction)| ... 
        # Actually, for a regular hexagon with apothem r = sqrt(3)/2, the support function
        # in direction n is: h(n) = r * max_k cos(angle(n) - (a + 60k + 30)) ... 
        # Simpler: just compute from vertices.
        
        total_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Vertices of hex i and j
                vi = get_vertices(centers[i, 0], centers[i, 1], 1.0, angles_deg[i])
                vj = get_vertices(centers[j, 0], centers[j, 1], 1.0, angles_deg[j])
                
                # 12 normals: 6 from hex i, 6 from hex j
                angles_i = np.deg2rad(angles_deg[i]) + np.deg2rad(30 + 60 * np.arange(6))
                normals_i = np.stack([np.cos(angles_i), np.sin(angles_i)], axis=-1)
                angles_j = np.deg2rad(angles_deg[j]) + np.deg2rad(30 + 60 * np.arange(6))
                normals_j = np.stack([np.cos(angles_j), np.sin(angles_j)], axis=-1)
                
                all_normals = np.vstack([normals_i, normals_j])  # 12 normals
                
                # Project vertices
                pi = vi @ all_normals.T  # (6, 12)
                pj = vj @ all_normals.T  # (6, 12)
                
                max_i = pi.max(axis=0)
                min_i = pi.min(axis=0)
                max_j = pj.max(axis=0)
                min_j = pj.min(axis=0)
                
                # Separation along each normal
                sep1 = min_j - max_i  # if > 0, j is above i
                sep2 = min_i - max_j  # if > 0, i is above j
                separation = np.maximum(sep1, sep2)  # (12,)
                
                max_sep = separation.max()
                if max_sep < 1e-6:
                    total_penalty += (1e-6 - max_sep) ** 2
        
        return total_penalty

    def cost_function(params):
        centers = params[:22].reshape(N, 2)
        angles_deg = params[22:33]
        
        verts = all_vertices(centers, angles_deg)
        S = calc_required_S(verts)
        
        pen = overlap_penalty(centers, angles_deg)
        
        return S + 1000.0 * pen

    def run_optimization(initial_params):
        res = minimize(cost_function, initial_params, method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.success or res.fun < np.inf:
            centers = res.x[:22].reshape(N, 2)
            angles_deg = res.x[22:33]
            # Verify validity
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    vi = get_vertices(centers[i, 0], centers[i, 1], 1.0, angles_deg[i])
                    vj = get_vertices(centers[j, 0], centers[j, 1], 1.0, angles_deg[j])
                    angles_i = np.deg2rad(angles_deg[i]) + np.deg2rad(30 + 60 * np.arange(6))
                    normals_i = np.stack([np.cos(angles_i), np.sin(angles_i)], axis=-1)
                    angles_j = np.deg2rad(angles_deg[j]) + np.deg2rad(30 + 60 * np.arange(6))
                    normals_j = np.stack([np.cos(angles_j), np.sin(angles_j)], axis=-1)
                    all_normals = np.vstack([normals_i, normals_j])
                    pi = vi @ all_normals.T
                    pj = vj @ all_normals.T
                    max_i = pi.max(axis=0)
                    min_i = pi.min(axis=0)
                    max_j = pj.max(axis=0)
                    min_j = pj.max(axis=0)
                    sep1 = min_j - max_i
                    sep2 = min_i - max_j
                    separation = np.maximum(sep1, sep2)
                    if separation.max() < 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                verts = all_vertices(centers, angles_deg)
                S = calc_required_S(verts)
                return centers, angles_deg, S
        return None, None, np.inf

    # Initial configuration: 7 central + 4 outer, all angle 0
    sqrt3 = np.sqrt(3)
    h = 1.5
    w = sqrt3 / 2

    c0 = np.array([0, 0])
    c1 = np.array([w, h])
    c2 = np.array([-w, h])
    c3 = np.array([sqrt3, 0])
    c4 = np.array([-sqrt3, 0])
    c5 = np.array([w, -h])
    c6 = np.array([-w, -h])
    c7 = np.array([2.598, 1.5])
    c8 = np.array([-2.598, 1.5])
    c9 = np.array([2.598, -1.5])
    c10 = np.array([-2.598, -1.5])

    init_centers = np.vstack([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    init_angles = np.zeros(N)
    init_params = np.concatenate([init_centers.flatten(), init_angles])

    best_centers = None
    best_angles = None
    best_S = np.inf

    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin

    # 1. Optimize the initial good guess
    centers, angles, S = run_optimization(init_params)
    if centers is not None and S < best_S:
        best_centers, best_angles, best_S = centers, angles, S

    if time.time() > deadline:
        return _build_result(best_centers, best_angles, best_S)

    # 2. Random restarts with perturbed positions and random angles
    np.random.seed(42)
    for i in range(30):
        if time.time() > deadline - 15:
            break

        if best_centers is not None:
            current_centers = best_centers + np.random.normal(0, 0.15, size=(N, 2))
            current_angles = best_angles + np.random.normal(0, 5.0, size=N)
        else:
            current_centers = np.random.normal(0, 1.5, size=(N, 2))
            current_angles = np.random.uniform(-30, 30, size=N)

        current_params = np.concatenate([current_centers.flatten(), current_angles])
        centers, angles, S = run_optimization(current_params)
        if centers is not None and S < best_S:
            best_centers, best_angles, best_S = centers, angles, S

    # 3. Try a few more restarts with larger perturbations
    for i in range(10):
        if time.time() > deadline - 10:
            break

        current_centers = np.random.normal(0, 2.0, size=(N, 2))
        current_angles = np.random.uniform(-30, 30, size=N)
        current_params = np.concatenate([current_centers.flatten(), current_angles])
        centers, angles, S = run_optimization(current_params)
        if centers is not None and S < best_S:
            best_centers, best_angles, best_S = centers, angles, S

    return _build_result(best_centers, best_angles, best_S)


def _build_result(centers, angles_deg, S):
    if centers is None:
        # Fallback to seed-like solution if optimization fails
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
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = float(angles_deg[i]) if angles_deg is not None else 0.0

    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = float(S)

    return inner_hex_data, outer_hex_data, outer_hex_side_length
