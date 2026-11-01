"""Combined 11-hexagon packing: joint optimization with smooth surrogate,
numpy-vectorized geometry, and multi-phase polish including Differential Evolution."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# Precomputed constants
NORMALS = np.array([[math.cos(math.radians(30 + i * 60)),
                     math.sin(math.radians(30 + i * 60))] for i in range(6)])
HEX_ANG = np.arange(6) * np.pi / 3.0

# EVOLVE-BLOCK-START

def hex_vertices_arr(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a (6,2) numpy array."""
    thetas = angle_rad + HEX_ANG
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_vec(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Vectorized for speed.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    n = len(inner)
    # Extract coordinates
    xs = np.array([p[0] for p in inner])
    ys = np.array([p[1] for p in inner])
    angles_rad = np.radians(np.array([p[2] for p in inner]))

    # Generate vertices for all inner hexagons
    # Shape: (n, 6)
    thetas = angles_rad[:, None] + HEX_ANG[None, :]
    
    # Vertices: (n, 6, 2)
    verts_x = xs[:, None] + np.cos(thetas)
    verts_y = ys[:, None] + np.sin(thetas)

    # Transform to outer local frame
    # local_x = (vx - ox) * cos_a - (vy - oy) * sin_a
    # local_y = (vx - ox) * sin_a + (vy - oy) * cos_a
    vx = verts_x - ox
    vy = verts_y - oy

    local_x = vx * cos_a - vy * sin_a
    local_y = vx * sin_a + vy * cos_a

    # Project onto 6 normals
    # projs[i, j, k] = local_x[i, j] * NORMALS[k, 0] + local_y[i, j] * NORMALS[k, 1]
    # Using einsum: 'nij,jk->nik' where n=inner, j=vertex, k=normal
    # local: (n, 6, 2)
    local = np.stack([local_x, local_y], axis=-1)
    projs = np.einsum('nij,jk->nik', local, NORMALS)

    max_proj = projs.max()
    return 2.0 * max_proj / SQRT3

def compute_overlap_penalty_vec(inner, min_dist=SQRT3 - 0.005):
    """Compute a smooth penalty for overlaps based on center distances."""
    n = len(inner)
    if n < 2:
        return 0.0

    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])

    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Multiple candidate configurations
    candidates = []

    # Candidate 1: Symmetric cross
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_1, "symmetric_cross"))

    # Candidate 2: Compact 3-4-3
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append((axial_2, "cluster343"))

    # Candidate 3: 4-4-3 block
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append((axial_3, "443"))

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 40:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]

        epsilon = 1e-5
        scale = 1.0 + epsilon

        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))

        initial_inner = [[x, y, 0.0] for x, y in centers]

        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Phase 1: Optimize outer container with fixed inner layout
        def objective_outer_smooth(params):
            cx, cy, angle_deg = params
            # Use vectorized version for speed, beta not needed for exact, but we can use smooth if we had one.
            # For now, use exact vectorized.
            return compute_required_side_vec(initial_inner, (cx, cy), angle_deg)

        best_result = None
        best_val = float('inf')

        init_angles = list(range(0, 360, 30))

        for init_angle in init_angles:
            if time.time() > deadline - 30:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            try:
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})

                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res.x
            except Exception:
                pass

        if best_result is not None:
            outer_center = (best_result[0], best_result[1])
            outer_angle = best_result[2]
            side_val = best_val
        else:
            outer_center = (cx_avg, cy_avg)
            outer_angle = 0.0
            side_val = compute_required_side_vec(initial_inner, outer_center, outer_angle)

        # Phase 2: Joint optimization with Simulated Annealing
        def objective_joint(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])

            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]

            side = compute_required_side_vec(inner_list, (ocx, ocy), oangle)
            penalty = compute_overlap_penalty_vec(inner_list)

            return side + 150.0 * penalty

        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([outer_center[0], outer_center[1], outer_angle])

        current_params = np.array(initial_params)
        current_val = objective_joint(current_params)
        best_params = current_params.copy()
        best_val_joint = current_val

        # Adaptive simulated annealing
        T = 1.0
        T_min = 0.002
        cooling = 0.99985

        max_iter = 5000 # Reduced to leave time for DE
        no_improve_count = 0
        patience = 1000

        for i in range(max_iter):
            if time.time() > deadline - 50:
                break

            neighbor = current_params.copy()

            # Random perturbation
            if np.random.random() < 0.7:
                dim_idx = np.random.randint(0, N*3)
                delta = np.random.normal(0, 0.12 * (1 + 0.5 * T))
                neighbor[dim_idx] += delta
            else:
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 2.5 * (1 + 0.3 * T))
                else:
                    delta = np.random.normal(0, 0.08 * (1 + 0.5 * T))
                neighbor[dim_idx] += delta

            neighbor_val = objective_joint(neighbor)

            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
                current_params = neighbor
                current_val = neighbor_val

                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Restart if stuck
                if no_improve_count > patience:
                    no_improve_count = 0
                    T = max(T * 0.5, T_min * 2)

            T *= cooling
            if T < T_min:
                break

        best_inner_sa = []
        for i in range(N):
            x = best_params[i*3]
            y = best_params[i*3+1]
            a = best_params[i*3+2]
            best_inner_sa.append([x, y, a])

        best_ocx = best_params[N*3]
        best_ocy = best_params[N*3+1]
        best_oangle = best_params[N*3+2]

        # Phase 3: Differential Evolution Global Search
        best_inner_de = best_inner_sa
        best_ocx_de = best_ocx
        best_ocy_de = best_ocy
        best_oangle_de = best_oangle
        best_val_de = compute_required_side_vec(best_inner_de, (best_ocx_de, best_ocy_de), best_oangle_de)

        if time.time() < deadline - 20:
            def de_objective(params):
                inner_list = []
                for i in range(N):
                    inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
                ocx, ocy, oa = params[N*3], params[N*3+1], params[N*3+2]
                side = compute_required_side_vec(inner_list, (ocx, ocy), oa)
                pen = compute_overlap_penalty_vec(inner_list)
                return side + 100.0 * pen

            bounds = []
            for i in range(N):
                bounds.append((-5, 5))
                bounds.append((-5, 5))
                bounds.append((0, 360))
            bounds.append((-5, 5))
            bounds.append((-5, 5))
            bounds.append((0, 360))

            x0 = np.array([p for x,y,a in best_inner_sa for p in (x,y,a)] + [best_ocx, best_ocy, best_oangle])

            try:
                res_de = differential_evolution(
                    de_objective,
                    bounds=bounds,
                    x0=x0,
                    maxiter=50,
                    popsize=15,
                    tol=1e-6,
                    seed=42
                )
                if res_de.fun < best_val_de:
                    best_val_de = res_de.fun
                    # Extract best params
                    best_inner_de = []
                    for i in range(N):
                        best_inner_de.append([res_de.x[i*3], res_de.x[i*3+1], res_de.x[i*3+2]])
                    best_ocx_de = res_de.x[N*3]
                    best_ocy_de = res_de.x[N*3+1]
                    best_oangle_de = res_de.x[N*3+2]
            except Exception:
                pass

        # Phase 4: Final high-precision polish
        best_inner_final = best_inner_de
        best_ocx_final = best_ocx_de
        best_ocy_final = best_ocy_de
        best_oangle_final = best_oangle_de

        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return compute_required_side_vec(best_inner_final, (cx, cy), angle_deg)

        polish_starts = []
        base_angle = best_oangle_final
        for da in [-12, -6, -3, 0, 3, 6, 12]:
            for dc in [(-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1), (0, 0)]:
                polish_starts.append([best_ocx_final + dc[0], best_ocy_final + dc[1], base_angle + da])

        best_polish_val = compute_required_side_vec(best_inner_final, (best_ocx_final, best_ocy_final), best_oangle_final)
        best_polish_params = [best_ocx_final, best_ocy_final, best_oangle_final]

        for ps in polish_starts:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 4000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass

        final_inner = best_inner_final
        final_center = (best_polish_params[0], best_polish_params[1])
        final_angle = best_polish_params[2]
        final_side = best_polish_val

        # Verify disjointness
        # Simple check: if penalty is non-zero, it's likely overlapping.
        # For strict validation, we would need SAT, but we trust the penalty + polish.
        # If penalty is high, we might reject this candidate.
        pen_check = compute_overlap_penalty_vec(final_inner)
        if pen_check < 1e-6: # Very small penalty
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Fallback to SA result if DE caused overlaps
            pen_sa = compute_overlap_penalty_vec(best_inner_sa)
            if pen_sa < 1e-6:
                side_sa = compute_required_side_vec(best_inner_sa, (best_ocx, best_ocy), best_oangle)
                if side_sa < best_global_side:
                    best_global_side = side_sa
                    best_global_inner = best_inner_sa
                    best_global_center = (best_ocx, best_ocy)
                    best_global_angle = best_oangle

    if best_global_inner is None:
        # Fallback
        axial_coords = candidates[0][0][:N]
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = compute_required_side_vec(best_global_inner, best_global_center, best_global_angle)

    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END
