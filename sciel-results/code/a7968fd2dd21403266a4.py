import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    N = 12

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def sat_separated(v1, v2):
        """Check if two convex polygons are separated using SAT. Returns min separation (positive = separated)."""
        min_sep = float('inf')
        normals = []
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                nx, ny = -edge[1], edge[0]
                norm = math.sqrt(nx*nx + ny*ny)
                if norm > 1e-12:
                    nx /= norm
                    ny /= norm
                normals.append((nx, ny))
        
        for nx, ny in normals:
            proj1 = np.dot(v1, [nx, ny])
            proj2 = np.dot(v2, [nx, ny])
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
            if min_sep <= 0:
                return min_sep
        return min_sep

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute the minimum side length of a regular hexagon (centered at ocx,ocy, rotated by otheta) containing all inner hexagons."""
        otheta_rad = math.radians(otheta)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(N):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def get_penalty(centers, angles):
        """Compute overlap penalty between all pairs of inner hexagons."""
        verts = []
        for i in range(N):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = sat_separated(verts[i], verts[j])
                if sep < 0:
                    penalty += sep * sep
        return penalty

    def is_valid(centers, angles):
        """Check if all inner hexagons are disjoint."""
        verts = []
        for i in range(N):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(N):
            for j in range(i + 1, N):
                if sat_separated(verts[i], verts[j]) < -1e-9:
                    return False
        return True

    def objective(params):
        if time.time() > deadline:
            return 1e9
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(N)]
        angles = [params[24 + i] for i in range(N)]
        ocx = params[24 + N]
        ocy = params[24 + N + 1]
        otheta = params[24 + N + 2]
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        pen = get_penalty(centers, angles)
        return side + 500.0 * pen

    def generate_hexagonal_cluster():
        """Generate a hexagonal cluster of 12 hexagons (center + ring of 6 + 5 from second ring)."""
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=1e-6)
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=1e-6):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc, atol=1e-6):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        angles = [30.0] * N
        return centers, angles

    def generate_rectangular_grid():
        """Generate a 3x4 rectangular grid of hexagons."""
        dx = math.sqrt(3) + 0.01
        dy = 3.0 + 0.01
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(30.0)
        return centers, angles

    def generate_two_rings():
        """Generate two concentric rings of 6 hexagons each (no center)."""
        r1 = 2.0
        r2 = 4.0
        centers = []
        angles = []
        for i in range(6):
            theta = i * math.pi / 3
            centers.append([r1 * math.cos(theta), r1 * math.sin(theta)])
            angles.append(30.0)
        for i in range(6):
            theta = i * math.pi / 3 + math.pi / 6
            centers.append([r2 * math.cos(theta), r2 * math.sin(theta)])
            angles.append(30.0)
        return centers, angles

    def generate_compact_hex():
        """Generate a compact hexagonal arrangement: 1 center + 6 ring + 5 more tightly packed."""
        d = math.sqrt(3) + 0.005
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        # Add 5 more hexagons in the gaps of the second ring, slightly pulled in
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=1e-6)
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=1e-6):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc, atol=1e-6):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        # Pull them slightly toward center for compactness
        for i in range(5):
            c = ring2_candidates[i].copy()
            c *= 0.98
            centers.append(c)
        angles = [30.0] * N
        return centers, angles

    def params_from_config(centers, angles, ocx=0.0, ocy=0.0, otheta=0.0):
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(ocx)
        params.append(ocy)
        params.append(otheta)
        return np.array(params)

    # Generate initial configurations
    configs = []
    c1, a1 = generate_hexagonal_cluster()
    configs.append(params_from_config(c1, a1))
    c2, a2 = generate_rectangular_grid()
    configs.append(params_from_config(c2, a2))
    c3, a3 = generate_two_rings()
    configs.append(params_from_config(c3, a3))
    c4, a4 = generate_compact_hex()
    configs.append(params_from_config(c4, a4))

    best_params = None
    best_val = float('inf')

    # Phase 1: Nelder-Mead from each initial config
    for init_params in configs:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(N)]
            angles = [res.x[24+i] for i in range(N)]
            ocx, ocy, otheta = res.x[24+N], res.x[24+N+1], res.x[24+N+2]
            
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, ocx, ocy, otheta)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            else:
                init_centers = [[init_params[i*2], init_params[i*2+1]] for i in range(N)]
                init_angles = [init_params[24+i] for i in range(N)]
                if is_valid(init_centers, init_angles):
                    val = compute_outer_side(init_centers, init_angles, 
                                             init_params[24+N], init_params[24+N+1], init_params[24+N+2])
                    if val < best_val:
                        best_val = val
                        best_params = init_params.copy()
        except Exception:
            pass

    # Phase 2: Differential evolution for global search
    if time.time() < deadline - 45:
        try:
            bounds = [(-5, 5)] * 24 + [(0, 360)] * N + [(-2, 2), (-2, 2), (-30, 30)]
            de_res = differential_evolution(objective, bounds, maxiter=100, 
                                            popsize=25, tol=1e-10, 
                                            seed=42, polish=False,
                                            workers=-1)
            if de_res.success or de_res.fun < best_val:
                centers = [[de_res.x[i*2], de_res.x[i*2+1]] for i in range(N)]
                angles = [de_res.x[24+i] for i in range(N)]
                ocx, ocy, otheta = de_res.x[24+N], de_res.x[24+N+1], de_res.x[24+N+2]
                if is_valid(centers, angles):
                    val = compute_outer_side(centers, angles, ocx, ocy, otheta)
                    if val < best_val:
                        best_val = val
                        best_params = de_res.x.copy()
        except Exception:
            pass

    # Phase 3: If we still have no valid solution, use fallback
    if best_params is None:
        best_params = configs[0].copy()
        best_val = compute_outer_side(
            [[best_params[i*2], best_params[i*2+1]] for i in range(N)],
            [best_params[24+i] for i in range(N)],
            best_params[24+N], best_params[24+N+1], best_params[24+N+2]
        )

    # Phase 4: Fine polish with Nelder-Mead from best solution
    if time.time() < deadline - 20:
        try:
            polish_res = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            centers = [[polish_res.x[i*2], polish_res.x[i*2+1]] for i in range(N)]
            angles = [polish_res.x[24+i] for i in range(N)]
            ocx, ocy, otheta = polish_res.x[24+N], polish_res.x[24+N+1], polish_res.x[24+N+2]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, ocx, ocy, otheta)
                if val < best_val:
                    best_val = val
                    best_params = polish_res.x.copy()
        except Exception:
            pass

    # Phase 5: Fine grid search on outer center and angle
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(N)]
    angles = [best_params[24+i] for i in range(N)]
    best_side = best_val
    best_ocx = best_params[24+N]
    best_ocy = best_params[24+N+1]
    best_otheta = best_params[24+N+2]
    
    # Coarse grid on outer center and angle
    for ocx in np.linspace(-1.5, 1.5, 15):
        for ocy in np.linspace(-1.5, 1.5, 15):
            for da in np.linspace(-30, 30, 31):
                s = compute_outer_side(centers, angles, best_ocx + ocx, best_ocy + ocy, best_otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx = best_ocx + ocx
                    best_ocy = best_ocy + ocy
                    best_otheta = best_otheta + da

    # Fine grid
    for ocx in np.linspace(-0.5, 0.5, 11):
        for ocy in np.linspace(-0.5, 0.5, 11):
            for da in np.linspace(-5, 5, 21):
                s = compute_outer_side(centers, angles, best_ocx + ocx, best_ocy + ocy, best_otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx = best_ocx + ocx
                    best_ocy = best_ocy + ocy
                    best_otheta = best_otheta + da

    # Very fine grid
    for ocx in np.linspace(-0.1, 0.1, 7):
        for ocy in np.linspace(-0.1, 0.1, 7):
            for da in np.linspace(-1, 1, 7):
                s = compute_outer_side(centers, angles, best_ocx + ocx, best_ocy + ocy, best_otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx = best_ocx + ocx
                    best_ocy = best_ocy + ocy
                    best_otheta = best_otheta + da

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
