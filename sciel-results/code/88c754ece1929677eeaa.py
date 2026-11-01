from scipy.optimize import minimize
import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_normals(verts):
        normals = []
        for i in range(len(verts)):
            edge = verts[(i + 1) % len(verts)] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normals.append(normal / norm)
        return normals

    def penetration_depth(v1, v2):
        """Compute minimum penetration depth between two convex polygons."""
        normals = get_normals(v1) + get_normals(v2)
        min_overlap = np.inf
        for normal in normals:
            p1 = v1 @ normal
            p2 = v2 @ normal
            overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if overlap < min_overlap:
                min_overlap = overlap
        return max(0.0, min_overlap)

    def inside_violation(inner_verts, outer_verts):
        """How much inner_verts violate being inside outer_verts."""
        outer_normals = get_normals(outer_verts)
        max_viol = 0.0
        for normal in outer_normals:
            inner_max = (inner_verts @ normal).max()
            outer_max = (outer_verts @ normal).max()
            viol = inner_max - outer_max
            if viol > max_viol:
                max_viol = viol
        return max(0.0, max_viol)

    def objective(params):
        if time.time() > deadline:
            return 1e10

        inner_params = params[:3 * N]
        cx, cy, s, psi = params[3 * N:3 * N + 4]

        if s <= 0.5:
            return 1e6

        outer_verts = hex_vertices(cx, cy, s, psi)

        total_penalty = 0.0
        inner_verts_list = []

        for i in range(N):
            x = inner_params[3 * i]
            y = inner_params[3 * i + 1]
            a = inner_params[3 * i + 2]
            verts = hex_vertices(x, y, 1.0, a)
            inner_verts_list.append(verts)
            viol = inside_violation(verts, outer_verts)
            total_penalty += viol

        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd

        return s + 20.0 * total_penalty

    def make_hex_config(d, rot=0.0):
        pos = []
        ang = []
        pos.append([0.0, 0.0])
        ang.append(rot)
        for k in range(6):
            a = np.radians(60 * k) + rot
            pos.append([d * np.cos(a), d * np.sin(a)])
            ang.append(rot)
        for k in range(5):
            a = np.radians(60 * k) + rot
            pos.append([2 * d * np.cos(a), 2 * d * np.sin(a)])
            ang.append(rot)
        return np.array(pos), np.array(ang)

    def make_staggered_config(d, rot=0.0):
        pos = []
        ang = []
        for row in range(3):
            for col in range(4):
                x = col * d
                y = row * d * np.sqrt(3)
                if row % 2 == 1:
                    x += d / 2
                pos.append([x, y])
                ang.append(rot)
        return np.array(pos), np.array(ang)

    def compute_minimal_outer(all_verts):
        """Compute minimal enclosing hexagon for given vertices."""
        def calc_side(theta):
            angles = theta + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = all_verts @ normals.T
            widths = dots.max(axis=0) - dots.min(axis=0)
            max_width = np.max(widths)
            apothem = max_width / 2.0
            side = 2 * apothem / np.sqrt(3)
            return side

        thetas_coarse = np.linspace(0, np.pi / 3, 180)
        sides = np.array([calc_side(th) for th in thetas_coarse])
        best_idx = np.argmin(sides)

        lo = thetas_coarse[max(0, best_idx - 2)]
        hi = thetas_coarse[min(len(thetas_coarse) - 1, best_idx + 2)]

        res = minimize_scalar_wrapper(calc_side, lo, hi)
        return res[0], res[1]

    def minimize_scalar_wrapper(func, lo, hi):
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(func, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-6})
        return res.fun, res.x

    def verify_solution(inner_positions, angles, outer_center, outer_side, outer_angle):
        """Verify that the solution is valid."""
        outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, np.radians(outer_angle))

        for i in range(N):
            verts = hex_vertices(inner_positions[i, 0], inner_positions[i, 1], 1.0, angles[i])
            if inside_violation(verts, outer_verts) > 1e-6:
                return False

        for i in range(N):
            for j in range(i + 1, N):
                v1 = hex_vertices(inner_positions[i, 0], inner_positions[i, 1], 1.0, angles[i])
                v2 = hex_vertices(inner_positions[j, 0], inner_positions[j, 1], 1.0, angles[j])
                if penetration_depth(v1, v2) > 1e-6:
                    return False
        return True

    # Generate diverse initial configurations
    configs = []
    configs.append(make_hex_config(2.1))
    configs.append(make_hex_config(2.1, rot=np.pi / 6))
    configs.append(make_hex_config(2.2))
    configs.append(make_hex_config(2.3))
    configs.append(make_staggered_config(2.1))
    configs.append(make_staggered_config(2.2))

    best_obj = 1e10
    best_params = None

    for pos, ang in configs:
        if time.time() > deadline:
            break

        # Estimate outer hexagon size
        all_pts = np.vstack([pos, pos + 1.0])  # Rough bounding
        cx0 = np.mean(all_pts[:, 0])
        cy0 = np.mean(all_pts[:, 1])
        s0 = 5.0
        psi0 = 0.0

        params = np.concatenate([pos.flatten(), ang, [cx0, cy0, s0, psi0]])

        try:
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Final refinement with more iterations from best
    if best_params is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Extract solution
    if best_params is not None and len(best_params) == 3 * N + 4:
        best_positions = np.array([[best_params[i * 3], best_params[i * 3 + 1]] for i in range(N)])
        best_angles = np.array([best_params[i * 3 + 2] for i in range(N)])
        outer_center = (best_params[3 * N], best_params[3 * N + 1])
        outer_side = best_params[3 * N + 2]
        outer_angle_deg = np.degrees(best_params[3 * N + 3])

        # Verify and if invalid, try to fix
        if not verify_solution(best_positions, best_angles, outer_center, outer_side, outer_angle_deg):
            # Recompute minimal enclosing hexagon
            all_verts = []
            for i in range(N):
                v = hex_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
                all_verts.append(v)
            all_verts = np.vstack(all_verts)

            outer_side, outer_angle_rad = compute_minimal_outer(all_verts)
            angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = all_verts @ normals.T
            mins = dots.min(axis=0)
            maxs = dots.max(axis=0)
            mids = (mins + maxs) / 2.0
            c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
            outer_center = tuple(c)
            outer_angle_deg = np.degrees(outer_angle_rad)
    else:
        # Fallback
        pos, ang = make_hex_config(2.5)
        best_positions = pos
        best_angles = ang
        all_verts = []
        for i in range(N):
            v = hex_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        outer_side, outer_angle_rad = compute_minimal_outer(all_verts)
        angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        outer_center = tuple(c)
        outer_angle_deg = np.degrees(outer_angle_rad)

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(N)]

    return inner, outer_center, outer_side, outer_angle_deg
