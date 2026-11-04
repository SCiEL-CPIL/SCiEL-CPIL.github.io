import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Unit hexagon vertices (pointy-top, side=1)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, -s], [s, c]])
        return unit_verts @ rot.T * side + np.array([cx, cy])

    def outer_side_length(positions, angles, outer_angle_deg):
        """Minimum side of regular hexagon (centered at origin, rotated by outer_angle_deg) containing all inner hexagons."""
        theta = math.radians(outer_angle_deg)
        # The 6 outward normals of the outer hexagon
        # For a regular hexagon with side s, the distance from center to each edge is s*sqrt(3)/2
        # Vertices are at angles: outer_angle + k*60 for k=0..5
        # Edge normals are at: outer_angle + 30 + k*60 for k=0..5
        normals = np.array([[math.cos(theta + math.pi/6 + k*math.pi/3),
                             math.sin(theta + math.pi/6 + k*math.pi/3)] for k in range(6)])
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        # apothem = s * sqrt(3)/2, so s = 2 * apothem / sqrt(3)
        return max_proj * 2.0 / SQRT3

    def sat_gap(p1, p2):
        """Minimum separation between two convex hexagons. Negative if overlapping."""
        min_gap = float('inf')
        all_axes = []
        for poly in (p1, p2):
            for i in range(6):
                edge = poly[(i+1)%6] - poly[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    all_axes.append(n / norm)
        for axis in all_axes:
            proj1 = p1 @ axis
            proj2 = p2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        return min_gap

    def smooth_penalty(polys, beta=50.0):
        """Smooth approximation of total overlap penalty."""
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    # Smooth: use softplus-like
                    pen += gap * gap
        return pen

    def is_valid(positions, angles, tol=1e-6):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective(params):
        if time.time() > deadline - 3:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:2*N+N]
        outer_angle = params[2*N+N]

        side = outer_side_length(positions, angles, outer_angle)

        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        pen = smooth_penalty(polys)

        return side + 50.0 * pen

    def generate_guesses():
        guesses = []

        # 1. Honeycomb cluster (compact)
        pts = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * SQRT3 / 2.0 + j * SQRT3
                pts.append((x, y))
        pts.sort(key=lambda p: p[0]**2 + p[1]**2)
        pos = np.array(pts[:12])
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        outer_a = 0.0
        guesses.append(np.concatenate([pos.flatten(), ang, [outer_a]]))

        # 2. Ring of 6 + inner 6
        ring_pos = []
        for k in range(6):
            a = math.radians(60 * k)
            ring_pos.append([2.6 * math.cos(a), 2.6 * math.sin(a)])
        inner_pos = []
        for k in range(6):
            a = math.radians(60 * k + 30)
            inner_pos.append([1.3 * math.cos(a), 1.3 * math.sin(a)])
        pos = np.array(ring_pos + inner_pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang, [0.0]]))

        # 3. Linear chain
        pos = np.array([[i * 1.5, 0] for i in range(12)])
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang, [0.0]]))

        # 4. 2x6 grid
        pos = []
        for r in range(2):
            for c in range(6):
                x = c * 1.5
                y = r * SQRT3
                if r % 2 == 1:
                    x += 0.75
                pos.append([x, y])
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang, [0.0]]))

        # 5. Rotated honeycomb variants
        for rot_deg in [15, 30, 45]:
            pts = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    x = i * 1.5
                    y = i * SQRT3 / 2.0 + j * SQRT3
                    pts.append((x, y))
            pts.sort(key=lambda p: p[0]**2 + p[1]**2)
            pos = np.array(pts[:12])
            pos -= np.mean(pos, axis=0)
            # Rotate positions
            a = math.radians(rot_deg)
            c, s = math.cos(a), math.sin(a)
            rot_mat = np.array([[c, -s], [s, c]])
            pos = pos @ rot_mat.T
            ang = np.full(N, rot_deg)
            guesses.append(np.concatenate([pos.flatten(), ang, [rot_deg]]))

        # 6. Random scatter
        np.random.seed(42)
        for _ in range(3):
            pos = np.random.uniform(-3, 3, (N, 2))
            ang = np.random.uniform(0, 360, N)
            guesses.append(np.concatenate([pos.flatten(), ang, [np.random.uniform(0, 60)]]))

        return guesses

    best_score = float('inf')
    best_params = None

    guesses = generate_guesses()

    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < 100.0 and res.fun < best_score:
                final_pos = res.x[:2*N].reshape(N, 2)
                final_ang = res.x[2*N:2*N+N]
                final_outer_a = res.x[2*N+N]
                if is_valid(final_pos, final_ang):
                    true_side = outer_side_length(final_pos, final_ang, final_outer_a)
                    if true_side < best_score:
                        best_score = true_side
                        best_params = res.x
        except Exception:
            pass

    # Final polish with SLSQP if we have a good result
    if best_params is not None and time.time() < deadline - 20:
        try:
            def objective_tight(params):
                if time.time() > deadline - 5:
                    return 1e6
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:2*N+N]
                outer_angle = params[2*N+N]
                side = outer_side_length(positions, angles, outer_angle)
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                pen = smooth_penalty(polys, beta=100.0)
                return side + 100.0 * pen

            res2 = minimize(
                objective_tight,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < 100.0:
                final_pos = res2.x[:2*N].reshape(N, 2)
                final_ang = res2.x[2*N:2*N+N]
                final_outer_a = res2.x[2*N+N]
                if is_valid(final_pos, final_ang):
                    true_side = outer_side_length(final_pos, final_ang, final_outer_a)
                    if true_side < best_score:
                        best_score = true_side
                        best_params = res2.x
        except Exception:
            pass

    # Fallback
    if best_params is None:
        final_pos = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_ang = np.zeros(N)
        final_outer_a = 0.0
        best_score = outer_side_length(final_pos, final_ang, final_outer_a)
    else:
        final_pos = best_params[:2*N].reshape(N, 2)
        final_ang = best_params[2*N:2*N+N]
        final_outer_a = best_params[2*N+N]

    inner = []
    for i in range(N):
        inner.append([float(final_pos[i][0]), float(final_pos[i][1]), float(final_ang[i])])

    return inner, (0.0, 0.0), float(best_score), float(final_outer_a)
