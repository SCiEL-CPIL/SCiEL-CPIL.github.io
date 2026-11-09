"""Circle packing in perimeter-4 rectangle: Force-relaxation + SLSQP polishing."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_score = 0.0
    best_result = None

    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_score, best_result
        if len(positions) != N or len(radii) != N:
            return
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            return
        if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
            return
        if np.any(radii < 0):
            return
        # Check boundaries
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-9:
                return
            if positions[i, 0] + radii[i] > w + 1e-9:
                return
            if positions[i, 1] - radii[i] < -1e-9:
                return
            if positions[i, 1] + radii[i] > h + 1e-9:
                return
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()

    def force_relaxation(positions, radii, w, h, steps=200, lr_pos=0.05, lr_rad=0.008):
        """
        Force-directed relaxation to maximize sum of radii.
        1. Try to grow radii.
        2. Push apart overlapping circles.
        3. Push circles inside boundaries.
        """
        pos = positions.astype(float).copy()
        rad = radii.astype(float).copy()

        # Ensure initial validity (shrink if needed)
        for _ in range(20):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.9
                    valid = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.95
                        rad[j] *= 0.95
                        valid = False
            if valid:
                break

        for step in range(steps):
            if time.time() > deadline - 5.0:
                break

            # 1. Attempt to grow radii
            rad += lr_rad

            # 2. Resolve Boundary Violations
            for i in range(N):
                if pos[i, 0] - rad[i] < 0:
                    pos[i, 0] = rad[i]
                if pos[i, 0] + rad[i] > w:
                    pos[i, 0] = w - rad[i]
                if pos[i, 1] - rad[i] < 0:
                    pos[i, 1] = rad[i]
                if pos[i, 1] + rad[i] > h:
                    pos[i, 1] = h - rad[i]

                # If still invalid, shrink
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95

            # 3. Resolve Overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = rad[i] + rad[j]

                    if dist < min_dist:
                        if dist < 1e-6:
                            dx = np.random.uniform(-1, 1)
                            dy = np.random.uniform(-1, 1)
                            dist = math.sqrt(dx * dx + dy * dy)

                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist

                        push = overlap * 0.5 * lr_pos * 8
                        pos[i, 0] += nx * push
                        pos[i, 1] += ny * push
                        pos[j, 0] -= nx * push
                        pos[j, 1] -= ny * push

                        if overlap > min_dist * 0.15:
                            rad[i] *= 0.99
                            rad[j] *= 0.99

        # Final cleanup
        for _ in range(10):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95
                    valid = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.95
                        rad[j] *= 0.95
                        valid = False
            if valid:
                break

        return pos, rad

    def run_slsqp(x0, y0, r0, w_init, maxiter=300):
        """Polish with SLSQP, treating w as variable with h = 2 - w."""
        h_init = 2.0 - w_init
        if h_init <= 0.1 or w_init <= 0.1:
            return None

        n_var = 3 * N + 1

        def objective(v):
            xs = v[0:3 * N:3]
            ys = v[1:3 * N:3]
            rs = v[2:3 * N:3]
            w = v[3 * N]
            h = 2.0 - w
            if h <= 0.1 or w <= 0.1:
                return 1e10

            penalty = 0.0
            for i in range(N):
                if xs[i] - rs[i] < 0:
                    penalty += 50.0 * (xs[i] - rs[i]) ** 2
                if xs[i] + rs[i] > w:
                    penalty += 50.0 * (xs[i] + rs[i] - w) ** 2
                if ys[i] - rs[i] < 0:
                    penalty += 50.0 * (ys[i] - rs[i]) ** 2
                if ys[i] + rs[i] > h:
                    penalty += 50.0 * (ys[i] + rs[i] - h) ** 2

            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        penalty += 50.0 * (min_d - dist) ** 2

            return -np.sum(rs) + penalty

        def objective_grad(v):
            xs = v[0:3 * N:3]
            ys = v[1:3 * N:3]
            rs = v[2:3 * N:3]
            w = v[3 * N]
            h = 2.0 - w
            if h <= 0.1 or w <= 0.1:
                return np.zeros(n_var)

            grad = np.zeros(n_var)
            # d(-sum(rs))/d(rs[i]) = -1
            for i in range(N):
                grad[2 + 3 * i] = -1.0

            # Penalty gradients
            for i in range(N):
                # Left boundary
                if xs[i] - rs[i] < 0:
                    viol = xs[i] - rs[i]
                    grad[3 * i] += 100.0 * viol
                    grad[2 + 3 * i] -= 100.0 * viol
                # Right boundary
                if xs[i] + rs[i] > w:
                    viol = xs[i] + rs[i] - w
                    grad[3 * i] += 100.0 * viol
                    grad[2 + 3 * i] += 100.0 * viol
                    grad[3 * N] -= 100.0 * viol
                # Bottom boundary
                if ys[i] - rs[i] < 0:
                    viol = ys[i] - rs[i]
                    grad[1 + 3 * i] += 100.0 * viol
                    grad[2 + 3 * i] -= 100.0 * viol
                # Top boundary
                if ys[i] + rs[i] > h:
                    viol = ys[i] + rs[i] - h
                    grad[1 + 3 * i] += 100.0 * viol
                    grad[2 + 3 * i] += 100.0 * viol
                    grad[3 * N] += 100.0 * viol  # d(h)/d(w) = -1

            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-10
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        viol = min_d - dist
                        px = dx / dist
                        py = dy / dist
                        grad[3 * i] += 100.0 * viol * px
                        grad[1 + 3 * i] += 100.0 * viol * py
                        grad[3 * j] -= 100.0 * viol * px
                        grad[1 + 3 * j] -= 100.0 * viol * py
                        grad[2 + 3 * i] += 100.0 * viol * 0.5
                        grad[2 + 3 * j] += 100.0 * viol * 0.5

            return grad

        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))
            bounds.append((0, 2.0))
            bounds.append((0.005, 1.0))
        bounds.append((0.5, 1.5))

        x_init = np.concatenate([x0, y0, r0, [w_init]])

        try:
            res = minimize(
                objective,
                x_init,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-8, 'disp': False}
            )
        except Exception:
            return None

        if not res.success and res.fun > 1e6:
            return None

        xs = res.x[0:3 * N:3]
        ys = res.x[1:3 * N:3]
        rs = res.x[2:3 * N:3]
        w = res.x[3 * N]
        h = 2.0 - w

        if w <= 0.1 or h <= 0.1:
            return None

        # Check validity
        valid = True
        for i in range(N):
            if xs[i] - rs[i] < -1e-8 or xs[i] + rs[i] > w + 1e-8:
                valid = False
                break
            if ys[i] - rs[i] < -1e-8 or ys[i] + rs[i] > h + 1e-8:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                    if dist < rs[i] + rs[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break

        if valid:
            positions = np.column_stack([xs, ys])
            validate_and_update(positions, rs, w, h)
            return positions, rs, w, h

        return None

    # Try multiple aspect ratios
    aspect_ratios = np.linspace(0.6, 1.7, 12)

    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break

        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)

        if w < 0.3 or h < 0.3:
            continue

        # Multiple random initializations
        for init_seed in range(3):
            if time.time() > deadline - 8.0:
                break

            np.random.seed(int(ar * 1000 + init_seed * 100) % 100000)

            # Grid-based initialization with hexagonal offset
            cols = int(np.ceil(math.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))

            positions = np.zeros((N, 2))
            radii = np.zeros(N)

            area_per_circle = w * h / N
            r_avg = math.sqrt(area_per_circle / math.pi)

            for idx in range(N):
                i = idx % cols
                j = idx // cols
                x_offset = (i + 0.5) * w / cols
                if j % 2 == 1:
                    x_offset += 0.4 * w / cols
                y_offset = (j + 0.5) * h / rows

                positions[idx, 0] = x_offset + np.random.uniform(-0.03, 0.03)
                positions[idx, 1] = y_offset + np.random.uniform(-0.03, 0.03)
                radii[idx] = r_avg * (0.85 + 0.3 * np.random.random())

            # Phase 1: Force relaxation
            pos_rel, rad_rel = force_relaxation(positions, radii, w, h, steps=150)
            validate_and_update(pos_rel, rad_rel, w, h)

            # Phase 2: SLSQP polishing
            if time.time() > deadline - 5.0:
                break

            result = run_slsqp(pos_rel[:, 0], pos_rel[:, 1], rad_rel, w, maxiter=200)
            if result is not None:
                # Try a few more SLSQP iterations with slightly different params
                if time.time() > deadline - 3.0:
                    break

    # Fallback: simple grid if nothing worked
    if best_result is None:
        side = int(np.ceil(math.sqrt(N)))
        cells = []
        for j in range(side):
            for i in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
        best_result = np.array(cells[:N])

    return best_result
