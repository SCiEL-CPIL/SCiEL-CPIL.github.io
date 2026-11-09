import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    best_val = -np.inf
    best_out = None

    def time_left():
        return deadline - time.time()

    def validate(x, y, r, w, h, tol=1e-5):
        """Check all constraints strictly."""
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
            if r[i] < 0:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        if w + h > 2.0 + 1e-9 or w < 0.1 or h < 0.1:
            return False
        return True

    def make_output(x, y, r, w, h):
        """Scale positions to fit in [0,1]x[0,1] equivalent, return (N,3) array."""
        # The evaluator expects circles in a rectangle of perimeter 4.
        # Return (x, y, r) where the bounding box has perimeter <= 4.
        # We return raw coordinates in the (w, h) rectangle.
        out = np.zeros((N, 3))
        out[:, 0] = x
        out[:, 1] = y
        out[:, 2] = r
        return out

    def physics_simulate(x0, y0, r0, w0, h0, steps=5000, dt=0.002, damping=0.95):
        """
        Physics-based relaxation:
        - Repulsive forces between overlapping circles
        - Attractive forces toward walls (to maximize radii)
        - Gentle pressure to increase radii
        - Rectangle dimensions also adjusted
        """
        x = x0.copy().astype(np.float64)
        y = y0.copy().astype(np.float64)
        r = r0.copy().astype(np.float64)
        w = w0
        h = h0

        # Initialize velocities
        vx = np.zeros(N)
        vy = np.zeros(N)
        vr = np.zeros(N)
        vw = 0.0
        vh = 0.0

        best_local = -np.inf
        best_local_state = None

        for step in range(steps):
            if time.time() > deadline - 2:
                break

            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            fw = 0.0
            fh = 0.0

            # Inter-circle repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-10:
                        dx = 0.01
                        dy = 0.01
                        d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        # Repulsive force proportional to overlap
                        overlap = min_d - d
                        force = 10.0 * overlap
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
                        # Push radii apart
                        fr[i] -= 0.5 * force
                        fr[j] -= 0.5 * force

            # Wall constraints and attraction
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    fx[i] += 10.0 * (r[i] - x[i])
                else:
                    # Gentle pull toward wall to encourage larger radius
                    fx[i] -= 0.1
                # Right wall
                if x[i] > w - r[i]:
                    fx[i] -= 10.0 * (x[i] - (w - r[i]))
                else:
                    fx[i] += 0.1
                # Bottom wall
                if y[i] < r[i]:
                    fy[i] += 10.0 * (r[i] - y[i])
                else:
                    fy[i] -= 0.1
                # Top wall
                if y[i] > h - r[i]:
                    fy[i] -= 10.0 * (y[i] - (h - r[i]))
                else:
                    fy[i] += 0.1

                # Radius pressure: try to grow radius
                # Limited by nearest wall and neighbors
                max_r_left = x[i]
                max_r_right = w - x[i]
                max_r_bottom = y[i]
                max_r_top = h - y[i]
                max_r_wall = min(max_r_left, max_r_right, max_r_bottom, max_r_top)

                # Check neighbor constraints
                max_r_neigh = max_r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-10:
                        max_r_neigh = min(max_r_neigh, (d - r[j]) / 2.0)
                max_r_neigh = max(0, max_r_neigh)

                target_r = min(max_r_wall, max_r_neigh)
                if r[i] < target_r:
                    fr[i] += 0.5 * (target_r - r[i])
                else:
                    fr[i] -= 0.5 * (r[i] - target_r)

            # Rectangle dimension forces: try to expand w and h
            # But constrained by w + h = 2
            # Use a Lagrange-like approach: if circles are near walls, expand
            # For now, keep w and h fixed in physics sim, optimize separately

            # Update velocities with damping
            vx = damping * vx + fx * dt
            vy = damping * vy + fy * dt
            vr = damping * vr + fr * dt

            # Update positions
            x += vx * dt
            y += vy * dt
            r += vr * dt

            # Enforce hard constraints
            for i in range(N):
                r[i] = max(0.001, r[i])
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

            # Check objective
            val = np.sum(r)
            if val > best_local:
                best_local = val
                best_local_state = (x.copy(), y.copy(), r.copy(), w, h)

            # Adaptive dt: reduce if energy is high
            energy = np.sum(fx * vx) + np.sum(fy * vy) + np.sum(fr * vr)
            if energy > 1.0:
                dt *= 0.9
            elif energy < 0.01:
                dt = min(dt * 1.05, 0.005)

        return best_local_state

    def grid_init(w, h, n_cols, n_rows):
        """Create a grid initialization."""
        xs = []
        ys = []
        rs = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(xs) >= N:
                    break
                x = (j + 0.5) * w / n_cols
                y = (i + 0.5) * h / n_rows
                r = min(w / n_cols, h / n_rows) / 2 * 0.8
                xs.append(x)
                ys.append(y)
                rs.append(r)
        # Pad if needed
        while len(xs) < N:
            xs.append(w / 2)
            ys.append(h / 2)
            rs.append(0.05)
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def random_init(w, h):
        """Random initialization with non-overlapping circles."""
        xs = []
        ys = []
        rs = []
        max_attempts = 1000
        for _ in range(N):
            placed = False
            for _ in range(max_attempts):
                r = np.random.uniform(0.02, min(w, h) / 4)
                x = np.random.uniform(r, w - r)
                y = np.random.uniform(r, h - r)
                # Check overlap with existing circles
                ok = True
                for i in range(len(xs)):
                    d = np.sqrt((x - xs[i])**2 + (y - ys[i])**2)
                    if d < r + rs[i] + 0.01:
                        ok = False
                        break
                if ok:
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
                    placed = True
                    break
            if not placed:
                # Fallback: place at center with small radius
                xs.append(w / 2 + np.random.uniform(-0.1, 0.1))
                ys.append(h / 2 + np.random.uniform(-0.1, 0.1))
                rs.append(0.02)
        return np.array(xs), np.array(ys), np.array(rs)

    # Main loop: try different aspect ratios and initializations
    aspect_ratios = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 0.8, 0.66, 0.5]
    inits_per_aspect = 4

    for aspect in aspect_ratios:
        if time.time() > deadline - 5:
            break

        w = aspect / (1 + aspect) * 2.0
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            continue

        # Try different grid configurations
        grid_configs = [
            (5, 5), (4, 6), (3, 7), (6, 4), (7, 3), (5, 4), (4, 5)
        ]

        for (nc, nr) in grid_configs[:3]:
            if time.time() > deadline - 5:
                break
            x0, y0, r0 = grid_init(w, h, nc, nr)
            result = physics_simulate(x0, y0, r0, w, h, steps=3000)
            if result:
                x, y, r, w_r, h_r = result
                if validate(x, y, r, w_r, h_r):
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_out = make_output(x, y, r, w_r, h_r)

        # Random initializations
        for _ in range(inits_per_aspect):
            if time.time() > deadline - 5:
                break
            x0, y0, r0 = random_init(w, h)
            result = physics_simulate(x0, y0, r0, w, h, steps=3000)
            if result:
                x, y, r, w_r, h_r = result
                if validate(x, y, r, w_r, h_r):
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_out = make_output(x, y, r, w_r, h_r)

    # If we have a good result, try to improve it with a final refinement pass
    if best_out is not None and time.time() < deadline - 10:
        x = best_out[:, 0].copy()
        y = best_out[:, 1].copy()
        r = best_out[:, 2].copy()
        # Estimate w, h from bounding box
        w_est = np.max(x) + np.max(r)
        h_est = np.max(y) + np.max(r)
        # Scale to perimeter 4
        scale = 4.0 / (2 * (w_est + h_est))
        w_final = w_est * scale
        h_final = h_est * scale
        x = x * scale
        y = y * scale
        r = r * scale

        # Run one more physics simulation to refine
        result = physics_simulate(x, y, r, w_final, h_final, steps=5000, dt=0.001, damping=0.98)
        if result:
            x2, y2, r2, w2, h2 = result
            if validate(x2, y2, r2, w2, h2):
                val = np.sum(r2)
                if val > best_val:
                    best_val = val
                    best_out = make_output(x2, y2, r2, w2, h2)

    # Fallback: simple grid in unit square scaled to perimeter 4
    if best_out is None:
        # 5x5 grid in a square of side 1 (perimeter 4)
        side = 5
        out = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            out[i, 0] = (col + 0.5) / side
            out[i, 1] = (row + 0.5) / side
            out[i, 2] = 0.5 / side - 0.001
        best_out = out

    return best_out
