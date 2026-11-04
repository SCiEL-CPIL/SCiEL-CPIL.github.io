"""Circle packing in perimeter-4 rectangle: force-based repulsion optimizer."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def _pack_force_based(start_time, time_limit=110.0):
    """Pack N circles via iterative repulsion/attraction with wall constraints.

    Strategy:
      - Parametrize rectangle by width w, height h with 2(w+h)=4 => h = 2 - w.
      - Start with a hexagonal-ish seed layout.
      - Iteratively: push overlapping circles apart, pull toward center,
        clamp to walls, and adjust w to balance aspect ratio.
      - The "radius" of each circle is its assigned value; we maximize sum(r_i)
        subject to non-overlap and wall constraints.
    """
    best_sum = 0.0
    best_result = None

    # Multiple random restarts with different initial aspect ratios
    np.random.seed(42)

    for restart in range(30):
        # Vary initial aspect ratio
        w = 1.0 + 0.3 * np.random.randn()
        w = np.clip(w, 0.5, 1.5)
        h = 2.0 - w

        # Initialize circles in a rough hexagonal pattern
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        cell_w = w / cols
        cell_h = h / rows

        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)

        for i in range(N):
            c = i % cols
            r = i // cols
            xs[i] = (c + 0.5) * cell_w
            ys[i] = (r + 0.5) * cell_h
            # Initial radius estimate
            rs[i] = min(cell_w, cell_h) * 0.4

        # Iterative optimization
        n_iter = 3000
        dt = 0.01
        for it in range(n_iter):
            if time.time() - start_time > time_limit:
                break

            # Compute pairwise forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)

            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rs[i] + rs[j]
                    if dist < 1e-12:
                        dx = 0.01 * np.random.randn()
                        dy = 0.01 * np.random.randn()
                        dist = np.sqrt(dx * dx + dy * dy)
                    if dist < min_dist:
                        # Push apart
                        overlap = min_dist - dist
                        force = overlap * 2.0
                        fx[i] += (dx / dist) * force
                        fy[i] += (dy / dist) * force
                        fx[j] -= (dx / dist) * force
                        fy[j] -= (dy / dist) * force
                        # Try to shrink radii a bit to resolve
                        fr[i] -= overlap * 0.1
                        fr[j] -= overlap * 0.1

            # Wall repulsion
            for i in range(N):
                # Left wall
                if xs[i] < rs[i]:
                    fx[i] += (rs[i] - xs[i]) * 5.0
                    fr[i] -= (rs[i] - xs[i]) * 0.5
                # Right wall
                if xs[i] + rs[i] > w:
                    fx[i] -= (xs[i] + rs[i] - w) * 5.0
                    fr[i] -= (xs[i] + rs[i] - w) * 0.5
                # Bottom wall
                if ys[i] < rs[i]:
                    fy[i] += (rs[i] - ys[i]) * 5.0
                    fr[i] -= (rs[i] - ys[i]) * 0.5
                # Top wall
                if ys[i] + rs[i] > h:
                    fy[i] -= (ys[i] + rs[i] - h) * 5.0
                    fr[i] -= (ys[i] + rs[i] - h) * 0.5

            # Centering attraction (mild)
            for i in range(N):
                fx[i] += (w / 2 - xs[i]) * 0.01
                fy[i] += (h / 2 - ys[i]) * 0.01

            # Radius growth: encourage larger radii
            for i in range(N):
                fr[i] += 0.001

            # Update positions and radii
            for i in range(N):
                xs[i] += fx[i] * dt
                ys[i] += fy[i] * dt
                rs[i] += fr[i] * dt * 0.1

                # Clamp to walls
                xs[i] = np.clip(xs[i], 0, w)
                ys[i] = np.clip(ys[i], 0, h)
                # Radius must fit in box
                max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = np.clip(rs[i], 0.01, max_r)

            # Adjust aspect ratio: if circles are more spread horizontally, widen
            spread_x = xs.max() - xs.min()
            spread_y = ys.max() - ys.min()
            if spread_x > spread_y * 1.2 and w < 1.4:
                w += 0.001
            elif spread_y > spread_x * 1.2 and w > 0.6:
                w -= 0.001
            w = np.clip(w, 0.5, 1.5)
            h = 2.0 - w

        # Validate and compute score
        valid = True
        for i in range(N):
            if xs[i] < -1e-9 or xs[i] + rs[i] > w + 1e-9:
                valid = False
                break
            if ys[i] < -1e-9 or ys[i] + rs[i] > h + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < rs[i] + rs[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break

        if valid:
            total_r = rs.sum()
            if total_r > best_sum:
                best_sum = total_r
                # Normalize to unit perimeter representation
                # Store as (x/w, y/h, r) where the actual rectangle is w x h
                # But we need to output in a consistent frame.
                # The task says "bounding rectangle has perimeter at most 4".
                # We'll scale to fit in a box of perimeter 4.
                # Actually, let's just output x, y, r in the w x h box.
                # The evaluator checks perimeter of bounding rect <= 4.
                # Our box has perimeter 2(w+h) = 2(w + 2-w) = 4. Good.
                best_result = np.column_stack([xs, ys, rs])

    if best_result is not None:
        return best_result

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def construct():
    start = time.time()
    result = _pack_force_based(start, time_limit=110.0)
    return result
# EVOLVE-BLOCK-END
