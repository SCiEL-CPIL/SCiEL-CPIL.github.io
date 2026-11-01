import numpy as np
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin vs 150s wall clock

    def is_valid(xy, r):
        n = N
        if np.any(r < -1e-9):
            return False
        if np.any(xy[:, 0] - r < -1e-9) or np.any(1 - xy[:, 0] - r < -1e-9):
            return False
        if np.any(xy[:, 1] - r < -1e-9) or np.any(1 - xy[:, 1] - r < -1e-9):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - 1e-9):
            return False
        return True

    def score_of(xy, r):
        return float(np.sum(r))

    def build_grid_packing(radii, grid, rng):
        """Place circles in descending radius order into the best valid slot.
        Returns (xy, r) or None if any circle cannot be placed."""
        n = len(radii)
        order = np.argsort(-radii)
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for idx in order:
            rad = radii[idx]
            best_pos = None
            best_val = -np.inf
            for gx, gy in grid:
                if gx < rad - 1e-9 or gy < rad - 1e-9:
                    continue
                if gx > 1 - rad + 1e-9 or gy > 1 - rad + 1e-9:
                    continue
                # check against already placed
                ok = True
                for j in range(n):
                    if j == idx or r[j] <= 0:
                        continue
                    d = np.hypot(gx - xy[j, 0], gy - xy[j, 1])
                    if d < rad + r[j] - 1e-9:
                        ok = False
                        break
                if not ok:
                    continue
                # prefer corners/edges (smaller sum of distances to boundaries)
                val = min(gx, 1 - gx) + min(gy, 1 - gy)
                if val > best_val:
                    best_val = val
                    best_pos = (gx, gy)
            if best_pos is None:
                return None
            xy[idx] = best_pos
            r[idx] = rad
        return xy, r

    def swap_local_search(xy, r, grid, max_passes=8):
        """Try swapping positions of pairs of circles; accept valid improving swaps."""
        n = N
        current = score_of(xy, r)
        for _ in range(max_passes):
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    # swap positions
                    xy[i], xy[j] = xy[j].copy(), xy[i].copy()
                    if is_valid(xy, r):
                        s = score_of(xy, r)
                        if s > current + 1e-12:
                            current = s
                            improved = True
                        else:
                            xy[i], xy[j] = xy[j].copy(), xy[i].copy()
                    else:
                        xy[i], xy[j] = xy[j].copy(), xy[i].copy()
            if not improved:
                break
        return xy, r

    def nudge_local_search(xy, r, grid, max_passes=6):
        """Try moving each circle to a nearby grid slot (valid + improving)."""
        n = N
        current = score_of(xy, r)
        for _ in range(max_passes):
            improved = False
            for i in range(n):
                rad = r[i]
                # candidate slots near current position
                cx, cy = xy[i]
                candidates = []
                for gx, gy in grid:
                    if abs(gx - cx) > 0.15 or abs(gy - cy) > 0.15:
                        continue
                    if gx < rad - 1e-9 or gy < rad - 1e-9:
                        continue
                    if gx > 1 - rad + 1e-9 or gy > 1 - rad + 1e-9:
                        continue
                    candidates.append((gx, gy))
                best_val = -np.inf
                best_pos = None
                for gx, gy in candidates:
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        d = np.hypot(gx - xy[j, 0], gy - xy[j, 1])
                        if d < rad + r[j] - 1e-9:
                            ok = False
                            break
                    if not ok:
                        continue
                    val = min(gx, 1 - gx) + min(gy, 1 - gy)
                    if val > best_val:
                        best_val = val
                        best_pos = (gx, gy)
                if best_pos is not None:
                    old = xy[i].copy()
                    xy[i] = best_pos
                    if is_valid(xy, r):
                        s = score_of(xy, r)
                        if s > current + 1e-12:
                            current = s
                            improved = True
                        else:
                            xy[i] = old
                    else:
                        xy[i] = old
            if not improved:
                break
        return xy, r

    # Grid of candidate centers
    M = 16
    grid = [(i / (M - 1), j / (M - 1)) for i in range(M) for j in range(M)]

    rng = np.random.RandomState(7)

    best_xy = None
    best_r = None
    best_score = -np.inf

    def consider(xy, r):
        nonlocal best_xy, best_r, best_score
        if is_valid(xy, r):
            s = score_of(xy, r)
            if s > best_score:
                best_score = s
                best_xy = xy.copy()
                best_r = r.copy()

    # Several radius templates (sums chosen to be aggressive but placeable)
    templates = [
        # descending with a large leader
        np.array([0.15, 0.14, 0.13, 0.12, 0.12, 0.11, 0.11, 0.10, 0.10, 0.09,
                  0.09, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06,
                  0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04,
                  0.04, 0.04]),
        # more uniform
        np.array([0.085] * 32),
        # two big, many small
        np.array([0.18, 0.16] + [0.07] * 30),
        # geometric decay
        np.array([0.14 * (0.93 ** i) for i in range(32)]),
        # mixed
        np.array([0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08,
                  0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
                  0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
                  0.06, 0.06]),
    ]

    for t_idx, radii in enumerate(templates):
        if time.time() > deadline:
            break
        # shuffle order of assignment indices a couple times for variety
        for attempt in range(3):
            if time.time() > deadline:
                break
            radii_shuffled = radii.copy()
            if attempt > 0:
                rng.shuffle(radii_shuffled)
            built = build_grid_packing(radii_shuffled, grid, rng)
            if built is None:
                continue
            xy, r = built
            # local search phases
            xy, r = swap_local_search(xy, r, grid, max_passes=5)
            xy, r = nudge_local_search(xy, r, grid, max_passes=4)
            consider(xy, r)

    # Fallback: simple equal grid
    if best_xy is None:
        side = 6
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
            r[i] = 0.5 / side - 1e-3
        best_xy = xy
        best_r = r

    # Final safety clamp
    r = np.clip(best_r, 0, 0.5)
    xy = best_xy.copy()
    xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
    xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
    if is_valid(xy, r):
        return np.column_stack([xy, r])

    # Absolute fallback
    side = 6
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
