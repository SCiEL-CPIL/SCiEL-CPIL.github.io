"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.spatial import cKDTree

N = 21
PERIMETER = 4.0
DEADLINE_FRACTION = 0.92  # safety margin vs 120s evaluator / 150s kill
EPS = 1e-9


def _max_radius_at(cx, cy, W, H, circles):
    """Max radius a circle at (cx,cy) can have while staying in [0,W]x[0,H] and
    disjoint from existing circles. circles: list of (x,y,r)."""
    r_max = min(cx, W - cx, cy, H - cy)
    if r_max <= 0:
        return 0.0
    for (px, py, pr) in circles:
        d = np.hypot(cx - px, cy - py)
        if d <= 1e-12:
            return 0.0
        # need r + pr <= d  =>  r <= d - pr
        cand = d - pr
        if cand < r_max:
            r_max = cand
    return max(r_max, 0.0)


def _greedy_pack(W, H, order, n=N, max_samples=4000):
    """Place circles in given order. For each, pick a position that maximizes the
    achievable radius via sampling + local refine. Returns list of (x,y,r) or None
    if a circle cannot be placed with r>0."""
    circles = []
    rng = random.Random(hash((W, H, tuple(order))) & 0xffffffff)
    for idx in order:
        best_r = 0.0
        best_pos = None
        # candidate positions: random samples + center + corners-inset
        samples = [(rng.uniform(0.05 * W, 0.95 * W), rng.uniform(0.05 * H, 0.95 * H))
                   for _ in range(max_samples)]
        samples.append((W / 2.0, H / 2.0))
        # also near existing circle boundaries
        for (px, py, pr) in circles:
            for ang in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi / 4, 3 * np.pi / 4,
                        5 * np.pi / 4, 7 * np.pi / 4):
                nx = px + (pr + 0.5 * pr) * np.cos(ang)
                ny = py + (pr + 0.5 * pr) * np.sin(ang)
                samples.append((nx, ny))
        for (sx, sy) in samples:
            sx = min(max(sx, 0.0), W)
            sy = min(max(sy, 0.0), H)
            r = _max_radius_at(sx, sy, W, H, circles)
            if r > best_r:
                best_r = r
                best_pos = (sx, sy)
        if best_pos is None or best_r <= 1e-9:
            return None
        circles.append((best_pos[0], best_pos[1], best_r))
    return circles


def _refine(circles, W, H, steps=2000, lr0=0.05):
    """Local search: nudge each circle to increase its radius (grow into free space)
    while keeping disjointness and bounds. Uses simple gradient-free push."""
    circles = [list(c) for c in circles]
    n = len(circles)
    rng = random.Random(12345)
    for step in range(steps):
        lr = lr0 * (1.0 - step / steps)
        i = rng.randrange(n)
        x, y, r = circles[i]
        # try to grow: find direction of least resistance
        improved = False
        for _ in range(4):
            dx = rng.uniform(-1, 1)
            dy = rng.uniform(-1, 1)
            norm = np.hypot(dx, dy)
            if norm < 1e-9:
                continue
            dx, dy = dx / norm, dy / norm
            # move a tiny bit and try to increase radius
            nx = x + dx * lr
            ny = y + dy * lr
            nx = min(max(nx, 0.0), W)
            ny = min(max(ny, 0.0), H)
            nr = _max_radius_at(nx, ny, W, H, [c for j, c in enumerate(circles) if j != i])
            if nr > r + 1e-12:
                circles[i] = [nx, ny, nr]
                improved = True
                break
        # also try pure growth in place
        nr = _max_radius_at(x, y, W, H, [c for j, c in enumerate(circles) if j != i])
        if nr > r + 1e-12:
            circles[i][2] = nr
    return circles


def _validity(circles, W, H):
    """Check disjoint interiors and bounds. Returns (ok, perimeter, sum_radii)."""
    n = len(circles)
    if n != N:
        return False, 0.0, 0.0
    # perimeter of bounding rectangle of centers+radius must be <= 4
    # bounding box of the union
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per > PERIMETER + 1e-6:
        return False, per, 0.0
    # disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(circles[i][0] - circles[j][0], circles[i][1] - circles[j][1])
            if d < circles[i][2] + circles[j][2] - 1e-7:
                return False, per, 0.0
    s = sum(c[2] for c in circles)
    return True, per, s


def _scale_to_perimeter(circles, target_per=PERIMETER):
    """Scale the whole packing so bounding-box perimeter equals target_per."""
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per <= 0:
        return circles
    s = target_per / per
    out = []
    for (x, y, r) in circles:
        nx = (x - minx) * s
        ny = (y - miny) * s
        nr = r * s
        out.append((nx, ny, nr))
    return out


def construct():
    t0 = time.time()
    deadline = 120.0 * DEADLINE_FRACTION
    best = None
    best_val = -1.0

    # aspect ratio candidates: W + H = 2 (since perimeter 4 => W+H=2)
    # W in (0,2). Try a spread.
    aspect_choices = []
    for k in range(1, 21):
        W = 2.0 * k / 21.0
        aspect_choices.append(W)
    # include near-square
    aspect_choices.append(1.0)
    random.shuffle(aspect_choices)

    orders = [list(range(N)) for _ in range(1)]
    # a few random orders
    for _ in range(7):
        o = list(range(N))
        random.shuffle(o)
        orders.append(o)

    attempts = 0
    for W in aspect_choices:
        H = 2.0 - W
        for order in orders:
            if time.time() - t0 > deadline:
                break
            attempts += 1
            circles = _greedy_pack(W, H, order, max_samples=2500)
            if circles is None:
                continue
            circles = _refine(circles, W, H, steps=1500, lr0=0.03)
            # scale to exact perimeter 4
            circles = _scale_to_perimeter(circles, PERIMETER)
            ok, per, s = _validity(circles, 2.0, 2.0)
            if ok and s > best_val:
                best_val = s
                best = circles
            if time.time() - t0 > deadline:
                break
        if time.time() - t0 > deadline:
            break

    if best is None:
        # fallback to seed-like grid (valid, score 1.0)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = cells[:N]
        # scale to perimeter 4
        best = _scale_to_perimeter(best, PERIMETER)

    return np.array(best)


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
