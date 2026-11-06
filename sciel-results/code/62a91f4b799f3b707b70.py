import time
import random
import math
from itertools import combinations

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points using index-sets per line."""
    n = 16
    line_pts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n = -dx_n, -dy_n
            a, b = dy_n, -dx_n
            c = a * x1 + b * y1
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 1:
                a, b, c = a // g2, b // g2, c // g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            s = line_pts.get(key)
            if s is None:
                line_pts[key] = {i, j}
            else:
                s.add(i)
                s.add(j)

    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _line_key_of(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * x1 + b * y1
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _line_key_from_ab(a, b, pt):
    x, y = pt
    c = a * x + b * y
    g = math.gcd(abs(a), abs(b))
    if g > 1:
        a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _base_config():
    return [[x, y] for x in range(4) for y in range(4)]

def _perturbed_base(rng, depth):
    cur = _base_config()
    for _ in range(depth):
        idx = rng.randint(0, 15)
        nx = cur[idx][0] + rng.randint(-4, 4)
        ny = cur[idx][1] + rng.randint(-4, 4)
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if any(p == [nx, ny] for p in cur):
            continue
        cur[idx] = [nx, ny]
    return cur

def construct():
    start = time.time()
    deadline = start + 135
    rng = random.Random(12345)

    best_points = _base_config()
    best_score, _ = _score_points(best_points)

    def anneal(current, current_score, T0, max_no_improve):
        nonlocal best_points, best_score
        T = T0
        no_improve = 0
        while time.time() < deadline - 8 and no_improve < max_no_improve:
            move_type = rng.random()
            idx = rng.randint(0, 15)
            old = current[idx][:]

            if move_type < 0.35:
                # Line-completion move: try to create a 4-point line.
                # Find a line with exactly 3 of the 16 points, then place a 4th.
                done = False
                # Build line->index-set over current points
                line_pts = {}
                for i in range(16):
                    x1, y1 = current[i]
                    for j in range(i + 1, 16):
                        x2, y2 = current[j]
                        k = _line_key_of(current[i], current[j])
                        if k is None:
                            continue
                        s = line_pts.get(k)
                        if s is None:
                            line_pts[k] = {i, j}
                        else:
                            s.add(i)
                            s.add(j)
                triples = [s for s in line_pts.values() if len(s) == 3]
                if triples:
                    s = triples[rng.randint(0, len(triples) - 1)]
                    pi, pj, pk = list(s)
                    p1, p2 = current[pi], current[pj]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g != 0:
                        ux, uy = dx // g, dy // g
                        if ux < 0 or (ux == 0 and uy < 0):
                            ux, uy = -ux, -uy
                        a, b = uy, -ux
                        # Candidate 4th points on this line at integer offsets
                        cands = []
                        for t in range(-6, 7):
                            if t == 0:
                                continue
                            cx = p1[0] + t * ux
                            cy = p1[1] + t * uy
                            if abs(cx) <= 1000 and abs(cy) <= 1000:
                                cands.append([cx, cy])
                        if cands:
                            new_pt = cands[rng.randint(0, len(cands) - 1)]
                        else:
                            new_pt = [p1[0] + rng.randint(-5, 5), p1[1] + rng.randint(-5, 5)]
                        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                            new_pt = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                        if any(p == new_pt for p in current):
                            new_pt = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                        if any(p == new_pt for p in current):
                            continue
                        current[idx] = new_pt
                        done = True
                if not done:
                    # fall through to a random move
                    nx = old[0] + rng.randint(-5, 5)
                    ny = old[1] + rng.randint(-5, 5)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = [nx, ny]
                    if any(p == new_pt for p in current):
                        continue
                    current[idx] = new_pt
            else:
                # Random perturbation
                if rng.random() < 0.5:
                    nx = old[0] + rng.randint(-5, 5)
                    ny = old[1] + rng.randint(-5, 5)
                else:
                    nx = rng.randint(-1000, 1000)
                    ny = rng.randint(-1000, 1000)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                new_pt = [nx, ny]
                if any(p == new_pt for p in current):
                    continue
                current[idx] = new_pt

            s_new, h5_new = _score_points(current)
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue

            if s_new >= current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
                if s_new > current_score:
                    no_improve = 0
                    if s_new > best_score:
                        best_score = s_new
                        best_points = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = s_new
            else:
                current[idx] = old
                no_improve += 1
            T = max(0.05, T * 0.999)

    for restart in range(200):
        if time.time() > deadline - 20:
            break
        if restart == 0:
            current = _base_config()
        elif rng.random() < 0.6:
            current = [p[:] for p in best_points]
        else:
            current = _perturbed_base(rng, rng.randint(1, 8))
        s, h5 = _score_points(current)
        if h5:
            continue
        # Adaptive time budget per restart: shrink as we approach deadline
        remain = deadline - 8 - time.time()
        budget = min(6.0, max(0.8, remain / max(1, (200 - restart))))
        t0 = time.time()
        # Run anneal for a bounded number of steps or until budget
        T = 0.6
        no_improve = 0
        steps = 0
        while time.time() < t0 + budget and no_improve < 400 and steps < 60000:
            steps += 1
            idx = rng.randint(0, 15)
            old = current[idx][:]
            if rng.random() < 0.35:
                # line completion
                line_pts = {}
                for i in range(16):
                    x1, y1 = current[i]
                    for j in range(i + 1, 16):
                        k = _line_key_of(current[i], current[j])
                        if k is None:
                            continue
                        sset = line_pts.get(k)
                        if sset is None:
                            line_pts[k] = {i, j}
                        else:
                            sset.add(i)
                            sset.add(j)
                triples = [sset for sset in line_pts.values() if len(sset) == 3]
                if triples:
                    sset = triples[rng.randint(0, len(triples) - 1)]
                    pi, pj = list(sset)[:2]
                    p1, p2 = current[pi], current[pj]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g != 0:
                        ux, uy = dx // g, dy // g
                        if ux < 0 or (ux == 0 and uy < 0):
                            ux, uy = -ux, -uy
                        cands = []
                        for t in range(-6, 7):
                            if t == 0:
                                continue
                            cx = p1[0] + t * ux
                            cy = p1[1] + t * uy
                            if abs(cx) <= 1000 and abs(cy) <= 1000:
                                cands.append([cx, cy])
                        if cands:
                            new_pt = cands[rng.randint(0, len(cands) - 1)]
                        else:
                            new_pt = [p1[0] + rng.randint(-5, 5), p1[1] + rng.randint(-5, 5)]
                        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                            new_pt = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                        if any(p == new_pt for p in current):
                            new_pt = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                        if any(p == new_pt for p in current):
                            continue
                        current[idx] = new_pt
                    else:
                        continue
                else:
                    nx = old[0] + rng.randint(-5, 5)
                    ny = old[1] + rng.randint(-5, 5)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = [nx, ny]
                    if any(p == new_pt for p in current):
                        continue
                    current[idx] = new_pt
            else:
                if rng.random() < 0.5:
                    nx = old[0] + rng.randint(-5, 5)
                    ny = old[1] + rng.randint(-5, 5)
                else:
                    nx = rng.randint(-1000, 1000)
                    ny = rng.randint(-1000, 1000)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                new_pt = [nx, ny]
                if any(p == new_pt for p in current):
                    continue
                current[idx] = new_pt

            s_new, h5_new = _score_points(current)
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue
            if s_new >= current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
                if s_new > current_score:
                    no_improve = 0
                    if s_new > best_score:
                        best_score = s_new
                        best_points = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = s_new
            else:
                current[idx] = old
                no_improve += 1
            T = max(0.05, T * 0.999)

    # Final local polish on the best configuration
    cur = [p[:] for p in best_points]
    cs, _ = _score_points(cur)
    for _ in range(3000):
        if time.time() > deadline - 2:
            break
        idx = rng.randint(0, 15)
        old = cur[idx][:]
        nx = old[0] + rng.randint(-2, 2)
        ny = old[1] + rng.randint(-2, 2)
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if any(p == [nx, ny] for p in cur):
            continue
        cur[idx] = [nx, ny]
        s_new, h5_new = _score_points(cur)
        if h5_new or s_new < cs:
            cur[idx] = old
        else:
            cs = s_new
            if cs > best_score:
                best_score = cs
                best_points = [p[:] for p in cur]

    return best_points
