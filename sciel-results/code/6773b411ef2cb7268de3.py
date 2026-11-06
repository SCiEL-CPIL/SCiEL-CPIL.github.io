import time, math, random
from collections import defaultdict

def _line_key(i, j, pts):
    x1, y1 = pts[i]
    x2, y2 = pts[j]
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g:
        dx //= g; dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx; dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _line_score(pts):
    n = len(pts)
    line_pts = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            line_pts[_line_key(i, j, pts)].add(i)
            line_pts[_line_key(i, j, pts)].add(j)
    return sum(1 for s in line_pts.values() if len(s) == 4)

def _fallback():
    # 4x4 grid centered at origin: 16 distinct points, |x|,|y|<=1, no five collinear.
    return [[x, y] for x in range(-2, 2) for y in range(-2, 2)]

def construct():
    start = time.time()
    deadline = start + 132.0  # safety margin under 150s wall-clock

    def in_bounds(p):
        return abs(p[0]) <= 1000 and abs(p[1]) <= 1000

    def valid(pts):
        if len(set(tuple(p) for p in pts)) != 16:
            return False
        return all(in_bounds(p) for p in pts)

    def build_pair(pts):
        pair = {}
        for i in range(16):
            for j in range(i+1, 16):
                pair[(i, j)] = _line_key(i, j, pts)
        return pair

    def score4(pts, pair):
        cnt = defaultdict(int)
        for k, v in pair.items():
            cnt[v] += 1
        return sum(1 for v in cnt.values() if v == 4)

    def delta_move(pts, i, nx, ny, pair):
        # incremental score change for moving point i to (nx, ny)
        xi, yi = pts[i]
        new = [nx, ny]
        others = [j for j in range(16) if j != i]
        old = {}
        for j in others:
            old[j] = pair[(i, j)] if i < j else pair[(j, i)]
        new = {}
        for j in others:
            new[j] = _line_key(i, j, [pts[k] if k != i else [nx, ny] for k in range(16)])
        oldcnt = defaultdict(int)
        for j in others:
            oldcnt[old[j]] += 1
        newcnt = defaultdict(int)
        for j in others:
            newcnt[new[j]] += 1
        gain = 0
        for k in set(oldcnt) | set(newcnt):
            o = oldcnt.get(k, 0)
            nn = newcnt.get(k, 0)
            gain += (1 if nn == 4 else 0) - (1 if o == 4 else 0)
        return gain

    def apply_move(pts, i, nx, ny, pair):
        pts[i] = [nx, ny]
        others = [j for j in range(16) if j != i]
        for j in others:
            k = _line_key(i, j, pts)
            if i < j:
                pair[(i, j)] = k
            else:
                pair[(j, i)] = k

    # ---- multi-start simulated annealing with single-point moves ----
    rng = random.Random(20240517)
    best = _fallback()
    best_score = _line_score(best)

    def run_anneal(init, t0, t1, iters):
        nonlocal best, best_score
        pts = [list(p) for p in init]
        pair = build_pair(pts)
        sc = score4(pts, pair)
        t = t0
        step = (t1 - t0) / max(1, iters)
        for it in range(iters):
            if time.time() >= deadline:
                break
            i = rng.randrange(16)
            x, y = pts[i]
            nx = x + rng.randint(-8, 8)
            ny = y + rng.randint(-8, 8)
            if not in_bounds((nx, ny)):
                continue
            if any(nx == pts[j][0] and ny == pts[j][1] for j in range(16) if j != i):
                continue
            d = delta_move(pts, i, nx, ny, pair)
            if d >= 0 or rng.random() < math.exp(d / max(1e-9, t)):
                apply_move(pts, i, nx, ny, pair)
                sc += d
            if sc > best_score:
                best_score = sc
                best = [list(p) for p in pts]
            t = max(t1, t - step)
        return best_score

    def make_grid():
        cx = rng.randint(-400, 400)
        cy = rng.randint(-400, 400)
        ax = rng.randint(1, 6)
        ay = rng.randint(1, 6)
        pts = []
        for a in range(4):
            for b in range(4):
                pts.append([cx + a*ax, cy + b*ay])
        return pts

    def make_cluster():
        pts = []
        for _ in range(16):
            pts.append([rng.randint(-300, 300), rng.randint(-300, 300)])
        return pts

    inits = [_fallback()]
    for _ in range(6):
        inits.append(make_grid())
    for _ in range(4):
        inits.append(make_cluster())

    for init in inits:
        if time.time() >= deadline:
            break
        iters = 20000
        run_anneal(init, 20.0, 0.3, iters)

    # ---- 2-point swap local search on the best configuration ----
    if time.time() < deadline:
        pts = [list(p) for p in best]
        pair = build_pair(pts)
        sc = score4(pts, pair)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in range(16):
                for b in range(a+1, 16):
                    if time.time() >= deadline:
                        break
                    xa, ya = pts[a]
                    xb, yb = pts[b]
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            nxa, nya = xa+dx, ya+dy
                            nxb, nyb = xb-dx, yb-dy
                            if not (in_bounds((nxa, nya)) and in_bounds((nxb, nyb))):
                                continue
                            if any(nxa == pts[j][0] and nya == pts[j][1] for j in range(16) if j != a):
                                continue
                            if any(nxb == pts[j][0] and nyb == pts[j][1] for j in range(16) if j != b):
                                continue
                            da = delta_move(pts, a, nxa, nya, pair)
                            db = delta_move(pts, b, nxb, nyb, pair)
                            if da + db > 0:
                                apply_move(pts, a, nxa, nya, pair)
                                apply_move(pts, b, nxb, nyb, pair)
                                sc += da + db
                                improved = True
                                if sc > best_score:
                                    best_score = sc
                                    best = [list(p) for p in pts]
            if time.time() >= deadline:
                break

    # ---- final safety: ensure 16 distinct in-bounds points ----
    seen = set()
    out = []
    for p in best:
        t = tuple(p)
        if t not in seen and in_bounds(p):
            seen.add(t)
            out.append([p[0], p[1]])
    # pad if somehow short (should not happen)
    guard = 0
    while len(out) < 16 and guard < 100000:
        guard += 1
        cand = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
        if tuple(cand) not in seen:
            seen.add(tuple(cand))
            out.append(cand)
    return out[:16]
