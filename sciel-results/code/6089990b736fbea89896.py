"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter

# ---------- Line geometry ----------
def _canonical_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_pair(p1, p2):
    """Return (A,B,C) with gcd(|A|,|B|)=1 (C may not be coprime) and fixed sign."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = math.gcd(abs(A), abs(B))
    if g:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

# ---------- Scoring (fast, vectorized-ish) ----------
def _score(points):
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    line_counts = Counter()
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            line_counts[_canonical_line((xi, yi), (xs[j], ys[j]))] += 1
    score = 0
    for count in line_counts.values():
        if count >= 10:  # 5 collinear -> invalid
            return -1
        if count == 6:   # 4 collinear
            score += 1
    return score

def _score_fast(points):
    """Same as _score but avoids tuple construction."""
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    line_counts = Counter()
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            xj, yj = xs[j], ys[j]
            A = yi - yj
            B = xj - xi
            C = xi * yj - xj * yi
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g:
                A //= g; B //= g; C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            line_counts[(A, B, C)] += 1
    score = 0
    for count in line_counts.values():
        if count >= 10:
            return -1
        if count == 6:
            score += 1
    return score

# ---------- Validation ----------
def _validate(points):
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return _score(points) >= 0

# ---------- Starting configurations ----------
def _grid_config(base=4, ox=0, oy=0):
    return [[ox + x, oy + y] for x in range(base) for y in range(base)]

def _random_valid_config(max_tries=4000):
    pts = []
    for _ in range(max_tries):
        p = [random.randint(-40, 40), random.randint(-40, 40)]
        if p not in pts:
            pts.append(p)
            if len(pts) == 16:
                if _score(pts) >= 0:
                    return pts
                else:
                    pts = []
    return _grid_config(4)

# ---------- Proposals ----------
def _propose_move(points, idx):
    """Return a candidate new point for index idx."""
    ox, oy = points[idx]
    r = random.random()
    if r < 0.45:
        # local jitter
        nx = ox + random.randint(-4, 4)
        ny = oy + random.randint(-4, 4)
    elif r < 0.75:
        # medium jump
        nx = random.randint(-60, 60)
        ny = random.randint(-60, 60)
    else:
        # line-targeted: pick a random pair of OTHER points and place near their line
        others = [k for k in range(16) if k != idx]
        a, b = random.sample(others, 2)
        A, B, C = _line_from_pair(points[a], points[b])
        # point = (t*B, (A*t + C)/B) ... choose integer t near ox
        # Solve A*x + B*y + C = 0. Parametrize by x: y = -(A*x + C)/B if B!=0
        if B != 0:
            x0 = ox if abs(ox) <= 1000 else random.randint(-100, 100)
            for t in range(0, 6):
                x = x0 + random.randint(-2, 2)
                num = -(A * x + C)
                if num % B == 0:
                    y = num // B
                    if abs(x) <= 1000 and abs(y) <= 1000:
                        nx, ny = x, y
                        break
            else:
                nx = ox + random.randint(-5, 5)
                ny = oy + random.randint(-5, 5)
        else:
            # A*x + C = 0 -> x = -C/A
            if A != 0 and (-C) % A == 0:
                nx = -C // A
                ny = random.randint(-60, 60)
            else:
                nx = ox + random.randint(-5, 5)
                ny = oy + random.randint(-5, 5)
    nx = max(-1000, min(1000, nx))
    ny = max(-1000, min(1000, ny))
    return [nx, ny]

def _is_distinct(points, idx, newp):
    for i, p in enumerate(points):
        if i != idx and p[0] == newp[0] and p[1] == newp[1]:
            return False
    return True

# ---------- Simulated Annealing ----------
def _sa_run(points, deadline):
    current = [p[:] for p in points]
    current_score = _score_fast(current)
    if current_score < 0:
        current = _grid_config(4)
        current_score = _score_fast(current)
    best = [p[:] for p in current]
    best_score = current_score

    T = 8.0
    cooling = 0.9995
    min_T = 0.05

    while time.time() < deadline:
        if T < min_T:
            T = 8.0
            # reheat from best with small perturb
            current = [p[:] for p in best]
            for _ in range(3):
                idx = random.randint(0, 15)
                current[idx][0] += random.randint(-3, 3)
                current[idx][1] += random.randint(-3, 3)
                current[idx][0] = max(-1000, min(1000, current[idx][0]))
                current[idx][1] = max(-1000, min(1000, current[idx][1]))
            current_score = _score_fast(current)
            if current_score < 0:
                current_score = -100

        if random.random() < 0.6:
            # single-point move
            idx = random.randint(0, 15)
            old = current[idx][:]
            newp = _propose_move(current, idx)
            if newp == old or not _is_distinct(current, idx, newp):
                continue
            current[idx] = newp
            ns = _score_fast(current)
            if ns < 0:
                current[idx] = old
                continue
            delta = ns - current_score
            if delta > 0:
                current_score = ns
                if ns > best_score:
                    best_score = ns
                    best = [p[:] for p in current]
            else:
                if random.random() < math.exp(delta / T):
                    current_score = ns
                else:
                    current[idx] = old
        else:
            # 2-point swap
            i1, i2 = random.sample(range(16), 2)
            p1, p2 = current[i1][:], current[i2][:]
            current[i1], current[i2] = p2, p1
            ns = _score_fast(current)
            if ns < 0:
                current[i1], current[i2] = p1, p2
                continue
            delta = ns - current_score
            if delta > 0:
                current_score = ns
                if ns > best_score:
                    best_score = ns
                    best = [p[:] for p in current]
            else:
                if random.random() < math.exp(delta / T):
                    current_score = ns
                else:
                    current[i1], current[i2] = p1, p2
        T *= cooling
    return best, best_score

# ---------- Local polish (deterministic-ish hill climb) ----------
def _local_polish(points, deadline):
    current = [p[:] for p in points]
    cur = _score_fast(current)
    best = [p[:] for p in current]
    best_score = cur
    if cur < 0:
        return best, best_score
    no_improve = 0
    max_no = 4000
    while time.time() < deadline and no_improve < max_no:
        idx = random.randint(0, 15)
        ox, oy = current[idx]
        found = False
        # try small moves
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if not _is_distinct(current, idx, (nx, ny)):
                    continue
                current[idx] = [nx, ny]
                ns = _score_fast(current)
                if ns >= 0 and ns > cur:
                    cur = ns
                    if ns > best_score:
                        best_score = ns
                        best = [p[:] for p in current]
                    no_improve = 0
                    found = True
                    break
                else:
                    current[idx] = [ox, oy]
            if found:
                break
        if not found:
            no_improve += 1
    return best, best_score

def construct():
    start = time.time()
    deadline = start + 115.0

    # Multi-start SA
    candidates = [
        _grid_config(4, 0, 0),
        _grid_config(4, -2, -2),
        _grid_config(4, 1, 1),
        _random_valid_config(),
        _random_valid_config(),
    ]
    # dedup
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(tuple(p) for p in c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    best_points = unique[0]
    best_score = _score_fast(best_points)

    # budget per start
    n_starts = len(unique)
    per = (deadline - 20.0) / n_starts  # reserve 20s for polish
    for c in unique:
        t0 = time.time()
        b, s = _sa_run(c, min(deadline - 20.0, t0 + per))
        if s > best_score:
            best_score = s
            best_points = b

    # Final polish
    best_points, best_score = _local_polish(best_points, deadline)

    # Safety: ensure valid; fall back to grid if needed
    if _score(best_points) < 0:
        best_points = _grid_config(4, 0, 0)
        best_score = _score(best_points)

    return best_points

# EVOLVE-BLOCK-END
