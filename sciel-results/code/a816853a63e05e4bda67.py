import time
import random
import math

_DEADLINE = time.time() + 138.0

def _collinear(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0

def _analyze(pts):
    n = len(pts)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g:
                A //= g; B //= g; C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A; B = -B; C = -C
            key = (A, B, C)
            s = lines.get(key)
            if s is None:
                lines[key] = {i, j}
            else:
                s.add(i); s.add(j)
    triples = 0
    has4 = False
    for key, idxs in lines.items():
        k = len(idxs)
        if k >= 3:
            triples += k*(k-1)*(k-2)//6
            if k >= 4:
                has4 = True
    return triples, has4

def _greedy_3free_size(pts):
    n = len(pts)
    active = list(range(n))
    aset = set(active)
    def triples_of(i):
        c = 0
        for j in aset:
            if j == i:
                continue
            for k in aset:
                if k <= j:
                    continue
                if _collinear(pts[i], pts[j], pts[k]):
                    c += 1
        return c
    while True:
        best_i = None
        best_c = 0
        for i in aset:
            c = triples_of(i)
            if c > best_c:
                best_c = c
                best_i = i
        if best_i is None or best_c == 0:
            break
        aset.remove(best_i)
    return len(aset)

def _distinct(pts):
    s = set()
    for p in pts:
        if (p[0], p[1]) in s:
            return False
        s.add((p[0], p[1]))
    return True

def _inbounds(pts):
    for p in pts:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            return False
    return True

def construct():
    random.seed(20240517)
    N = 20
    def rnd():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    # initial valid config
    pts = []
    seen = set()
    while len(pts) < N:
        p = rnd()
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            pts.append(p)
    for _ in range(500):
        _, h4 = _analyze(pts)
        if not h4:
            break
        i = random.randrange(N)
        pts[i] = rnd()
    if _analyze(pts)[1]:
        return [[x, x*x] for x in range(-10, 10)]

    old_triples, _ = _analyze(pts)
    best = [p[:] for p in pts]
    best_triples = old_triples

    T = 6.0
    while time.time() < _DEADLINE - 25.0:
        i = random.randrange(N)
        if random.random() < 0.35:
            # targeted: put point i on line through two others
            a, b = random.sample(range(N), 2)
            dx = pts[b][0]-pts[a][0]
            dy = pts[b][1]-pts[a][1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            sx, sy = dx//g, dy//g
            t = random.randint(-30, 30)
            nx = pts[a][0] + t*sx
            ny = pts[a][1] + t*sy
            if abs(nx) > 100 or abs(ny) > 100:
                continue
        else:
            axis = random.choice(0, 1)
            d = random.randint(-8, 8)
            if d == 0:
                continue
            nx = pts[i][0] + (d if axis == 0 else 0)
            ny = pts[i][1] + (d if axis == 1 else 0)
            if abs(nx) > 100:
                nx = -nx
            if abs(ny) > 100:
                ny = -ny
        new_pts = [p[:] for p in pts]
        new_pts[i] = [nx, ny]
        if not _distinct(new_pts):
            continue
        new_triples, h4 = _analyze(new_pts)
        if h4:
            continue
        if new_triples >= old_triples or random.random() < math.exp((new_triples-old_triples)/max(T, 1e-6)):
            pts = new_pts
            old_triples = new_triples
            if old_triples > best_triples:
                best_triples = old_triples
                best = [p[:] for p in pts]
        T *= 0.99997
        if T < 0.05:
            T = 0.05

    # refinement: minimize greedy 3-free size
    pts = best
    cur_g = _greedy_3free_size(pts)
    best_g_cfg = [p[:] for p in pts]
    best_g = cur_g
    T2 = 2.0
    while time.time() < _DEADLINE:
        i = random.randrange(N)
        if random.random() < 0.4:
            a, b = random.sample(range(N), 2)
            dx = pts[b][0]-pts[a][0]
            dy = pts[b][1]-pts[a][1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            sx, sy = dx//g, dy//g
            t = random.randint(-30, 30)
            nx = pts[a][0] + t*sx
            ny = pts[a][1] + t*sy
            if abs(nx) > 100 or abs(ny) > 100:
                continue
        else:
            axis = random.choice(0, 1)
            d = random.randint(-6, 6)
            if d == 0:
                continue
            nx = pts[i][0] + (d if axis == 0 else 0)
            ny = pts[i][1] + (d if axis == 1 else 0)
            if abs(nx) > 100:
                nx = -nx
            if abs(ny) > 100:
                ny = -ny
        new_pts = [p[:] for p in pts]
        new_pts[i] = [nx, ny]
        if not _distinct(new_pts):
            continue
        nt, h4 = _analyze(new_pts)
        if h4:
            continue
        ng = _greedy_3free_size(new_pts)
        if ng <= cur_g or random.random() < math.exp((cur_g-ng)/max(T2, 1e-6)):
            pts = new_pts
            cur_g = ng
            if ng < best_g:
                best_g = ng
                best_g_cfg = [p[:] for p in pts]
        T2 *= 0.9999
        if T2 < 0.05:
            T2 = 0.05

    result = best_g_cfg
    # final validation
    if len(result) != N or not _distinct(result) or not _inbounds(result) or _analyze(result)[1]:
        return [[x, x*x] for x in range(-10, 10)]
    return result
