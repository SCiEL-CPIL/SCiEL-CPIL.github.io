import math
import random
import time

N = 12
T0 = time.time()
DEADLINE = 110.0

def _hex_local():
    return [(math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))) for i in range(6)]

_LOCAL = _hex_local()
_LOCAL_X = [p[0] for p in _LOCAL]
_LOCAL_Y = [p[1] for p in _LOCAL]

def _time_left():
    return DEADLINE - (time.time() - T0)

def _eval_state(s):
    n = len(s)
    if n < 2:
        return 0.0
    best = 1e18
    for i in range(n):
        xi, yi, ai = s[i]
        ca, sa = math.cos(ai), math.sin(ai)
        for j in range(i + 1, n):
            xj, yj, aj = s[j]
            cb, sb = math.cos(aj), math.sin(aj)
            dx = xi - xj
            dy = yi - yj
            for k in range(6):
                lx1 = _LOCAL_X[k]; ly1 = _LOCAL_Y[k]
                nx1 = ca * lx1 - sa * ly1
                ny1 = sa * lx1 + ca * ly1
                lx2 = _LOCAL_X[k]; ly2 = _LOCAL_Y[k]
                nx2 = cb * lx2 - sb * ly2
                ny2 = sb * lx2 + cb * ly2
                p = dx * nx1 + dy * ny1 - (lx1 * nx2 + ly1 * ny2)
                if p < best:
                    best = p
    return -best

def _improve(s, budget, rng):
    best = list(s)
    bestv = _eval_state(best)
    scale = 2.0
    it = 0
    while _time_left() > 2.0 and it < budget:
        it += 1
        cand = [list(x) for x in best]
        idx = rng.randrange(len(cand))
        dim = rng.randrange(3)
        amp = scale * (1.0 if rng.random() < 0.5 else -1.0)
        if dim == 2:
            amp *= math.radians(1.0)
        cand[idx][dim] += amp
        v = _eval_state(cand)
        if v < bestv:
            best = cand
            bestv = v
            scale = min(scale * 1.2, 8.0)
        else:
            scale = max(scale * 0.85, 0.02)
    return best, bestv

def _side_from_state(s):
    mx = my = 1e18
    for i in range(6):
        lx = _LOCAL_X[i]; ly = _LOCAL_Y[i]
        for (x, y, a) in s:
            ca = math.cos(a); sa = math.sin(a)
            px = ca * lx - sa * ly + x
            py = sa * lx + ca * ly + y
            if px > mx: mx = px
            if px < -mx: mx = -px
            if py > my: my = py
            if py < -my: my = -py
    return max(mx, my)

def construct():
    global T0
    T0 = time.time()
    rng = random.Random(20240517)
    best = None
    bestside = 1e18
    start = [10.0, 8.0, 6.0, 4.0, 3.0]
    while _time_left() > 8.0:
        scale = start.pop(0) if start else 1.0
        s = []
        for _ in range(N):
            a = rng.random() * math.pi
            r = scale * math.sqrt(rng.random())
            s.append([r * math.cos(a), r * math.sin(a), rng.random() * math.pi])
        s, v = _improve(s, 4000, rng)
        side = _side_from_state(s)
        if side < bestside:
            bestside = side
            best = s
    if best is None:
        best = [[0.0, 0.0, 0.0]] * N
    return [[round(x, 9), round(y, 9), round(math.degrees(a) % 360.0, 9)] for (x, y, a) in best], \
           (0.0, 0.0), round(bestside, 9), 0.0
