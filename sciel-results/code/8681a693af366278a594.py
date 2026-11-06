import random
import time

def _col3(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _valid(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _col3(points[i], points[j], points[k]) and _col3(points[i], points[j], points[l]):
                        return False
    return True

def _score(pts):
    n = len(pts)
    best = 0
    for _ in range(100):
        idx = list(range(n))
        random.shuffle(idx)
        sub = []
        for i in idx:
            ok = True
            for a in range(len(sub)):
                for b in range(a+1, len(sub)):
                    if _col3(pts[i], pts[sub[a]], pts[sub[b]]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sub.append(i)
        if len(sub) > best:
            best = len(sub)
    return best

def construct():
    t0 = time.time()
    dl = t0 + 140
    random.seed(7)
    bp = None
    bs = 999
    for tr in range(500):
        if time.time() > dl:
            break
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        if not _valid(pts):
            continue
        s = _score(pts)
        if s < bs:
            bs = s
            bp = [p[:] for p in pts]
        for _ in range(20):
            if time.time() > dl:
                break
            i = random.randint(0, 19)
            ox, oy = pts[i]
            for _ in range(10):
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if (nx, ny) in used and (nx, ny) != (ox, oy):
                    continue
                pts[i] = [nx, ny]
                if _valid(pts):
                    s2 = _score(pts)
                    if s2 < bs:
                        bs = s2
                        bp = [p[:] for p in pts]
                        used.discard((ox, oy))
                        used.add((nx, ny))
                        break
                pts[i] = [ox, oy]
    if bp is None:
        return [[x, x*x] for x in range(-10, 10)]
    return bp
