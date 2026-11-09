import time
import random
from math import gcd

# ---------- fast line-key helpers ----------
def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def _score(points):
    n = len(points)
    lc = {}
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            k = _line_key(pi, points[j])
            if k is None:
                continue
            lc[k] = lc.get(k, 0) + 1
    four = 0
    for cnt in lc.values():
        if cnt >= 5:
            return -1
        if cnt == 4:
            four += 1
    return four

# ---------- seeds ----------
def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _seed_3x3_far():
    # 3x3 grid plus a far point; the far point is placed to create many 4-lines
    pts = [[x, y] for x in range(3) for y in range(3)]
    # candidate far points: search small set of far locations
    best = None
    best_s = -1
    for fx in [20, 21, 22, 23, 24, 25, 30, 31, 33, 37, 40, 41, 43, 47, 50, 53, 57, 61, 67, 71, 73, 79, 83, 97, 100, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]:
        for fy in [0, 1, 2, 3, 5, 7, 10, 11, 13, 17, 20, 21, 23, 29, 30, 31, 37, 40, 41, 43, 47, 50, 53, 57, 61, 67, 70, 71, 73, 79, 80, 83, 89, 90, 97, 100, 101, 103, 107, 109, 113, 120, 121, 127, 130, 131, 137, 139, 149, 150, 151, 157, 160, 163, 167, 169, 173, 179, 180, 181, 191, 193, 197, 199, 200, 211, 220, 223, 227, 229, 233, 239, 240, 241, 251, 257, 260, 263, 269, 270, 271, 277, 280, 281, 283, 293, 300, 307, 311, 313, 317, 320, 331, 337, 340, 347, 349, 350, 353, 359, 360, 367, 373, 379, 380, 383, 389, 390, 397, 400, 401, 409, 410, 419, 421, 430, 431, 433, 439, 440, 443, 449, 450, 457, 461, 463, 467, 470, 479, 480, 487, 491, 499, 500, 503, 509, 520, 521, 523, 540, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]:
            cand = pts + [[fx, fy]]
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
    return best

def _seed_staggered():
    # staggered 4x4 to break some 4-collinear but keep others
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x, y + (x % 2)])
    return pts

def _seed_hexagonal():
    pts = []
    for i in range(4):
        for j in range(4):
            x = i + (j % 2)
            y = j * 2
            pts.append([x, y])
    return pts

def _seed_lattice():
    # 4x4 with a shear
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts

def _seed_affine():
    # random affine-ish structured
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([2*x + y, x + 2*y])
    return pts

# ---------- local search with fast delta ----------
def _local_search(points, deadline, max_iter=20000, step=3):
    cur = [p[:] for p in points]
    cur_s = _score(cur)
    best = [p[:] for p in cur]
    best_s = cur_s
    n = len(cur)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        idx = random.randrange(n)
        ox, oy = cur[idx]
        nx = ox + random.randint(-step, step)
        ny = oy + random.randint(-step, step)
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        np_ = [nx, ny]
        # collision
        col = False
        for i in range(n):
            if i != idx and cur[i] == np_:
                col = True
                break
        if col:
            continue
        cur[idx] = np_
        ns = _score(cur)
        if ns < 0:
            cur[idx] = [ox, oy]
            continue
        if ns > cur_s:
            cur_s = ns
            if ns > best_s:
                best_s = ns
                best = [p[:] for p in cur]
        elif ns == cur_s and random.random() < 0.25:
            pass  # keep
        else:
            cur[idx] = [ox, oy]
    return best, best_s

def _hill_climb_pairs(points, deadline, max_iter=15000):
    # try moving each point to a few structured candidate spots
    cur = [p[:] for p in points]
    cur_s = _score(cur)
    best = [p[:] for p in cur]
    best_s = cur_s
    n = len(cur)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        idx = random.randrange(n)
        ox, oy = cur[idx]
        # candidate: near other points to create collinearities
        k = random.randrange(n)
        if k == idx:
            continue
        px, py = cur[k]
        # small offsets around a random other point
        np_ = [px + random.randint(-2, 2), py + random.randint(-2, 2)]
        if abs(np_[0]) > 1000 or abs(np_[1]) > 1000:
            continue
        col = False
        for i in range(n):
            if i != idx and cur[i] == np_:
                col = True
                break
        if col:
            continue
        cur[idx] = np_
        ns = _score(cur)
        if ns < 0:
            cur[idx] = [ox, oy]
            continue
        if ns > cur_s:
            cur_s = ns
            if ns > best_s:
                best_s = ns
                best = [p[:] for p in cur]
        elif ns == cur_s and random.random() < 0.2:
            pass
        else:
            cur[idx] = [ox, oy]
    return best, best_s

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    seeds = [
        _seed_grid(),
        _seed_3x3_far(),
        _seed_staggered(),
        _seed_hexagonal(),
        _seed_lattice(),
        _seed_affine(),
    ]

    best = None
    best_s = -1
    for s in seeds:
        if time.time() > deadline:
            break
        sc = _score(s)
        if sc > best_s:
            best_s = sc
            best = [p[:] for p in s]

    # local search from each seed
    for s in seeds:
        if time.time() > deadline:
            break
        sc0 = _score(s)
        if sc0 < best_s - 2:
            continue
        b, bs = _local_search(s, deadline, max_iter=8000, step=3)
        if bs > best_s:
            best_s = bs
            best = b

    # pair-based hill climb from best
    if time.time() < deadline:
        b, bs = _hill_climb_pairs(best, deadline, max_iter=12000)
        if bs > best_s:
            best_s = bs
            best = b

    # final polish
    if time.time() < deadline:
        b, bs = _local_search(best, deadline, max_iter=15000, step=2)
        if bs > best_s:
            best_s = bs
            best = b

    # validate
    fs = _score(best)
    if fs < 0:
        best = _seed_grid()
    return best
