"""Fresh design: simulated annealing over 16 integer points, maximizing exactly-4 collinear lines, forbidding 5+ collinear."""
import time
import random
import numpy as np

_START = None

def _line_key(p1, p2):
    """Canonical key for the line through two integer points (p1,p2 as (x,y))."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # normalize direction so it's unique up to sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # line constant: c = dx*y - dy*x  (so that dx*Y - dy*X = c for all points on line)
    c = dx * p1[1] - dy * p1[0]
    return (dx, dy, c)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score(points):
    """
    points: list of 16 (x,y) tuples.
    Returns (score, valid) where score = number of lines with exactly 4 points,
    valid = False if any line has >=5 points or points not distinct.
    """
    # distinctness
    if len(set(points)) != 16:
        return 0, False
    line_counts = {}
    for i in range(16):
        for j in range(i + 1, 16):
            k = _line_key(points[i], points[j])
            if k is None:
                continue
            line_counts[k] = line_counts.get(k, 0) + 1
    # a line with m points on it contributes C(m,2) pairs
    # We need to recover m from pair count: m = (1 + sqrt(1+8*pairs))/2
    score = 0
    for k, pairs in line_counts.items():
        # solve m(m-1)/2 = pairs
        disc = 1 + 8 * pairs
        s = int(_isqrt(disc))
        if s * s != disc:
            # shouldn't happen
            continue
        m = (1 + s) // 2
        if m >= 5:
            return 0, False
        if m == 4:
            score += 1
    return score, True

def _isqrt(n):
    if n < 0:
        return 0
    x = int(n ** 0.5)
    while (x + 1) * (x + 1) <= n:
        x += 1
    while x * x > n:
        x -= 1
    return x

def _init_grid():
    """4x4 grid in [0,3]x[0,3], gives 10 exactly-4 lines, no 5 collinear."""
    return [(x, y) for x in range(4) for y in range(4)]

def construct():
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0  # safety margin before 150s kill

    random.seed(20240501)
    np.random.seed(20240501)

    # start from the known-good 4x4 grid
    best = list(_init_grid())
    best_score, _ = _score(best)

    # also try a few other structured starts
    starts = [best]
    # a 4x4 grid shifted/scaled variants
    for off in range(0, 4):
        cand = [(x + off, y) for x in range(4) for y in range(4)]
        starts.append(cand)
    for cand in starts:
        s, v = _score(cand)
        if v and s > best_score:
            best_score = s
            best = list(cand)

    # simulated annealing
    T = 3.0
    T_min = 0.01
    cooling = 0.9999
    cur = list(best)
    cur_score = best_score
    n_iter = 0

    def _move(points):
        """Return a new candidate by perturbing one point."""
        p = list(points)
        idx = random.randrange(16)
        x, y = p[idx]
        # random new integer coordinate within bounds
        nx = random.randint(-50, 50)
        ny = random.randint(-50, 50)
        p[idx] = (nx, ny)
        return p

    while time.time() < deadline:
        n_iter += 1
        cand = _move(cur)
        c_score, c_valid = _score(cand)
        if not c_valid:
            c_score = -1000  # strong penalty
        if c_score > cur_score:
            cur = cand
            cur_score = c_score
            if cur_score > best_score:
                best_score = cur_score
                best = list(cur)
        else:
            # accept with probability exp((c_score - cur_score)/T)
            delta = c_score - cur_score
            if delta < 0:
                r = random.random()
                # avoid overflow
                if delta / T > -50:
                    if r < 2.718281828459045 ** (delta / T):
                        cur = cand
                        cur_score = c_score
        T = max(T_min, T * cooling)

    # final local refinement: try moving each point to all nearby integer coords
    # (cheap hill-climb) with deadline
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            x, y = best[idx]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    cand = list(best)
                    cand[idx] = (nx, ny)
                    s, v = _score(cand)
                    if v and s > best_score:
                        best_score = s
                        best = cand
                        improved = True
        # occasionally also try a random jump to escape
        if time.time() < deadline and random.random() < 0.1:
            cand = _move(best)
            s, v = _score(cand)
            if v and s > best_score:
                best_score = s
                best = cand
                improved = True

    # convert to list of [x, y]
    return [[int(x), int(y)] for (x, y) in best]
