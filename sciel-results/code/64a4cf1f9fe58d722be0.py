"""Search for 30 integer points in [-30,30]^2 maximizing the largest area multiplicity."""
import time
import random
from collections import Counter

N = 30
LIMIT = 60  # coordinate bound
DEADLINE = time.time() + 50.0  # safety margin before 60s/90s limit

def doubled_areas(pts):
    """Return list of abs doubled areas for all non-collinear triples."""
    n = len(pts)
    out = []
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i+1, n):
            xj, yj = pts[j]
            for k in range(j+1, n):
                xk, yk = pts[k]
                d = abs((xj-xi)*(yk-yi) - (yj-yi)*(xk-xi))
                if d != 0:
                    out.append(d)
    return out

def score(pts):
    """Largest multiplicity of a single positive doubled area."""
    cnt = Counter(doubled_areas(pts))
    return max(cnt.values()) if cnt else 0

def init_counter(pts):
    return Counter(doubled_areas(pts))

def apply_move(cnt, pts, i, new_pt):
    """Update counter after moving point i to new_pt. Returns (new_cnt, new_max)."""
    n = len(pts)
    old = pts[i]
    # remove areas involving i
    for j in range(n):
        if j == i:
            continue
        xj, yj = pts[j]
        for k in range(j+1, n):
            if k == i:
                continue
            xk, yk = pts[k]
            d = abs((xj-old[0])*(yk-old[1]) - (yj-old[1])*(xk-old[0]))
            if d != 0:
                cnt[d] -= 1
                if cnt[d] == 0:
                    del cnt[d]
    # add areas with new point
    for j in range(n):
        if j == i:
            continue
        xj, yj = pts[j]
        for k in range(j+1, n):
            if k == i:
                continue
            xk, yk = pts[k]
            d = abs((xj-new_pt[0])*(yk-new_pt[1]) - (yj-new_pt[1])*(xk-new_pt[0]))
            if d != 0:
                cnt[d] += 1
    return cnt

def apply_swap(cnt, pts, i, j, pi, pj):
    """Update counter after swapping points i and j. Returns new_cnt."""
    n = len(pts)
    # remove all areas involving i or j (union, avoiding double count)
    # Areas: (i,j,k) for k not in {i,j}; (i,k,l) for k,l not in {i,j}; (j,k,l) for k,l not in {i,j}
    # Easier: remove all triples containing i or j
    for a in range(n):
        if a == i or a == j:
            continue
        for b in range(a+1, n):
            if b == i or b == j:
                continue
            for c in range(b+1, n):
                if c == i or c == j:
                    continue
                # triple (a,b,c) - no i or j, skip
                pass
    # Actually simpler: remove all triples that contain i or j
    # Triples containing i: (i, x, y) with x<y, x,y != i
    # Triples containing j: (j, x, y) with x<y, x,y != j
    # But triples containing both i and j: (i, j, x) counted in both
    # Remove: for each pair (u,v) with u<v, u,v != i, compute area(i,u,v) and area(j,u,v)
    # and area(i,j,w) for w != i,j
    # Let's do it by iterating all triples and removing those with i or j
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                if a == i or b == i or c == i or a == j or b == j or c == j:
                    xa, ya = pts[a]
                    xb, yb = pts[b]
                    xc, yc = pts[c]
                    d = abs((xb-xa)*(yc-ya) - (yb-ya)*(xc-xa))
                    if d != 0:
                        cnt[d] -= 1
                        if cnt[d] == 0:
                            del cnt[d]
    # Now swap in pts
    pts[i], pts[j] = pts[j], pts[i]
    # Add back all triples containing i or j
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                if a == i or b == i or c == i or a == j or b == j or c == j:
                    xa, ya = pts[a]
                    xb, yb = pts[b]
                    xc, yc = pts[c]
                    d = abs((xb-xa)*(yc-ya) - (yb-ya)*(xc-xa))
                    if d != 0:
                        cnt[d] += 1
    return cnt

def random_point():
    return (random.randint(-LIMIT, LIMIT), random.randint(-LIMIT, LIMIT))

def distinct_random_points(n, seed):
    rng = random.Random(seed)
    seen = set()
    pts = []
    while len(pts) < n:
        p = (rng.randint(-LIMIT, LIMIT), rng.randint(-LIMIT, LIMIT))
        if p not in seen:
            seen.add(p)
            pts.append(p)
    return pts

def search_one(seed, max_iters, use_swaps=True):
    pts = distinct_random_points(N, seed)
    cnt = init_counter(pts)
    best = max(cnt.values()) if cnt else 0
    best_pts = pts[:]
    cur_max = best
    for it in range(max_iters):
        if time.time() > DEADLINE - 5.0:
            break
        if use_swaps and random.random() < 0.3:
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                continue
            pi, pj = random_point(), random_point()
            old_i, old_j = pts[i], pts[j]
            # tentative: move i to pi, j to pj (swap-like reconfig)
            # To keep counter consistent, do two moves
            cnt = apply_move(cnt, pts, i, pi)
            pts[i] = pi
            cnt = apply_move(cnt, pts, j, pj)
            pts[j] = pj
            new_max = max(cnt.values()) if cnt else 0
            if new_max > cur_max:
                cur_max = new_max
                if new_max > best:
                    best = new_max
                    best_pts = pts[:]
            else:
                # revert
                pts[j] = old_j
                cnt = apply_move(cnt, pts, j, old_j)
                pts[i] = old_i
                cnt = apply_move(cnt, pts, i, old_i)
                cur_max = max(cnt.values()) if cnt else 0
        else:
            i = random.randrange(N)
            new_pt = random_point()
            if new_pt == pts[i]:
                continue
            old = pts[i]
            cnt = apply_move(cnt, pts, i, new_pt)
            pts[i] = new_pt
            new_max = max(cnt.values()) if cnt else 0
            if new_max > cur_max:
                cur_max = new_max
                if new_max > best:
                    best = new_max
                    best_pts = pts[:]
            else:
                pts[i] = old
                cnt = apply_move(cnt, pts, i, old)
                cur_max = max(cnt.values()) if cnt else 0
    return best, best_pts

def construct():
    global DEADLINE
    DEADLINE = time.time() + 50.0
    best = 0
    best_pts = None
    seed = 0
    # initial deterministic-ish start
    while time.time() < DEADLINE - 5.0:
        b, pts = search_one(seed, 200, use_swaps=True)
        if b > best:
            best = b
            best_pts = pts
        seed += 1
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    return best_pts
