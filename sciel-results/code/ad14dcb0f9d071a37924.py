import time
import math
import random
from collections import defaultdict

def _gcd3(a, b, c):
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        for j in range(i + 1, n):
            x2, y2 = points[j][0], points[j][1]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            g = _gcd3(abs(a), abs(b), abs(c))
            if g:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            lines[(a, b, c)].update((i, j))
    return sum(1 for s in lines.values() if len(s) == 2)

def _is_valid(points):
    if len(points) != 12:
        return False
    if len(set(map(tuple, points))) < 12:
        return False
    if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in points):
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    for k in range(3, 12):
        xi, yi = points[k]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _build_pencil_4x3():
    # 4 lines through origin, 3 points each (not including origin)
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    pts = []
    for d in range(1, 4):
        for (ux, uy) in dirs:
            pts.append([d * ux, d * uy])
    return pts

def _build_grid_3x4():
    return [[x, y] for x in range(3) for y in range(4)]

def _build_random():
    rng = random.Random(42)
    pts = []
    while len(pts) < 12:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if [x, y] not in pts:
            pts.append([x, y])
    return pts

def construct():
    start = time.time()
    deadline = start + 140
    
    best_points = None
    best_score = float('inf')
    
    def consider(pts):
        nonlocal best_points, best_score
        if not _is_valid(pts):
            return
        s = count_ordinary_lines(pts)
        if s < best_score:
            best_score = s
            best_points = [[int(x), int(y)] for x, y in pts]

    # 1) Try diverse initial configurations
    initial_configs = [
        _build_pencil_4x3(),
        _build_grid_3x4(),
        _build_random()
    ]
    
    for cfg in initial_configs:
        consider(cfg)
        
    if best_score <= 6:
        return best_points

    # 2) Build lattice for constraint-based search
    L = 20
    lattice = [(x, y) for x in range(L) for y in range(L)]
    lattice_set = set(lattice)
    n_lat = len(lattice)
    
    # Precompute collinear triples in the lattice
    # For each pair (i,j), find all k such that i,j,k are collinear
    # Store as: for each pair (i,j), the list of indices k (k != i, k != j) on the same line
    pair_collinear = {}
    for i in range(n_lat):
        xi, yi = lattice[i]
        for j in range(i + 1, n_lat):
            xj, yj = lattice[j]
            dx = xj - xi
            dy = yj - yi
            # Check all other points
            collinear_k = []
            for k in range(n_lat):
                if k == i or k == j:
                    continue
                xk, yk = lattice[k]
                if dx * (yk - yi) == dy * (xk - xi):
                    collinear_k.append(k)
            pair_collinear[(i, j)] = collinear_k
    
    # 3) Local search with pair-swap moves on lattice indices
    rng = random.Random(7)
    
    # Start from best found, map to lattice if possible, else start fresh
    cur = [p[:] for p in best_points]
    cur_score = best_score
    
    # Try to map current points to lattice
    cur_lattice_idx = []
    valid_map = True
    for p in cur:
        found = False
        for li, lp in enumerate(lattice):
            if lp[0] == p[0] and lp[1] == p[1]:
                cur_lattice_idx.append(li)
                found = True
                break
        if not found:
            valid_map = False
            break
    
    if not valid_map or len(set(cur_lattice_idx)) != 12:
        # Start with a random 12-point subset of the lattice
        cur_lattice_idx = rng.sample(range(n_lat), 12)
        cur = [list(lattice[i]) for i in cur_lattice_idx]
        cur_score = count_ordinary_lines(cur)
        if cur_score < best_score:
            best_score = cur_score
            best_points = [p[:] for p in cur]
    
    def try_swap(i, j):
        """Swap point at position i with lattice point at index j."""
        nonlocal cur, cur_score, best_points, best_score, cur_lattice_idx
        if j in cur_lattice_idx:
            return
        old_idx = cur_lattice_idx[i]
        new_pts = [p[:] for p in cur]
        new_pts[i] = list(lattice[j])
        new_lattice_idx = cur_lattice_idx[:]
        new_lattice_idx[i] = j
        
        if not _is_valid(new_pts):
            return
        
        s = count_ordinary_lines(new_pts)
        if s < cur_score:
            cur = new_pts
            cur_lattice_idx = new_lattice_idx
            cur_score = s
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in cur]
    
    def try_double_swap(i, j, ni, nj):
        """Swap point at position i with lattice ni, and point at position j with lattice nj."""
        nonlocal cur, cur_score, best_points, best_score, cur_lattice_idx
        if ni in cur_lattice_idx or nj in cur_lattice_idx:
            return
        if ni == nj:
            return
        old_i = cur_lattice_idx[i]
        old_j = cur_lattice_idx[j]
        if old_i == ni or old_i == nj or old_j == ni or old_j == nj:
            return
        new_pts = [p[:] for p in cur]
        new_pts[i] = list(lattice[ni])
        new_pts[j] = list(lattice[nj])
        new_lattice_idx = cur_lattice_idx[:]
        new_lattice_idx[i] = ni
        new_lattice_idx[j] = nj
        
        if not _is_valid(new_pts):
            return
        
        s = count_ordinary_lines(new_pts)
        if s < cur_score:
            cur = new_pts
            cur_lattice_idx = new_lattice_idx
            cur_score = s
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in cur]
    
    iters = 0
    while time.time() < deadline and iters < 300000:
        iters += 1
        r = rng.random()
        
        if r < 0.6:
            # Single swap: pick a point position and a random lattice point
            i = rng.randint(0, 11)
            j = rng.randint(0, n_lat - 1)
            try_swap(i, j)
        elif r < 0.85:
            # Double swap
            i, j = rng.sample(range(12), 2)
            ni = rng.randint(0, n_lat - 1)
            nj = rng.randint(0, n_lat - 1)
            try_double_swap(i, j, ni, nj)
        else:
            # Guided swap: snap onto a line through two other points in the lattice
            i = rng.randint(0, 11)
            a, b = rng.sample(range(12), 2)
            if a == i or b == i:
                continue
            # Find lattice points collinear with cur[a] and cur[b]
            la = cur_lattice_idx[a]
            lb = cur_lattice_idx[b]
            key = (min(la, lb), max(la, lb))
            collinear = pair_collinear.get(key, [])
            if collinear:
                j = rng.choice(collinear)
                try_swap(i, j)
            else:
                j = rng.randint(0, n_lat - 1)
                try_swap(i, j)
        
        if best_score <= 6:
            break
    
    # 4) Final polish: try all single swaps from current best
    if time.time() < deadline - 10:
        for i in range(12):
            for j in range(n_lat):
                if j not in cur_lattice_idx:
                    try_swap(i, j)
                if best_score <= 6:
                    break
            if best_score <= 6:
                break
    
    if best_points is None:
        best_points = _build_grid_3x4()
    
    return best_points
