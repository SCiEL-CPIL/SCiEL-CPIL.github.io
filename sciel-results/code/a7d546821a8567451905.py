# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 138.0

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score_fast(pts):
    n = 16
    four_lines = set()
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            extra = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = pts[k]
                if dx * (yk - yi) == dy * (xk - xi):
                    extra += 1
            total = 2 + extra
            if total >= 5:
                return -1
            if total == 4:
                members = [i, j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    xk, yk = pts[k]
                    if dx * (yk - yi) == dy * (xk - xi):
                        members.append(k)
                four_lines.add(tuple(sorted(members)))
    return len(four_lines)

def _valid(pts):
    seen = set()
    for p in pts:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _mutate(pts, rng):
    """Create a mutated copy."""
    new_pts = [p[:] for p in pts]
    n_moves = rng.randint(1, 3)
    for _ in range(n_moves):
        idx = rng.randint(0, 15)
        move_type = rng.random()
        if move_type < 0.5:
            # Small shift
            sx = rng.randint(-4, 4)
            sy = rng.randint(-4, 4)
            if sx == 0 and sy == 0:
                sx = 1
            new_pts[idx] = [new_pts[idx][0] + sx, new_pts[idx][1] + sy]
        else:
            # Larger jump
            new_pts[idx] = [rng.randint(-200, 200), rng.randint(-200, 200)]
    return new_pts

def construct():
    start = time.time()
    rng = random.Random(42)
    
    # Initial population
    base = [[x, y] for x in range(4) for y in range(4)]
    pop = []
    for _ in range(20):
        if _ == 0:
            pop.append([p[:] for p in base])
        else:
            pop.append(_mutate(base, rng))
    
    # Score initial population
    scored = []
    for p in pop:
        if _valid(p):
            s = _score_fast(p)
            scored.append((s, p))
        else:
            scored.append((-1, p))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    best_pts = [p[:] for p in scored[0][1]]
    
    # Evolutionary loop
    gen = 0
    while time.time() - start < _DEADLINE:
        gen += 1
        new_pop = []
        
        # Elitism: keep top 5
        for s, p in scored[:5]:
            new_pop.append(p)
        
        # Generate offspring
        while len(new_pop) < 20:
            # Pick parent via tournament
            a, b = rng.sample(scored, 2)
            parent = a[1] if a[0] >= b[0] else b[1]
            child = _mutate(parent, rng)
            if _valid(child):
                s = _score_fast(child)
            else:
                s = -1
            new_pop.append(child)
            scored.append((s, child))
        
        # Re-score and sort
        scored = []
        for p in new_pop:
            s = _score_fast(p)
            scored.append((s, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_pts = [p[:] for p in scored[0][1]]
    
    return best_pts
# EVOLVE-BLOCK-END
