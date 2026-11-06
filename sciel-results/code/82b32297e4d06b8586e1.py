import random
import itertools
import numpy as np

def _check_no_four_collinear(points):
    """Check no four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Check if there's a fourth point on the line through i,j,k
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                # Line equation: (yj-yi)(x - xi) - (xj-xi)(y - yi) = 0
                a = yj - yi
                b = xi - xj
                c = xj*yi - xi*yj
                # Check k is on line (should be by construction of triple)
                # Check all other points
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    xl, yl = points[l]
                    if a * xl + b * yl + c == 0:
                        return False
    return True

def _score(points):
    """Largest subset with no three collinear (exact branch and bound)."""
    n = len(points)
    # Precompute which triples are collinear
    collinear = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                # Check if i,j,k collinear
                if (xj - xi) * (yk - yi) == (yk - yi) * (xj - xi):
                    # Wait, that's always true. Correct check:
                    # (xj-xi)*(yk-yi) == (yk-yi)*(xj-xi) -> wrong
                    # Correct: (xj-xi)*(yk-yi) == (xk-xi)*(yj-yi)
                    if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                        collinear[i][k] = True
                        collinear[k][i] = True
    # Actually let me redo: collinear triple i,j,k
    triple_collinear = {}
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                    triple_collinear[(i,j,k)] = True
    
    # Branch and bound to find largest subset with no three collinear
    best = [0]
    
    def dfs(idx, current_size, chosen):
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            best[0] = max(best[0], current_size)
            return
        # Option 1: skip point idx
        dfs(idx + 1, current_size, chosen)
        # Option 2: take point idx if no two in chosen form a collinear triple with idx
        can_take = True
        for a in chosen:
            for b in chosen:
                if a == b:
                    continue
                key = tuple(sorted((a, b, idx)))
                if key in triple_collinear:
                    can_take = False
                    break
            if not can_take:
                break
        if can_take:
            chosen.append(idx)
            dfs(idx + 1, current_size + 1, chosen)
            chosen.pop()
    
    dfs(0, 0, [])
    return best[0]

def construct():
    # Start with a 5x4 grid and perturb
    base = [(i, j) for i in range(5) for j in range(4)]
    # Try random perturbations
    best_score = 20
    best_points = None
    
    random.seed(42)
    for attempt in range(2000):
        pts = []
        for i, j in base:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            x = i + dx
            y = j + dy
            if -100 <= x <= 100 and -100 <= y <= 100:
                pts.append([x, y])
        # Check distinct
        if len(set(tuple(p) for p in pts)) != 20:
            continue
        if not _check_no_four_collinear(pts):
            continue
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = pts
    
    if best_points is None:
        # Fallback: use parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
