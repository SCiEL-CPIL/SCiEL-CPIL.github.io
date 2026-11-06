import time
import random
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    def evaluate(points):
        """Compute the maximum multiplicity of any positive doubled area."""
        counts = Counter()
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    dx2 = x3 - x1
                    dy2 = y3 - y1
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counts[det] += 1
        if not counts:
            return 0
        return max(counts.values())

    # Strategy: Start with the two-parallel-lines configuration (15 points on y=0,
    # 15 on y=1) which is known to give a high score. Then try to improve via
    # local search and other structured configurations.

    # Initial configuration: two horizontal lines
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate(best_points)

    # Try grid configurations
    grid_configs = [
        (5, 6), (6, 5), (10, 3), (3, 10), (15, 2), (2, 15)
    ]
    for w, h in grid_configs:
        if w * h != 30:
            continue
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
            continue
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
        if time.time() > deadline:
            return best_points

    # Try shifted/scaled versions of the two-line configuration
    for d in range(1, 6):
        pts = [(x, 0) for x in range(15)] + [(x, d) for x in range(15)]
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
            continue
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
        if time.time() > deadline:
            return best_points

    # Try 3-line configurations with varying distributions
    for d in range(1, 4):
        for a in range(8, 23):
            for b in range(8, 23):
                c = 30 - a - b
                if c < 1 or c > 30:
                    continue
                if b < 1:
                    continue
                if 2 * d > 30:
                    continue
                pts = [(x, 0) for x in range(a)] + [(x, d) for x in range(b)] + [(x, 2 * d) for x in range(c)]
                if len(set(pts)) != 30:
                    continue
                if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
                    continue
                score = evaluate(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts[:]
                if time.time() > deadline:
                    return best_points

    # Local search: try moving individual points to improve score
    random.seed(42)

    # First pass: small perturbations (dx, dy in [-2, 2])
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Second pass: larger perturbations (dx, dy in [-5, 5])
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(30):
                if time.time() > deadline:
                    break
                old_x, old_y = best_points[idx]
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                            continue
                        new_points = list(best_points)
                        new_points[idx] = (new_x, new_y)
                        if len(set(new_points)) != 30:
                            continue
                        score = evaluate(new_points)
                        if score > best_score:
                            best_score = score
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Third pass: random restarts with local search
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Random restart
            current_points = [(random.randint(-15, 15), random.randint(-15, 15)) for _ in range(30)]
            # Ensure distinct
            while len(set(current_points)) < 30:
                idx = random.randint(0, 29)
                current_points[idx] = (random.randint(-15, 15), random.randint(-15, 15))
            
            current_score = evaluate(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
            
            # Local search on this configuration
            improved = True
            while improved and time.time() < deadline:
                improved = False
                idx = random.randint(0, 29)
                old_x, old_y = current_points[idx]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                            continue
                        new_points = list(current_points)
                        new_points[idx] = (new_x, new_y)
                        if len(set(new_points)) != 30:
                            continue
                        score = evaluate(new_points)
                        if score > current_score:
                            current_score = score
                            current_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]

    return best_points
