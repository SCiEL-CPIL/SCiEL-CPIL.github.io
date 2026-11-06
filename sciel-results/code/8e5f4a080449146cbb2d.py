"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    # Precompute all triple indices for 30 points
    n = 30
    triple_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triple_indices.append((i, j, k))
    num_triples = len(triple_indices)

    def evaluate_fast(points):
        """Faster evaluation using numpy-style vectorized approach with lists."""
        counts = {}
        for (i, j, k) in triple_indices:
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det > 0:
                counts[det] = counts.get(det, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    # Initial configuration: two horizontal lines at y=0 and y=1, each with 15 points
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate_fast(best_points)

    # Try grid configurations: w x h grids with exactly 30 points
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
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts
        if time.time() > deadline:
            return best_points

    # Try different line distances for two parallel lines
    for d in range(1, 8):
        pts = [(x, 0) for x in range(15)] + [(x, d) for x in range(15)]
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
            continue
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts
        if time.time() > deadline:
            return best_points

    # Try 3-line configurations with varying distributions
    for d in range(1, 5):
        for a in range(5, 26):
            for b in range(5, 26):
                c = 30 - a - b
                if c < 1 or c > 30:
                    continue
                if 2 * d > 30:
                    continue
                pts = [(x, 0) for x in range(a)] + [(x, d) for x in range(b)] + [(x, 2 * d) for x in range(c)]
                if len(set(pts)) != 30:
                    continue
                if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
                    continue
                score = evaluate_fast(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
                if time.time() > deadline:
                    return best_points

    # Try 4-line configurations
    for d in range(1, 4):
        for a in range(4, 16):
            for b in range(4, 16):
                for c in range(4, 16):
                    e = 30 - a - b - c
                    if e < 1 or e > 30:
                        continue
                    if 3 * d > 30:
                        continue
                    pts = [(x, 0) for x in range(a)] + [(x, d) for x in range(b)] + [(x, 2 * d) for x in range(c)] + [(x, 3 * d) for x in range(e)]
                    if len(set(pts)) != 30:
                        continue
                    if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
                        continue
                    score = evaluate_fast(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
                    if time.time() > deadline:
                        return best_points

    # Local search: try moving individual points to improve score
    random.seed(42)

    # First pass: small perturbations (dx, dy in [-3, 3])
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
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
                    score = evaluate_fast(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Second pass: random perturbations with larger moves (±8)
    for iteration in range(3000):
        if time.time() > deadline:
            break
        new_points = list(best_points)
        idx = random.randint(0, 29)
        old_x, old_y = new_points[idx]
        new_x = old_x + random.randint(-8, 8)
        new_y = old_y + random.randint(-8, 8)
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        new_points[idx] = (new_x, new_y)
        if len(set(new_points)) != 30:
            continue
        score = evaluate_fast(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    # Third pass: try swapping pairs of points
    for iteration in range(1500):
        if time.time() > deadline:
            break
        new_points = list(best_points)
        i = random.randint(0, 29)
        j = random.randint(0, 29)
        if i == j:
            continue
        new_points[i], new_points[j] = new_points[j], new_points[i]
        if len(set(new_points)) != 30:
            continue
        score = evaluate_fast(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    # Fourth pass: try moving two points simultaneously (±5)
    for iteration in range(800):
        if time.time() > deadline:
            break
        new_points = list(best_points)
        i = random.randint(0, 29)
        j = random.randint(0, 29)
        if i == j:
            continue
        dx1 = random.randint(-5, 5)
        dy1 = random.randint(-5, 5)
        dx2 = random.randint(-5, 5)
        dy2 = random.randint(-5, 5)
        if dx1 == 0 and dy1 == 0:
            continue
        if dx2 == 0 and dy2 == 0:
            continue
        nx1 = new_points[i][0] + dx1
        ny1 = new_points[i][1] + dy1
        nx2 = new_points[j][0] + dx2
        ny2 = new_points[j][1] + dy2
        if not (-30 <= nx1 <= 30 and -30 <= ny1 <= 30):
            continue
        if not (-30 <= nx2 <= 30 and -30 <= ny2 <= 30):
            continue
        new_points[i] = (nx1, ny1)
        new_points[j] = (nx2, ny2)
        if len(set(new_points)) != 30:
            continue
        score = evaluate_fast(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    # Fifth pass: random restarts with larger perturbations
    for restart in range(50):
        if time.time() > deadline:
            break
        # Create a perturbed version of best_points
        new_points = list(best_points)
        for idx in range(30):
            if random.random() < 0.4:
                old_x, old_y = new_points[idx]
                new_x = old_x + random.randint(-10, 10)
                new_y = old_y + random.randint(-10, 10)
                if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                    new_points[idx] = (new_x, new_y)
        if len(set(new_points)) != 30:
            continue
        score = evaluate_fast(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    # Final local search with smaller steps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
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
                    score = evaluate_fast(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_points
# EVOLVE-BLOCK-END
