# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _score(pts):
    """Compute the max multiplicity of a positive doubled area among all triples."""
    n = len(pts)
    if n < 3:
        return 0
    cnt = Counter()
    for i in range(n - 2):
        x1, y1 = pts[i]
        for j in range(i + 1, n - 1):
            x2, y2 = pts[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                det = dx12 * (y3 - y1) - dy12 * (x3 - x1)
                if det > 0:
                    cnt[det] += 1
                elif det < 0:
                    cnt[-det] += 1
    if not cnt:
        return 0
    return max(cnt.values())

def _try_move(pts, i, dx, dy, deadline, best_score, best_pts):
    """Try moving point i by (dx,dy). Return (new_score, new_pts) if improved, else (best_score, best_pts)."""
    x, y = pts[i]
    nx, ny = x + dx, y + dy
    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
        return best_score, best_pts
    # Check collision
    for j in range(len(pts)):
        if j == i:
            continue
        if pts[j][0] == nx and pts[j][1] == ny:
            return best_score, best_pts
    if time.time() > deadline:
        return best_score, best_pts
    new_pts = list(pts)
    new_pts[i] = (nx, ny)
    s = _score(new_pts)
    if s > best_score:
        return s, new_pts
    return best_score, best_pts

def _local_search(pts, deadline, max_rounds=200):
    """Hill-climbing local search with random restarts within budget."""
    best_pts = list(pts)
    best_score = _score(best_pts)
    rng = random.Random(42)
    rounds = 0
    while time.time() < deadline and rounds < max_rounds:
        improved = False
        # Try all single-point moves in random order
        idxs = list(range(len(best_pts)))
        rng.shuffle(idxs)
        for i in idxs:
            if time.time() > deadline:
                break
            x, y = best_pts[i]
            # Try moves in increasing radius
            for r in range(1, 6):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) == r or abs(dy) == r:
                            if time.time() > deadline:
                                break
                            s, pts_new = _try_move(best_pts, i, dx, dy, deadline, best_score, best_pts)
                            if s > best_score:
                                best_score = s
                                best_pts = pts_new
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # No improvement; try a random perturbation to escape local optimum
            if time.time() > deadline:
                break
            perturbed = list(best_pts)
            for _ in range(3):
                i = rng.randrange(len(perturbed))
                x, y = perturbed[i]
                nx = max(-30, min(30, x + rng.randint(-8, 8)))
                ny = max(-30, min(30, y + rng.randint(-8, 8)))
                # Ensure distinct
                for j in range(len(perturbed)):
                    if j != i and perturbed[j] == (nx, ny):
                        nx = max(-30, min(30, nx + 1))
                perturbed[i] = (nx, ny)
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_pts = perturbed
            else:
                # Keep best_pts, continue
                pass
        rounds += 1
    return best_pts, best_score

def _generate_candidates():
    """Generate diverse initial configurations."""
    candidates = []
    rng = random.Random(123)
    
    # 1. Grid-like patterns (rows with offsets)
    for rows in range(3, 8):
        per_row = 30 // rows
        rem = 30 % rows
        pts = []
        y = -15
        for r in range(rows):
            count = per_row + (1 if r < rem else 0)
            x_start = -15 + r * 2
            for c in range(count):
                x = x_start + c * 4
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
            y += 6
        if len(pts) == 30 and len(set(pts)) == 30:
            candidates.append(pts)
    
    # 2. Concentric ring patterns
    for r1 in range(2, 10):
        for r2 in range(r1 + 2, 20):
            pts = []
            # Center
            pts.append((0, 0))
            # Ring 1
            import math
            n1 = min(14, 30 - 1 - 14)
            for k in range(n1):
                ang = 2 * math.pi * k / n1
                x = round(r1 * math.cos(ang))
                y = round(r1 * math.sin(ang))
                if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
            # Ring 2
            n2 = 30 - len(pts)
            for k in range(n2 * 2):
                ang = 2 * math.pi * k / (n2 * 2)
                x = round(r2 * math.cos(ang))
                y = round(r2 * math.sin(ang))
                if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
                    if len(pts) == 30:
                        break
            if len(pts) == 30 and len(set(pts)) == 30:
                candidates.append(pts)
    
    # 3. Arithmetic progression in one direction (collinear-ish but perturbed)
    for base_dx in [1, 2, 3]:
        for base_dy in [1, 2, 3]:
            pts = []
            for i in range(30):
                x = (i * base_dx) % 61 - 30
                y = (i * base_dy) % 61 - 30
                if (x, y) not in pts:
                    pts.append((x, y))
                else:
                    x = (x + 1) % 61 - 30
                    pts.append((x, y))
            if len(pts) == 30 and len(set(pts)) == 30:
                candidates.append(pts)
    
    # 4. Random configurations
    for trial in range(20):
        pts = set()
        while len(pts) < 30:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 5. Structured: points on a parabola-like curve
    for a in [1, 2, 3]:
        pts = []
        for i in range(-15, 15):
            x = i
            y = (a * i * i) % 61 - 30
            if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) == 30 and len(set(pts)) == 30:
            candidates.append(pts)
    
    # 6. Lattice with small perturbations
    for scale in [3, 4, 5]:
        pts = []
        x = -30
        y = -30
        while len(pts) < 30:
            if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
            x += scale
            if x > 30:
                x = -30
                y += scale
        if len(pts) == 30 and len(set(pts)) == 30:
            candidates.append(pts)
    
    # 7. Points arranged in a hexagonal pattern
    pts = []
    for q in range(-5, 5):
        for r in range(-5, 5):
            x = q + r // 2
            y = r * 3
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                pts.append((x, y))
            if len(pts) == 30:
                break
        if len(pts) == 30:
            break
    if len(pts) == 30 and len(set(pts)) == 30:
        candidates.append(pts)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Generate initial candidates
    candidates = _generate_candidates()
    
    # Score all candidates
    best_pts = None
    best_score = 0
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = list(pts)
    
    if best_pts is None:
        # Fallback: simple grid
        best_pts = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score(best_pts)
    
    # Local search on best candidate
    if time.time() < deadline:
        best_pts, best_score = _local_search(best_pts, deadline, max_rounds=500)
    
    # Try local search on a few other good candidates
    for pts in candidates[:10]:
        if time.time() > deadline:
            break
        s = _score(pts)
        if s >= best_score - 2:
            pts2, s2 = _local_search(pts, deadline, max_rounds=100)
            if s2 > best_score:
                best_score = s2
                best_pts = pts2
    
    # Final: ensure valid
    if len(best_pts) != 30 or len(set(best_pts)) != 30:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    for p in best_pts:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            best_pts = [(x, y) for x in range(6) for y in range(5)]
            break
    
    return best_pts
# EVOLVE-BLOCK-END
