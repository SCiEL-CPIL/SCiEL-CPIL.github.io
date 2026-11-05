"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

def _normalize_line(x1, y1, x2, y2):
    """Normalize line through two integer points to a canonical key."""
    dx = x2 - x1
    dy = y2 - y1
    A = dy
    B = -dx
    C = dy * x1 - dx * y1
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if C != 0 else gcd(abs(A), abs(B))
    if g > 1:
        A //= g
        B //= g
        C //= g
    return (A, B, C)

def _count_4lines(points):
    """Count lines with exactly 4 points. Returns -1 if any line has 5+ points."""
    n = len(points)
    line_pts = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _normalize_line(points[i][0], points[i][1], points[j][0], points[j][1])
            line_pts[key].add(i)
    count = 0
    for key, idxs in line_pts.items():
        if len(idxs) == 4:
            count += 1
        elif len(idxs) >= 5:
            return -1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # 4x4 grid baseline: 10 four-point lines
    base = [[x, y] for x in range(4) for y in range(4)]
    best = [p[:] for p in base]
    best_score = _count_4lines(best)  # should be 10
    
    rng = random.Random(12345)
    n = 16
    
    # Hill climbing with random restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            current = [p[:] for p in base]
        else:
            # Start from best with some perturbation, or fresh random
            if rng.random() < 0.5:
                current = [p[:] for p in best]
                # Perturb 1-3 points
                for _ in range(rng.randint(1, 3)):
                    idx = rng.randint(0, n - 1)
                    current[idx] = [rng.randint(-10, 10), rng.randint(-10, 10)]
            else:
                # Random starting point
                current = [[rng.randint(-5, 5), rng.randint(-5, 5)] for _ in range(n)]
        
        cur_score = _count_4lines(current)
        if cur_score == -1:
            continue
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 5:
                    break
                old = current[idx][:]
                # Try nearby moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = old[0] + dx, old[1] + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        new_pt = [nx, ny]
                        if any(p == new_pt for p in current):
                            continue
                        current[idx] = new_pt
                        new_score = _count_4lines(current)
                        current[idx] = old
                        if new_score != -1 and new_score > cur_score:
                            current[idx] = new_pt
                            cur_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if cur_score > best_score:
            best = [p[:] for p in current]
            best_score = cur_score
    
    # Final check
    if _count_4lines(best) == -1:
        return base
    return best
# EVOLVE-BLOCK-END
