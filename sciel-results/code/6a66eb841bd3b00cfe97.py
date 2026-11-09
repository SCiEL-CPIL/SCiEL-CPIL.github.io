import time
import random
import math

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_START_TIME = None

def _elapsed():
    return time.time() - _START_TIME

def _has_four_collinear(pts):
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        slopes = {}
        for j in range(n):
            if i == j: continue
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0: continue
            g = math.gcd(abs(dx), abs(dy))
            dxn = dx // g
            dyn = dy // g
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            key = (dxn, dyn)
            slopes[key] = slopes.get(key, 0) + 1
        for cnt in slopes.values():
            if cnt >= 3:
                return True
    return False

def _max_no_three_collinear(pts):
    n = len(pts)
    line_mask = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            ax, ay = pts[a]
            bx, by = pts[b]
            dx = bx - ax
            dy = by - ay
            mask = 0
            for c in range(n):
                if c == a or c == b: continue
                cx, cy = pts[c]
                if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) == 0:
                    mask |= (1 << c)
            line_mask[a][b] = mask
            line_mask[b][a] = mask

    best = [0]
    
    def search(start_idx, current_size, mask):
        if current_size > best[0]:
            best[0] = current_size
        if current_size + (n - start_idx) <= best[0]:
            return

        for i in range(start_idx, n):
            can_add = True
            # Check if i conflicts with any pair in mask
            # Iterate over pairs in mask
            # To avoid iterating over all n^2, we can precompute for each i, the list of pairs that conflict?
            # Or just iterate.
            m = mask
            # Extract indices
            # Since n=20, we can just loop a, b
            # But we only care about a,b in mask.
            # Let's loop a from 0 to n-1
            # If mask has bit a:
            #   Loop b from a+1 to n-1
            #   If mask has bit b:
            #      if line_mask[a][b] & (1 << i): can_add = False
            for a in range(n):
                if not (mask & (1 << a)):
                    continue
                for b in range(a + 1, n):
                    if not (mask & (1 << b)):
                        continue
                    if line_mask[a][b] & (1 << i):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                search(i + 1, current_size + 1, mask | (1 << i))

    search(0, 0, 0)
    return best[0]

def _score_config(pts):
    if _has_four_collinear(pts):
        return 10**9
    return _max_no_three_collinear(pts)

def _generate_random_valid(rng):
    while True:
        pts = []
        used = set()
        for _ in range(20):
            while True:
                x = rng.randint(-100, 100)
                y = rng.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append([x, y])
                    break
        if not _has_four_collinear(pts):
            return pts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_pts = _generate_random_valid(rng)
    best_score = _score_config(best_pts)
    
    no_improve = 0
    
    while _elapsed() < _DEADLINE:
        # Local search step
        idx = rng.randint(0, 19)
        old_pt = best_pts[idx]
        
        new_pt = None
        for _ in range(50):
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) == (old_pt[0], old_pt[1]):
                continue
            dup = False
            for j in range(20):
                if j != idx and best_pts[j][0] == x and best_pts[j][1] == y:
                    dup = True
                    break
            if not dup:
                new_pt = [x, y]
                break
        
        if new_pt is None:
            continue
            
        new_pts = best_pts.copy()
        new_pts[idx] = new_pt
        
        if _has_four_collinear(new_pts):
            continue
            
        new_score = _max_no_three_collinear(new_pts)
        
        if new_score < best_score:
            best_score = new_score
            best_pts = new_pts
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > 200:
            best_pts = _generate_random_valid(rng)
            best_score = _score_config(best_pts)
            no_improve = 0

    return best_pts

# EVOLVE-BLOCK-END
