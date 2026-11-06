"""Improved program: faster scoring + structured search for 4-point lines."""
import time
import math
import random
from collections import defaultdict

def _score_fast(points):
    """Count lines containing exactly 4 of the 16 points. Returns (count, valid).
    valid is False if any line has >=5 points or points are not distinct."""
    n = len(points)
    if n != 16:
        return 0, False
    pts = [tuple(p) for p in points]
    if len(set(pts)) != 16:
        return 0, False
    for p in pts:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0, False
    
    # Build line -> set of point indices
    line_pts = defaultdict(set)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line: dy*(x - x1) - dx*(y - y1) = 0 => dy*x - dx*y = dy*x1 - dx*y1
            a = dy
            b = -dx
            c = dy * x1 - dx * y1
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g == 0:
                continue
            a //= g
            b //= g
            c //= g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    four_lines = 0
    for key, indices in line_pts.items():
        k = len(indices)
        if k >= 5:
            return 0, False
        if k == 4:
            four_lines += 1
    return four_lines, True

def _score_pair(points):
    """Faster: count pairs per line, then infer point counts.
    A line with k points has C(k,2) pairs. We check for k>=5 (invalid) and k==4 (score).
    But multiple lines can share pairs? No, each pair defines exactly one line.
    However, a line with k points will have exactly C(k,2) pairs mapping to it.
    So we can count pairs per line key, and if count == C(4,2)=6, it's a 4-line.
    If count >= C(5,2)=10, it's invalid.
    But wait: a line with k points has exactly C(k,2) pairs. So pair_count = C(k,2).
    k=2 -> 1, k=3 -> 3, k=4 -> 6, k=5 -> 10, k=6 -> 15, etc.
    We need to detect if any line has k>=5.
    """
    n = len(points)
    if n != 16:
        return 0, False
    pts = [tuple(p) for p in points]
    if len(set(pts)) != 16:
        return 0, False
    for p in pts:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0, False
    
    line_pair_count = defaultdict(int)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = dy * x1 - dx * y1
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g == 0:
                continue
            a //= g
            b //= g
            c //= g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_pair_count[(a, b, c)] += 1
    
    four_lines = 0
    for key, pc in line_pair_count.items():
        # pc = C(k,2) for some k
        # Check if k >= 5: C(5,2)=10, C(6,2)=15, C(7,2)=21, ...
        # For 16 points, max k is 16, C(16,2)=120
        if pc >= 10:
            return 0, False
        if pc == 6:
            four_lines += 1
    return four_lines, True

def _initial_config():
    """Start with a 4x4 grid, which has some 4-point lines."""
    return [[x, y] for x in range(4) for y in range(4)]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    rng = random.Random(12345)
    
    # Start with 4x4 grid
    best_pts = _initial_config()
    best_score, best_valid = _score_pair(best_pts)
    if not best_valid:
        best_score = 0
    
    # Multi-start with different strategies
    num_starts = 30
    max_iter = 8000
    
    for s in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initialization strategies
        if s == 0:
            pts = [list(p) for p in best_pts]
        elif s == 1:
            # Perturbed grid
            pts = [list(p) for p in _initial_config()]
            for _ in range(rng.randint(2, 5)):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-3, 3), pts[idx][1] + rng.randint(-3, 3)]
        elif s == 2:
            # Random initialization in larger range
            pts = [[rng.randint(-20, 20), rng.randint(-20, 20)] for _ in range(16)]
        elif s == 3:
            # Perturb best more aggressively
            pts = [list(p) for p in best_pts]
            for _ in range(rng.randint(3, 7)):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-5, 5), pts[idx][1] + rng.randint(-5, 5)]
        else:
            # Mix: start from best with random perturbations
            pts = [list(p) for p in best_pts]
            num_perturb = rng.randint(1, 4)
            for _ in range(num_perturb):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-4, 4), pts[idx][1] + rng.randint(-4, 4)]
        
        cur_score, cur_valid = _score_pair(pts)
        if not cur_valid:
            cur_score = 0
        
        # Simulated annealing
        T = 3.0
        cooling = 0.9996
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            # Move: relocate one point
            idx = rng.randint(0, 15)
            old = list(pts[idx])
            
            # Choose new position: local or global
            if rng.random() < 0.7:
                # Local move
                delta = rng.randint(-4, 4)
                delta2 = rng.randint(-4, 4)
                if delta == 0 and delta2 == 0:
                    delta = 1
                new_x = old[0] + delta
                new_y = old[1] + delta2
            else:
                # Global move: jump to random location near other points
                ref_idx = rng.randint(0, 15)
                if ref_idx == idx:
                    ref_idx = (ref_idx + 1) % 16
                new_x = pts[ref_idx][0] + rng.randint(-5, 5)
                new_y = pts[ref_idx][1] + rng.randint(-5, 5)
            
            # Clamp to bounds
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            pts[idx] = [new_x, new_y]
            
            new_score, new_valid = _score_pair(pts)
            if not new_valid:
                new_score = 0
            
            if new_score > cur_score:
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in pts]
            else:
                delta_score = new_score - cur_score
                if delta_score < 0:
                    accept_prob = math.exp(delta_score / max(T, 0.01))
                    if rng.random() > accept_prob:
                        pts[idx] = old
                # else: accept (same score)
            
            T *= cooling
            if T < 0.05:
                T = 0.05
        
        # End of start
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in pts]
    
    # Final polish: focused search around best
    if time.time() < deadline - 3.0:
        pts = [list(p) for p in best_pts]
        cur_score, _ = _score_pair(pts)
        T = 0.8
        for it in range(5000):
            if time.time() > deadline - 1.0:
                break
            idx = rng.randint(0, 15)
            old = list(pts[idx])
            d1 = rng.randint(-2, 2)
            d2 = rng.randint(-2, 2)
            if d1 == 0 and d2 == 0:
                continue
            pts[idx] = [old[0] + d1, old[1] + d2]
            new_score, new_valid = _score_pair(pts)
            if not new_valid:
                new_score = 0
            if new_score > cur_score:
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in pts]
            else:
                delta_s = new_score - cur_score
                if delta_s < 0 and rng.random() > math.exp(delta_s / max(T, 0.01)):
                    pts[idx] = old
            T *= 0.999
    
    # Validate final result
    final_score, final_valid = _score_fast(best_pts)
    if not final_valid:
        # Fallback
        best_pts = _initial_config()
    
    return best_pts
