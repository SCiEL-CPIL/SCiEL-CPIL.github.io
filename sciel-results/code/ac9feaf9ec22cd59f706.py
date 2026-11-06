"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

def _score(points):
    """Count lines containing exactly 4 of the 16 points. Returns (count, valid).
    valid is False if any line has >=5 points or points are not distinct."""
    n = len(points)
    if n != 16:
        return 0, False
    # Check distinctness
    pts = [tuple(p) for p in points]
    if len(set(pts)) != 16:
        return 0, False
    # Check bounds
    for p in pts:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0, False
    # For each pair, determine the line, count points on it
    # Line through (x1,y1) and (x2,y2): direction (dx,dy), normal form: dy*x - dx*y = dy*x1 - dx*y1
    # Use a canonical key: (a, b, c) where a*x + b*y = c, with gcd normalization and sign convention
    line_counts = defaultdict(int)
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
            # Normalize: divide by gcd of |a|,|b|,|c|
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g == 0:
                continue
            a //= g
            b //= g
            c //= g
            # Sign convention: first nonzero of (a,b) should be positive
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            line_counts[key] += 1  # count pairs, not points
    # Convert pair counts to point counts: a line with k points has C(k,2) pairs
    # We need to find lines with exactly 4 points (C(4,2)=6 pairs) and check no line has 5+ (C(5,2)=10 pairs)
    # But multiple pairs map to same line. We need to reconstruct point counts.
    # Better approach: for each unique line key, count how many distinct points lie on it.
    # Let's redo: build line -> set of points
    line_pts = defaultdict(set)
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

def _score_fast(points):
    """Faster scoring using pair-count then dedup. Returns (count, valid)."""
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
        # pc = C(k,2) where k = number of points on line
        # C(k,2) = k*(k-1)/2
        # k=4 -> 6, k=5 -> 10, k=3 -> 3, k=2 -> 1
        # Check if pc corresponds to k>=5
        # Solve k*(k-1)/2 = pc for integer k
        if pc == 6:
            four_lines += 1
        elif pc == 10 or pc == 15 or pc == 21:  # k=5,6,7
            return 0, False
        # pc could also be from overlapping lines? No, each line is unique key.
        # But wait: pc counts pairs, and each pair belongs to exactly one line.
        # So pc = C(k,2) exactly for that line's point set.
    return four_lines, True

def _random_point():
    return [random.randint(-50, 50), random.randint(-50, 50)]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    # Initial seed: 4x4 grid
    best_pts = [[x, y] for x in range(4) for y in range(4)]
    best_score, best_valid = _score_fast(best_pts)
    
    rng = random.Random(42)
    
    # Multi-start SA
    num_starts = 20
    max_iter_per_start = 5000
    
    for s in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize: start from best or random perturbation
        if s == 0:
            pts = [list(p) for p in best_pts]
        else:
            # Perturb best
            pts = [list(p) for p in best_pts]
            for _ in range(rng.randint(1, 4)):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-3, 3), pts[idx][1] + rng.randint(-3, 3)]
        
        cur_score, cur_valid = _score_fast(pts)
        if not cur_valid:
            cur_score = 0
        
        T = 2.0  # temperature
        cooling = 0.9995
        
        for it in range(max_iter_per_start):
            if time.time() > deadline - 2.0:
                break
            
            # Move: swap one point to new random location
            idx = rng.randint(0, 15)
            old = list(pts[idx])
            # Perturb around current position
            delta = rng.randint(-4, 4)
            delta2 = rng.randint(-4, 4)
            if delta == 0 and delta2 == 0:
                delta = 1
            pts[idx] = [old[0] + delta, old[1] + delta2]
            
            new_score, new_valid = _score_fast(pts)
            if not new_valid:
                new_score = 0
            
            if new_score > cur_score:
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in pts]
            else:
                # Accept with probability exp((new - cur)/T)
                delta_score = new_score - cur_score
                if delta_score < 0:
                    accept_prob = math.exp(delta_score / max(T, 0.01))
                    if rng.random() > accept_prob:
                        pts[idx] = old
                else:
                    pass  # accept
                
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        # End of start: update best
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in pts]
    
    # Final: try to improve best with more focused search
    if time.time() < deadline - 3.0:
        pts = [list(p) for p in best_pts]
        cur_score, _ = _score_fast(pts)
        T = 1.0
        for it in range(3000):
            if time.time() > deadline - 1.0:
                break
            idx = rng.randint(0, 15)
            old = list(pts[idx])
            d1 = rng.randint(-2, 2)
            d2 = rng.randint(-2, 2)
            if d1 == 0 and d2 == 0:
                continue
            pts[idx] = [old[0] + d1, old[1] + d2]
            new_score, new_valid = _score_fast(pts)
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
        # Fallback to known valid 4x4 grid
        best_pts = [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts
# EVOLVE-BLOCK-END
