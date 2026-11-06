import time
import random
import math
from itertools import combinations

_DEADLINE = 110.0

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _valid_no_four(points):
    n = len(points)
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                return False
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
            if slopes[(dx, dy)] >= 3:
                return False
    return True

def _triple_count(points):
    """Count number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
        for k in slopes.values():
            if k >= 2:
                # C(k+1, 3) = (k+1)*k*(k-1)/6
                # k is the number of OTHER points on the line.
                # So total points on line is k+1.
                # Triples involving point i on this line: C(k, 2)
                count += k * (k - 1) // 2
    return count

def _build_conflict_graph(points):
    """Build conflict graph: edge between i,j if they share a collinear triple."""
    n = len(points)
    conflict_mask = [0] * n
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            if len(k) >= 2:
                # Any pair in k conflicts with i and each other
                for a_idx in range(len(k)):
                    for b_idx in range(a_idx + 1, len(k)):
                        a, b = k[a_idx], k[b_idx]
                        conflict_mask[i] |= (1 << a) | (1 << b)
                        conflict_mask[a] |= (1 << i) | (1 << b)
                        conflict_mask[b] |= (1 << i) | (1 << a)
    return conflict_mask

def _popcount(x):
    return bin(x).count('1')

def _max_independent_set(conflict_mask, n, deadline_start, current_best=20):
    """Find maximum independent set. Prune if count + available <= current_best."""
    best = [0]
    all_mask = (1 << n) - 1
    
    # Heuristic: order by degree
    degrees = [_popcount(conflict_mask[i]) for i in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    def dfs(available, count):
        if time.time() - deadline_start > 109.0:
            return
        # If we can't beat the current best, prune
        if count + _popcount(available) <= current_best:
            return
        if available == 0:
            if count > best[0]:
                best[0] = count
            return
        
        # Pick vertex with max degree in available
        v = None
        max_deg = -1
        for idx in order:
            if available & (1 << idx):
                deg = _popcount(conflict_mask[idx] & available)
                if deg > max_deg:
                    max_deg = deg
                    v = idx
        if v is None:
            return
        
        # Exclude v
        dfs(available & ~(1 << v), count)
        
        # Include v
        new_available = available & ~(1 << v) & ~conflict_mask[v]
        dfs(new_available, count + 1)
        
    dfs(all_mask, 0)
    return best[0]

def _score(points):
    n = len(points)
    conflict_mask = _build_conflict_graph(points)
    return _max_independent_set(conflict_mask, n, time.time())

def _generate_initial_grid(rng):
    # 5x4 grid with some randomness
    pts = []
    for x in range(5):
        for y in range(4):
            # Small perturbation
            px = x * 20 + rng.randint(-2, 2)
            py = y * 20 + rng.randint(-2, 2)
            pts.append((px, py))
    return pts

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = 20
    best_triples = -1
    
    def consider(pts):
        nonlocal best_pts, best_score, best_triples
        if not _valid_no_four(pts):
            return
        # Fast proxy: triple count
        tc = _triple_count(pts)
        # If triple count is not promising, skip expensive MIS?
        # But we need to be careful. A config with fewer triples might still have a smaller MIS?
        # Generally, more triples -> more conflicts -> smaller MIS.
        # So we prioritize high triple counts.
        
        # Only compute MIS if this looks like a candidate
        # Heuristic: if tc > best_triples, definitely check.
        # Also check periodically.
        if tc <= best_triples and best_score < 20:
             # If we already have a good score, and this one has fewer triples, 
             # it's unlikely to be better. But to be safe, check if tc is close.
             if tc < best_triples - 5:
                 return
        
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = list(pts)
            best_triples = tc
            return True
        elif s == best_score and tc > best_triples:
            # Same score, but more triples might be more robust or lead to further improvements
            best_pts = list(pts)
            best_triples = tc
            return True
        return False

    # Strategy 1: Multi-start Simulated Annealing on Triple Count
    
    # Initial candidates
    candidates = []
    for _ in range(10):
        candidates.append(_generate_initial_grid(rng))
    
    # Add some structured ones
    # 4x5 grid
    candidates.append([(x * 20, y * 20) for x in range(4) for y in range(5)])
    # Diagonal grid
    candidates.append([(x * 20, y * 20 + x * 5) for x in range(4) for y in range(5)])
    
    for init_pts in candidates:
        if time.time() - start > 105.0:
            break
        
        current_pts = list(init_pts)
        if not _valid_no_four(current_pts):
            continue
            
        current_tc = _triple_count(current_pts)
        consider(current_pts)
        
        temp = 10.0
        min_temp = 0.1
        for step in range(5000):
            if time.time() - start > 108.0:
                break
            
            # Move: pick a random point and perturb it
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            
            # Try a few moves
            best_move = None
            best_move_tc = current_tc
            
            for _ in range(5):
                # Random jump
                nx = x + rng.randint(-5, 5)
                ny = y + rng.randint(-5, 5)
                
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            if tc > best_move_tc:
                                best_move_tc = tc
                                best_move = new_pt
                        current_pts[idx] = old_pt
            
            if best_move is not None:
                current_pts[idx] = best_move
                current_tc = best_move_tc
                # Check if this is a new best for the SA run
                consider(current_pts)
            else:
                # Accept worse move with probability
                # Pick a random move again for acceptance check
                idx = rng.randrange(20)
                x, y = current_pts[idx]
                nx = x + rng.randint(-5, 5)
                ny = y + rng.randint(-5, 5)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            delta = tc - current_tc
                            if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.01)):
                                current_pts[idx] = new_pt
                                current_tc = tc
                                consider(current_pts)
                            else:
                                current_pts[idx] = old_pt
                        else:
                            current_pts[idx] = old_pt
            
            temp *= 0.999
            
    # Strategy 2: Focused Local Search on Best Found
    if best_pts:
        current_pts = list(best_pts)
        current_tc = _triple_count(current_pts)
        temp = 1.0
        for step in range(20000):
            if time.time() - start > 109.0:
                break
            
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            
            # Try to move point to a position that increases triples
            # Sample from existing lines?
            best_move = None
            best_move_tc = current_tc
            
            for _ in range(10):
                # Small perturbation
                nx = x + rng.randint(-3, 3)
                ny = y + rng.randint(-3, 3)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            if tc > best_move_tc:
                                best_move_tc = tc
                                best_move = new_pt
                        current_pts[idx] = old_pt
            
            if best_move:
                current_pts[idx] = best_move
                current_tc = best_move_tc
                consider(current_pts)
            else:
                # Random walk
                idx = rng.randrange(20)
                x, y = current_pts[idx]
                nx = x + rng.randint(-2, 2)
                ny = y + rng.randint(-2, 2)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            delta = tc - current_tc
                            if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.01)):
                                current_pts[idx] = new_pt
                                current_tc = tc
                                consider(current_pts)
                            else:
                                current_pts[idx] = old_pt
                        else:
                            current_pts[idx] = old_pt
            temp *= 0.9995

    if best_pts is None:
        # Fallback: simple grid
        return [[x * 20, y * 20] for x in range(5) for y in range(4)]
    
    return [list(p) for p in best_pts]
