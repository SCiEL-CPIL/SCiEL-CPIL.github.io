"""Improved program: fast scoring + grid-subset init + line-aware moves."""
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

def _score_pair(points):
    """Fast scoring: count pairs per line, infer point counts from binomial coefficients.
    C(2,2)=1, C(3,2)=3, C(4,2)=6, C(5,2)=10, C(6,2)=15, C(7,2)=21, ...
    For 16 points max pairs per line is C(16,2)=120."""
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
        if pc >= 10:
            return 0, False
        if pc == 6:
            four_lines += 1
    return four_lines, True

def _get_line_info(points):
    """Return dict: line_key -> sorted list of point indices on that line."""
    n = len(points)
    pts = [tuple(p) for p in points]
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
    return {k: sorted(v) for k, v in line_pts.items()}

def _find_line_extension(points, line_key, existing_indices):
    """Given a line and its existing points, find integer points on the same line
    within bounds that are not already in the set. Returns list of candidate (x,y)."""
    pts = [tuple(p) for p in points]
    existing = set(existing_indices)
    a, b, c = line_key
    # Line: a*x + b*y = c
    candidates = []
    # Find two existing points to determine direction
    if len(existing_indices) < 2:
        return candidates
    p1 = pts[existing_indices[0]]
    p2 = pts[existing_indices[1]]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return candidates
    dx //= g
    dy //= g
    # Step along the line in both directions
    for t in range(1, 30):
        for sign in (1, -1):
            nx = p1[0] + sign * t * dx
            ny = p1[1] + sign * t * dy
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                # Check not already a point
                pt = (nx, ny)
                if pt not in pts:
                    candidates.append([nx, ny])
    return candidates

def _initial_config():
    """Start with a 4x4 grid, which has some 4-point lines."""
    return [[x, y] for x in range(4) for y in range(4)]

def _grid5_subset(rng, size=5):
    """Pick 16 points from a 5x5 grid (indices 0..4)."""
    all_pts = [[x, y] for x in range(size) for y in range(size)]
    rng.shuffle(all_pts)
    return all_pts[:16]

def _grid6x5_subset(rng):
    """Pick 16 points from a 6x5 grid (indices 0..5 x 0..4)."""
    all_pts = [[x, y] for x in range(6) for y in range(5)]
    rng.shuffle(all_pts)
    return all_pts[:16]

def _grid7x4_subset(rng):
    """Pick 16 points from a 7x4 grid."""
    all_pts = [[x, y] for x in range(7) for y in range(4)]
    rng.shuffle(all_pts)
    return all_pts[:16]

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
    num_starts = 40
    max_iter = 6000
    
    for s in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initialization strategies
        if s == 0:
            pts = [list(p) for p in best_pts]
        elif s == 1:
            # Perturbed 4x4 grid
            pts = [list(p) for p in _initial_config()]
            for _ in range(rng.randint(2, 5)):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-3, 3), pts[idx][1] + rng.randint(-3, 3)]
        elif s == 2:
            # 5x5 grid subset
            pts = _grid5_subset(rng, 5)
        elif s == 3:
            # 6x5 grid subset
            pts = _grid6x5_subset(rng)
        elif s == 4:
            # 7x4 grid subset
            pts = _grid7x4_subset(rng)
        elif s == 5:
            # Random initialization
            pts = [[rng.randint(-15, 15), rng.randint(-15, 15)] for _ in range(16)]
        elif s == 6:
            # Perturb best more aggressively
            pts = [list(p) for p in best_pts]
            for _ in range(rng.randint(3, 7)):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-5, 5), pts[idx][1] + rng.randint(-5, 5)]
        elif s == 7:
            # Another 5x5 subset with different seed
            pts = _grid5_subset(rng, 5)
        elif s == 8:
            # Another 6x5 subset
            pts = _grid6x5_subset(rng)
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
        
        # Simulated annealing with line-aware moves
        T = 3.0
        cooling = 0.9995
        line_info = None
        line_info_timer = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            # Periodically refresh line info
            line_info_timer += 1
            if line_info_timer >= 50 or line_info is None:
                line_info = _get_line_info(pts)
                line_info_timer = 0
            
            # Move: relocate one point
            idx = rng.randint(0, 15)
            old = list(pts[idx])
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.40:
                # Line-extension move: try to extend a 3-point line to 4
                # Find lines with exactly 3 points, pick one, extend it
                three_lines = [(k, v) for k, v in line_info.items() if len(v) == 3]
                if three_lines:
                    lk, lpts = rng.choice(three_lines)
                    ext = _find_line_extension(pts, lk, lpts)
                    if ext:
                        new_pt = rng.choice(ext)
                        # Check it doesn't create a 5-line
                        pts[idx] = new_pt
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
                            if delta_s < 0:
                                accept_prob = math.exp(delta_s / max(T, 0.01))
                                if rng.random() > accept_prob:
                                    pts[idx] = old
                            # else accept
                        continue
                # Fallback to local move
                delta = rng.randint(-3, 3)
                delta2 = rng.randint(-3, 3)
                if delta == 0 and delta2 == 0:
                    delta = 1
                new_x = old[0] + delta
                new_y = old[1] + delta2
            
            elif move_type < 0.60:
                # Local move
                delta = rng.randint(-3, 3)
                delta2 = rng.randint(-3, 3)
                if delta == 0 and delta2 == 0:
                    delta = 1
                new_x = old[0] + delta
                new_y = old[1] + delta2
            elif move_type < 0.80:
                # Move near another point (to potentially form lines)
                ref_idx = rng.randint(0, 15)
                if ref_idx == idx:
                    ref_idx = (ref_idx + 1) % 16
                new_x = pts[ref_idx][0] + rng.randint(-4, 4)
                new_y = pts[ref_idx][1] + rng.randint(-4, 4)
            else:
                # Global jump
                new_x = rng.randint(-20, 20)
                new_y = rng.randint(-20, 20)
            
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
        line_info = _get_line_info(pts)
        line_info_timer = 0
        for it in range(6000):
            if time.time() > deadline - 1.0:
                break
            line_info_timer += 1
            if line_info_timer >= 40:
                line_info = _get_line_info(pts)
                line_info_timer = 0
            
            idx = rng.randint(0, 15)
            old = list(pts[idx])
            
            # Try line extension first
            three_lines = [(k, v) for k, v in line_info.items() if len(v) == 3]
            if three_lines and rng.random() < 0.5:
                lk, lpts = rng.choice(three_lines)
                ext = _find_line_extension(pts, lk, lpts)
                if ext:
                    new_pt = rng.choice(ext)
                    pts[idx] = new_pt
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
                    continue
            
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
