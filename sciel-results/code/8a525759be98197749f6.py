import time
import random
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _evaluate(points):
    """
    Returns (is_valid, score, line_counts).
    score: Number of lines containing exactly 4 points.
    line_counts: dict mapping line_key to count of points.
    """
    n = len(points)
    line_pair_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            key = _line_key(x1, y1, points[j][0], points[j][1])
            if key is not None:
                line_pair_counts[key] += 1
            
    valid = True
    score = 0
    line_counts = {}
    
    for key, count in line_pair_counts.items():
        k = -1
        # Solve k(k-1)/2 = count for k
        for possible_k in range(2, 17):
            if possible_k * (possible_k - 1) // 2 == count:
                k = possible_k
                break
        if k == -1:
            continue
        line_counts[key] = k
        if k >= 5:
            valid = False
            pass
        
    if not valid:
        return False, 0, line_counts
        
    for k in line_counts.values():
        if k == 4:
            score += 1
            
    return True, score, line_counts

def _fast_evaluate(points):
    """
    Fast evaluation using a set of points for O(1) lookups.
    Returns (is_valid, score, line_counts).
    """
    n = len(points)
    point_set = set(points)
    line_pair_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            key = _line_key(x1, y1, points[j][0], points[j][1])
            if key is not None:
                line_pair_counts[key] += 1
                
    valid = True
    score = 0
    line_counts = {}
    
    for key, count in line_pair_counts.items():
        k = -1
        for possible_k in range(2, 17):
            if possible_k * (possible_k - 1) // 2 == count:
                k = possible_k
                break
        if k == -1:
            continue
        line_counts[key] = k
        if k >= 5:
            valid = False
            
    if not valid:
        return False, 0, line_counts
        
    for k in line_counts.values():
        if k == 4:
            score += 1
            
    return True, score, line_counts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # --- Helper: Generate a pool of candidate points ---
    # We want points that are likely to form lines.
    # A grid is a good source of such points.
    # We will generate a large grid and sample from it.
    # However, a simple grid has many 5+ collinear points.
    # We need to be careful.
    
    # Strategy:
    # 1. Generate a pool of points from a perturbed grid or random distribution.
    # 2. Use a local search (Simulated Annealing) with a specific move set:
    #    - Replace a point with a new point from the pool.
    #    - Jitter a point.
    # 3. The pool should be large enough to allow exploration but small enough for fast membership checks.
    
    # Let's create a pool of points.
    # A 50x50 grid centered at 0? That's 2500 points.
    # Or a set of points on a "lattice" with some randomness.
    
    # Actually, the most effective pool for this problem is often the integer points within a certain radius,
    # but that's huge.
    # Let's stick to a moderate grid, say 40x40, centered at 0.
    # Points: (x, y) where -20 <= x, y <= 19.
    # Total 1600 points.
    
    pool_size = 40
    pool = set()
    for x in range(-pool_size // 2, pool_size // 2):
        for y in range(-pool_size // 2, pool_size // 2):
            pool.add((x, y))
    
    # Add some random points outside the grid to break symmetry?
    # Maybe not necessary initially.
    
    # --- Local Search ---
    
    def _sa_search(initial_points, max_time):
        current = [list(p) for p in initial_points]
        c_valid, c_score, c_lines = _fast_evaluate(current)
        if not c_valid:
            c_score = -1000
            
        best = [list(p) for p in current]
        best_score = c_score
        
        temp = 5.0
        cooling = 0.999
        no_improve = 0
        max_no_improve = 5000
        
        # Precompute set of current points for fast duplicate check
        current_set = set(tuple(p) for p in current)
        
        while time.time() < deadline - 2.0 and no_improve < max_no_improve:
            cand = [list(p) for p in current]
            move_type = random.random()
            
            if move_type < 0.6:
                # Replace a point with a random point from the pool
                idx = random.randint(0, 15)
                # Pick a random point from pool
                # To avoid O(N) iteration, we can use a list of pool
                # But pool is a set. Let's make a list.
                # Actually, picking from a set is O(1) in Python 3.7+? No, random.choice needs a sequence.
                # Let's keep a list of pool points.
                pool_list = list(pool)
                new_pt = random.choice(pool_list)
                new_x, new_y = new_pt
                
                # Check bounds
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                
                # Check duplicate
                # Remove current point from set, check if new point is in set
                old_pt = tuple(cand[idx])
                current_set.discard(old_pt)
                
                if new_pt in current_set:
                    current_set.add(old_pt)
                    continue
                
                cand[idx] = [new_x, new_y]
                current_set.add(new_pt)
                
            elif move_type < 0.8:
                # Jitter a point
                idx = random.randint(0, 15)
                old_x, old_y = cand[idx]
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                
                new_pt = (new_x, new_y)
                old_pt = tuple(cand[idx])
                
                if new_pt == old_pt:
                    continue
                    
                current_set.discard(old_pt)
                
                if new_pt in current_set:
                    current_set.add(old_pt)
                    continue
                
                cand[idx] = [new_x, new_y]
                current_set.add(new_pt)
                
            else:
                # Swap two points with random pool points?
                # Or just 2-point jitter
                i1, i2 = random.sample(range(16), 2)
                old1 = tuple(cand[i1])
                old2 = tuple(cand[i2])
                
                # Try to move both
                new1 = (old1[0] + random.randint(-5, 5), old1[1] + random.randint(-5, 5))
                new2 = (old2[0] + random.randint(-5, 5), old2[1] + random.randint(-5, 5))
                
                if abs(new1[0]) > 1000 or abs(new1[1]) > 1000 or abs(new2[0]) > 1000 or abs(new2[1]) > 1000:
                    continue
                    
                if new1 == old1 or new2 == old2:
                    continue
                    
                current_set.discard(old1)
                current_set.discard(old2)
                
                if new1 in current_set or new2 in current_set or new1 == new2:
                    current_set.add(old1)
                    current_set.add(old2)
                    continue
                
                cand[i1] = [new1[0], new1[1]]
                cand[i2] = [new2[0], new2[1]]
                current_set.add(new1)
                current_set.add(new2)
                
            cand_valid, cand_score, cand_lines = _fast_evaluate(cand)
            if not cand_valid:
                cand_score = -1000
            
            if cand_score > best_score:
                best = [list(p) for p in cand]
                best_score = cand_score
                no_improve = 0
            else:
                no_improve += 1
                
            # SA Acceptance
            if cand_score > c_score:
                current = cand
                c_score = cand_score
                c_lines = cand_lines
            else:
                delta = cand_score - c_score
                if delta < 0:
                    prob = pow(2.718281828, delta / max(temp, 0.01))
                    if random.random() < prob:
                        current = cand
                        c_score = cand_score
                        c_lines = cand_lines
                elif delta == 0 and random.random() < 0.1:
                    current = cand
                    c_score = cand_score
                    c_lines = cand_lines
                    
            temp *= cooling
            
        return best, best_score

    # Initializations
    inits = []
    
    # 1. 4x4 Grid
    grid4 = [[x, y] for x in range(4) for y in range(4)]
    inits.append(grid4)
    
    # 2. 4x4 Grid shifted
    grid4_shift = [[x+1, y+1] for x in range(4) for y in range(4)]
    inits.append(grid4_shift)
    
    # 3. Random subset from pool
    random.seed(42)
    pool_list = list(pool)
    random.shuffle(pool_list)
    inits.append([list(p) for p in pool_list[:16]])
    
    # 4. Another random subset
    random.seed(123)
    random.shuffle(pool_list)
    inits.append([list(p) for p in pool_list[:16]])
    
    # 5. 5x5 Grid Subset (Remove 9 points)
    grid5 = [[x, y] for x in range(-2, 3) for y in range(-2, 3)]
    random.seed(999)
    random.shuffle(grid5)
    subset5 = [list(p) for p in grid5[:16]]
    inits.append(subset5)
    
    best_overall = None
    best_score_overall = -1
    
    # Run SA for each init
    num_inits = len(inits)
    time_per_init = (deadline - time.time() - 10.0) / num_inits
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
        alloc_time = min(time_per_init, deadline - time.time() - 5.0)
        if alloc_time < 1.0:
            continue
            
        res_pts, res_score = _sa_search(init, alloc_time)
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_overall = res_pts
            
    # Final polish with remaining time
    if best_overall and time.time() < deadline - 5:
        res_pts, res_score = _sa_search(best_overall, deadline - time.time())
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_overall = res_pts
            
    if best_overall is None:
        return grid4
        
    return best_overall
