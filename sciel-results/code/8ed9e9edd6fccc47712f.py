import itertools
import time
import random
from collections import Counter
from math import gcd

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def compute_line_counts(points):
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts

def evaluate_score(line_counts):
    four_count = 0
    five_or_more = 0
    
    for key, count in line_counts.items():
        # The count in the dictionary is the number of PAIRS of points on that line.
        # If there are k points on a line, the number of pairs is k*(k-1)/2.
        # So we need to map pair_count back to point_count.
        # k=4 -> 6 pairs
        # k=5 -> 10 pairs
        # k=3 -> 3 pairs
        # k=2 -> 1 pair
        
        if count == 6:
            four_count += 1
        elif count >= 10:
            five_or_more += 1
            
    return 1000 * four_count - 10000 * five_or_more

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_score = -1000000
    
    # Initial configurations to try
    initial_configs = []
    
    # 1. 4x4 Grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 2. Perturbed 4x4 Grid
    random.seed(1)
    perturbed = [[x + random.randint(-2, 2), y + random.randint(-2, 2)] for x in range(4) for y in range(4)]
    # Ensure distinct
    seen = set()
    valid_perturbed = []
    for p in perturbed:
        if p not in seen:
            seen.add(p)
            valid_perturbed.append(p)
    while len(valid_perturbed) < 16:
        valid_perturbed.append([random.randint(-10, 10), random.randint(-10, 10)])
    # Fix duplicates
    seen = set()
    unique_perturbed = []
    for p in valid_perturbed:
        if p not in seen:
            seen.add(p)
            unique_perturbed.append(p)
    initial_configs.append(unique_perturbed)

    # 3. Random points
    random.seed(2)
    random_pts = []
    seen = set()
    while len(random_pts) < 16:
        p = [random.randint(-50, 50), random.randint(-50, 50)]
        if p not in seen:
            seen.add(p)
            random_pts.append(p)
    initial_configs.append(random_pts)

    for init_idx, points in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
            
        # Ensure points are distinct and within bounds
        seen = set()
        valid_points = []
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                p = [0, 0]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                valid_points.append(p)
        while len(valid_points) < 16:
            valid_points.append([random.randint(-1000, 1000), random.randint(-1000, 1000)])
        points = [p[:] for p in valid_points[:16]]
        
        line_counts = compute_line_counts(points)
        current_score = evaluate_score(line_counts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in points]
            
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.999
        
        random.seed(42 + init_idx)
        
        while T > T_min:
            if time.time() > deadline - 5:
                break
                
            idx = random.randint(0, 15)
            old_point = points[idx][:]
            
            # Move generation
            if random.random() < 0.2:
                # Large jump
                dx = random.randint(-100, 100)
                dy = random.randint(-100, 100)
            else:
                # Small jump
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                
            new_x = old_point[0] + dx
            new_y = old_point[1] + dy
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                T *= cooling_rate
                continue
                
            new_point = [new_x, new_y]
            
            # Check duplicates
            is_dup = False
            for i in range(16):
                if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                    is_dup = True
                    break
            if is_dup:
                T *= cooling_rate
                continue

            # Incremental update
            for j in range(16):
                if j == idx:
                    continue
                key = get_line_key(old_point, points[j])
                if key:
                    line_counts[key] -= 1
                    if line_counts[key] == 0:
                        del line_counts[key]
                        
            for j in range(16):
                if j == idx:
                    continue
                key = get_line_key(new_point, points[j])
                if key:
                    line_counts[key] = line_counts.get(key, 0) + 1
                    
            points[idx] = new_point
            
            new_score = evaluate_score(line_counts)
            
            if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
            else:
                # Revert
                for j in range(16):
                    if j == idx:
                        continue
                    key = get_line_key(new_point, points[j])
                    if key:
                        line_counts[key] -= 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                            
                for j in range(16):
                    if j == idx:
                        continue
                    key = get_line_key(old_point, points[j])
                    if key:
                        line_counts[key] = line_counts.get(key, 0) + 1
                            
                points[idx] = old_point
            
            T *= cooling_rate

    # Final validation and return
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(4)]
        
    # Verify no 5 collinear
    line_counts_best = compute_line_counts(best_points)
    for key, count in line_counts_best.items():
        if count >= 10:
            return [[x, y] for x in range(4) for y in range(4)]
            
    return best_points
