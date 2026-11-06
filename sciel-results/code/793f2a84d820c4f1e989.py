"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    timeout = 130  # Leave 20s margin for evaluator overhead
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def count_3_collinear(points):
        """Count number of collinear triples."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count
    
    def has_4_collinear(points):
        """Check if any 4 points are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                # Get all points collinear with i and j
                collinear = []
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        collinear.append(k)
                if len(collinear) >= 2:
                    return True
        return False
    
    def max_no_3_collinear(points):
        """Find the largest subset with no three collinear using greedy + local search."""
        n = len(points)
        if n <= 2:
            return n
        
        # Greedy: add points one by one if they don't create a collinear triple
        selected = []
        for i in range(n):
            can_add = True
            for j in range(len(selected)):
                for k in range(j+1, len(selected)):
                    if are_collinear(points[i], points[selected[j]], points[selected[k]]):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                selected.append(i)
        
        # Local search: try to improve by swapping
        improved = True
        while improved and (time.time() - start_time) < timeout - 5:
            improved = False
            current_size = len(selected)
            selected_set = set(selected)
            
            for i in range(n):
                if i in selected_set:
                    continue
                # Try adding i and removing one point that conflicts
                for j in selected:
                    # Check if i conflicts with any pair in selected excluding j
                    conflicts = False
                    for k in selected:
                        if k == j:
                            continue
                        for l in selected:
                            if l == j or l == k:
                                continue
                            if are_collinear(points[i], points[k], points[l]):
                                conflicts = True
                                break
                        if conflicts:
                            break
                    if not conflicts:
                        new_selected = [x for x in selected if x != j] + [i]
                        # Verify no three collinear in new set
                        valid = True
                        for a in range(len(new_selected)):
                            for b in range(a+1, len(new_selected)):
                                for c in range(b+1, len(new_selected)):
                                    if are_collinear(points[new_selected[a]], points[new_selected[b]], points[new_selected[c]]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid and len(new_selected) > current_size:
                            selected = new_selected
                            current_size = len(selected)
                            improved = True
                            break
                if improved:
                    break
        
        return len(selected)
    
    def score_config(points):
        """Compute the score: largest subset with no three collinear."""
        return max_no_3_collinear(points)
    
    # Generate initial configurations with many collinear triples
    # Strategy: place points on a few lines to create many collinear triples,
    # but ensure no 4 are collinear (max 3 per line)
    
    best_score = 20
    best_points = None
    
    # Predefined configurations with controlled collinearities
    # Each line has at most 3 points
    
    configs = []
    
    # Config 1: 7 lines with 3 points each (21 points, use 20)
    # Lines: y = 0, y = 1, y = 2, y = 3, y = 4, y = 5, y = 6
    # But need to avoid 4 collinear, so max 3 per horizontal line
    # 7 lines * 3 = 21, take 20
    
    # Use horizontal lines with 3 points each, spread out
    for line_y in range(0, 7):
        for x_offset in range(-2, 1):
            x = x_offset * 30 + line_y * 5
            y = line_y * 15
            configs.append([x, y])
    
    # Take first 20
    config1 = configs[:20]
    if not has_4_collinear(config1):
        s = score_config(config1)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in config1]
    
    # Config 2: Mix of horizontal and vertical lines
    configs2 = []
    # 5 horizontal lines with 3 points
    for line_y in [0, 20, 40, 60, 80]:
        for x in [-30, 0, 30]:
            configs2.append([x, line_y])
    # 5 vertical lines with 3 points
    for line_x in [-40, -20, 20, 40, 60]:
        for y in [-10, 10, 30]:
            configs2.append([line_x, y])
    
    config2 = configs2[:20]
    if not has_4_collinear(config2):
        s = score_config(config2)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in config2]
    
    # Config 3: Diagonal lines
    configs3 = []
    # Lines with slope 1
    for c in [-30, -10, 10, 30, 50]:
        for t in range(3):
            x = t * 20 + c
            y = t * 20
            configs3.append([x, y])
    # Lines with slope -1
    for c in [10, 30, 50, 70, 90]:
        for t in range(3):
            x = t * 20
            y = c - t * 20
            configs3.append([x, y])
    
    config3 = configs3[:20]
    if not has_4_collinear(config3):
        s = score_config(config3)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in config3]
    
    # Random search with local optimization
    random.seed(42)
    attempts = 0
    while time.time() - start_time < timeout - 10 and attempts < 500:
        attempts += 1
        
        # Generate random points
        points = []
        used = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                points.append([x, y])
        
        if has_4_collinear(points):
            continue
        
        s = score_config(points)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in points]
            
            # Local search: try to improve by moving points
            improved = True
            while improved and time.time() - start_time < timeout - 10:
                improved = False
                for i in range(20):
                    for _ in range(10):
                        old_x, old_y = points[i]
                        new_x = random.randint(max(-100, old_x - 10), min(100, old_x + 10))
                        new_y = random.randint(max(-100, old_y - 10), min(100, old_y + 10))
                        if (new_x, new_y) == (old_x, old_y):
                            continue
                        if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(points) if j != i):
                            continue
                        points[i] = [new_x, new_y]
                        if has_4_collinear(points):
                            points[i] = [old_x, old_y]
                            continue
                        new_s = score_config(points)
                        if new_s < s:
                            s = new_s
                            if s < best_score:
                                best_score = s
                                best_points = [p[:] for p in points]
                            improved = True
                            break
                        else:
                            points[i] = [old_x, old_y]
                    if improved:
                        break
    
    if best_points is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
# EVOLVE-BLOCK-END
