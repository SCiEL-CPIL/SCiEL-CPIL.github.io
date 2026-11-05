import random
import time
from itertools import combinations
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def count_ordinary_lines_fast(points):
    """Faster counting using a different approach."""
    n = len(points)
    if n < 3:
        return 0
    
    # Count lines through each pair, track which lines have exactly 2 points
    line_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    return sum(1 for pts in line_points.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_score = float('inf')
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return
        if len(set(map(tuple, pts))) != 12:
            return
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return
        score = count_ordinary_lines_fast(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
    
    # Generate candidate points in a bounded region
    # Try multiple random constructions with greedy + local search
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random seed
        pts = []
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in pts:
                pts.append((x, y))
        
        # Local search: try to improve by moving points to intersections
        current = [list(p) for p in pts]
        current_score = count_ordinary_lines_fast(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try moving each point to intersection of two lines
            for idx in range(12):
                if time.time() > deadline:
                    break
                    
                old_x, old_y = current[idx]
                
                # Get all lines through pairs of other points
                other_idx = [i for i in range(12) if i != idx]
                lines_dict = {}
                
                for a_pos in range(len(other_idx)):
                    for b_pos in range(a_pos + 1, len(other_idx)):
                        a = other_idx[a_pos]
                        b = other_idx[b_pos]
                        x1, y1 = current[a]
                        x2, y2 = current[b]
                        
                        A = y1 - y2
                        B = x2 - x1
                        C = x1 * y2 - x2 * y1
                        
                        if A == 0 and B == 0:
                            continue
                        
                        if A < 0 or (A == 0 and B < 0):
                            A, B, C = -A, -B, -C
                        
                        key = (A, B, C)
                        if key not in lines_dict:
                            lines_dict[key] = set()
                        lines_dict[key].add(a)
                        lines_dict[key].add(b)
                
                # Try intersections of pairs of lines
                line_keys = list(lines_dict.keys())
                random.shuffle(line_keys)
                
                for l1_pos in range(min(50, len(line_keys))):
                    if time.time() > deadline:
                        break
                    l1 = line_keys[l1_pos]
                    A1, B1, C1 = l1
                    
                    for l2_pos in range(l1_pos + 1, min(50, len(line_keys))):
                        l2 = line_keys[l2_pos]
                        A2, B2, C2 = l2
                        
                        det = A1 * B2 - A2 * B1
                        if det == 0:
                            continue
                        
                        x_num = B1 * C2 - B2 * C1
                        y_num = A2 * C1 - A1 * C2
                        
                        if x_num % det != 0 or y_num % det != 0:
                            continue
                        
                        new_x = x_num // det
                        new_y = y_num // det
                        
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        
                        if (new_x, new_y) == (old_x, old_y):
                            continue
                        
                        # Check not colliding with other points
                        collision = False
                        for j in range(12):
                            if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                                collision = True
                                break
                        if collision:
                            continue
                        
                        # Try this move
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines_fast(current)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            try_update(current)
                            break
                        else:
                            current[idx] = [old_x, old_y]
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            # If no improvement, try random perturbation
            if not improved and random.random() < 0.3:
                idx = random.randint(0, 11)
                old_x, old_y = current[idx]
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                
                if (new_x, new_y) != (old_x, old_y) and abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    collision = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            collision = True
                            break
                    if not collision:
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines_fast(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            try_update(current)
                        else:
                            current[idx] = [old_x, old_y]
        
        try_update(current)
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
