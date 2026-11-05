import time
import random
from itertools import combinations

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
            
    return sum(1 for pts in lines.values() if len(pts) == 2)

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
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            if score == 6: # Theoretical minimum
                return True
        return False

    # Try specific known good configurations
    # Config 1: Symmetric cross
    pts1 = [
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2),
        (2, 1), (2, -1), (-2, 1), (-2, -1)
    ]
    if try_update(pts1):
        return best_points

    # Config 2: 3x3 Grid + 3
    pts2 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (0,3), (3,0), (3,3)
    ]
    if try_update(pts2):
        return best_points

    # Local Search
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random start
        pts = []
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in pts:
                pts.append((x, y))
        
        current = [list(p) for p in pts]
        current_score = count_ordinary_lines(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for idx in range(12):
                if time.time() > deadline:
                    break
                    
                old_x, old_y = current[idx]
                other_idx = [i for i in range(12) if i != idx]
                
                # Collect lines from other points
                lines_dict = {}
                for a in other_idx:
                    for b in other_idx:
                        if a >= b: continue
                        x1, y1 = current[a]
                        x2, y2 = current[b]
                        A = y1 - y2
                        B = x2 - x1
                        C = x1 * y2 - x2 * y1
                        if A == 0 and B == 0: continue
                        if A < 0 or (A == 0 and B < 0):
                            A, B, C = -A, -B, -C
                        key = (A, B, C)
                        if key not in lines_dict:
                            lines_dict[key] = set()
                        lines_dict[key].add(a)
                        lines_dict[key].add(b)
                        
                line_keys = list(lines_dict.keys())
                random.shuffle(line_keys)
                
                # Try moving to intersection
                for l1 in line_keys[:20]:
                    if time.time() > deadline: break
                    A1, B1, C1 = l1
                    for l2 in line_keys[:20]:
                        if l1 == l2: continue
                        A2, B2, C2 = l2
                        det = A1 * B2 - A2 * B1
                        if det == 0: continue
                        x_num = B1 * C2 - B2 * C1
                        y_num = A2 * C1 - A1 * C2
                        if x_num % det != 0 or y_num % det != 0: continue
                        new_x = x_num // det
                        new_y = y_num // det
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9: continue
                        if (new_x, new_y) == (old_x, old_y): continue
                        
                        # Check collision
                        collision = False
                        for j in range(12):
                            if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                                collision = True
                                break
                        if collision: continue
                        
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if try_update(current):
                                return best_points
                            break
                        else:
                            current[idx] = [old_x, old_y]
                    
                    if improved:
                        break
                if improved:
                    break
            
            # Random kick
            if not improved and random.random() < 0.2:
                idx = random.randint(0, 11)
                old_x, old_y = current[idx]
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
                if (new_x, new_y) != (old_x, old_y) and abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    collision = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            collision = True
                            break
                    if not collision:
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if try_update(current):
                                return best_points
                        else:
                            current[idx] = [old_x, old_y]

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
