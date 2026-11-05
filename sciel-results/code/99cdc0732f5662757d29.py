import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115.0

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def evaluate(points):
        n = len(points)
        line_counts = {}
        
        # Check distinctness
        pts_set = set(points)
        if len(pts_set) != n:
            return -1000, False
            
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key:
                    if key not in line_counts:
                        line_counts[key] = set()
                    line_counts[key].add(i)
                    line_counts[key].add(j)
        
        score = 0
        for key, pts_on_line in line_counts.items():
            count = len(pts_on_line)
            if count == 4:
                score += 1
            elif count >= 5:
                return -1000, False
                
        return score, True

    def find_3_point_lines(points):
        n = len(points)
        line_pts = {}
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key:
                    if key not in line_pts:
                        line_pts[key] = set()
                    line_pts[key].add(i)
                    line_pts[key].add(j)
        
        three_lines = []
        for key, pts in line_pts.items():
            if len(pts) == 3:
                three_lines.append((key, pts))
        return three_lines

    def try_complete_line(points, key, pts_on_line, current_score):
        dx, dy, c = key
        
        candidates = []
        if dx != 0:
            xs = [p[0] for p in points]
            min_x, max_x = min(xs), max(xs)
            search_min_x = max(-1000, min_x - 50)
            search_max_x = min(1000, max_x + 50)
            
            for x in range(search_min_x, search_max_x + 1):
                num = dy * x - c
                if num % dx == 0:
                    y = num // dx
                    if -1000 <= y <= 1000:
                        candidates.append((x, y))
        else:
            x = c
            if -1000 <= x <= 1000:
                occupied_y = set(p[1] for p in points if p[0] == x)
                ys = [p[1] for p in points]
                min_y, max_y = min(ys), max(ys)
                for y in range(max(-1000, min_y - 50), min(1000, max_y + 50) + 1):
                    if y not in occupied_y:
                        candidates.append((x, y))

        if not candidates:
            return None
            
        random.shuffle(candidates)
        
        for target_pt in candidates:
            if target_pt in points:
                continue
                
            pts_list = list(pts_on_line)
            random.shuffle(pts_list)
            
            for idx in pts_list:
                old_pt = points[idx]
                points[idx] = target_pt
                
                new_score, valid = evaluate(points)
                
                if valid and new_score > current_score:
                    return idx, target_pt, new_score
                else:
                    points[idx] = old_pt
                    
        return None

    def local_search(initial_points):
        points = [p[:] for p in initial_points]
        score, valid = evaluate(points)
        if not valid:
            return None, -1
            
        best_score = score
        best_points = [p[:] for p in points]
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() - start_time < time_limit:
            if no_improve > max_no_improve:
                # Kick
                idx = random.randint(0, 15)
                points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                score, valid = evaluate(points)
                if not valid:
                    points[idx] = best_points[idx]
                    score = best_score
                    no_improve = 0
                    continue
                no_improve = 0
                continue
                
            three_lines = find_3_point_lines(points)
            
            if three_lines:
                key, pts_on_line = random.choice(three_lines)
                result = try_complete_line(points, key, pts_on_line, score)
                if result:
                    idx, new_pt, new_score = result
                    points[idx] = new_pt
                    score = new_score
                    no_improve = 0
                    if score > best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                else:
                    no_improve += 1
            else:
                # No 3-point lines. Random perturbation.
                idx = random.randint(0, 15)
                old_pt = points[idx]
                points[idx] = [old_pt[0] + random.randint(-3, 3), old_pt[1] + random.randint(-3, 3)]
                new_score, valid = evaluate(points)
                if valid and new_score > score:
                    score = new_score
                    no_improve = 0
                    if score > best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                else:
                    points[idx] = old_pt
                    no_improve += 1
                    
        return best_points, best_score

    # Initializations
    inits = []
    grid = [[x, y] for x in range(4) for y in range(4)]
    inits.append(grid)
    
    for _ in range(10):
        rand_pts = [[random.randint(-30, 30), random.randint(-30, 30)] for _ in range(16)]
        if len(set(tuple(p) for p in rand_pts)) == 16:
            inits.append(rand_pts)
            
    sheared = [[x+y, y] for x in range(4) for y in range(4)]
    if len(set(tuple(p) for p in sheared)) == 16:
        inits.append(sheared)
        
    # Another sheared version
    sheared2 = [[x, x+y] for x in range(4) for y in range(4)]
    if len(set(tuple(p) for p in sheared2)) == 16:
        inits.append(sheared2)

    best_overall = None
    best_score_overall = -1
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        pts, sc = local_search(init)
        if pts and sc > best_score_overall:
            best_score_overall = sc
            best_overall = pts
            
    if best_overall is None:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_overall
