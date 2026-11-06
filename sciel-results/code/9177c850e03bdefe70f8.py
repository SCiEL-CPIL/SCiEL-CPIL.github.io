import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds
    
    def count_on_line(points, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        cnt = 0
        for p in points:
            x, y = p
            # Cross product check
            if dx * (y - y1) == dy * (x - x1):
                cnt += 1
        return cnt

    def evaluate(points):
        n = len(points)
        raw = 0
        for i in range(n):
            for j in range(i+1, n):
                c = count_on_line(points, points[i], points[j])
                if c >= 5:
                    return -1000
                if c == 4:
                    raw += 1
        return raw // 6

    # Initial solution: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = evaluate(best_points) # Should be 10
    
    # Local Search
    def local_search(points, max_iters, T_start, T_end):
        curr_points = [p[:] for p in points]
        curr_score = evaluate(curr_points)
        best_local = [p[:] for p in curr_points]
        best_local_score = curr_score
        
        for i in range(max_iters):
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Pick random point and coordinate
            idx = random.randint(0, 15)
            coord = random.randint(0, 1)
            delta = random.choice([-2, -1, 1, 2])
            
            new_points = [p[:] for p in curr_points]
            new_points[idx][coord] += delta
            
            # Check bounds
            if abs(new_points[idx][0]) > 1000 or abs(new_points[idx][1]) > 1000:
                continue
                
            new_score = evaluate(new_points)
            
            if new_score >= curr_score:
                curr_points = new_points
                curr_score = new_score
                if curr_score > best_local_score:
                    best_local = [p[:] for p in curr_points]
                    best_local_score = curr_score
            else:
                # Accept with probability
                delta_e = curr_score - new_score
                if delta_e == 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta_e / T) if T > 0 else 0
                    accept = random.random() < accept_prob
                if accept:
                    curr_points = new_points
                    curr_score = new_score
                    
        return best_local, best_local_score

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Try grid based perturbation
        if restarts % 2 == 0:
            start_pts = [[x, y] for x in range(4) for y in range(4)]
            # Perturb slightly
            for _ in range(5):
                idx = random.randint(0, 15)
                coord = random.randint(0, 1)
                delta = random.choice([-2, -1, 1, 2])
                start_pts[idx][coord] += delta
        else:
            # Random start in small box
            start_pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
            # Ensure distinct
            while len(set(tuple(p) for p in start_pts)) < 16:
                idx = random.randint(0, 15)
                start_pts[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        
        res_pts, res_score = local_search(start_pts, 2000, 1.0, 0.01)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
            
        restarts += 1
        
    return best_points
