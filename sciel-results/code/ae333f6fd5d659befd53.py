import time
import random
import math
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def score_points(points):
        """Count lines with exactly 4 points, penalize 5+."""
        line_points = defaultdict(set)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_points[key].add(i)
                line_points[key].add(j)
        
        num_4 = 0
        num_5_plus = 0
        for pts in line_points.values():
            k = len(pts)
            if k == 4:
                num_4 += 1
            elif k >= 5:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate(n4, n5):
        return n4 - 10000 * n5

    def initial_grid():
        return [[i, j] for i in range(4) for j in range(4)]

    def random_config(box_size=30):
        pts = []
        used = set()
        for _ in range(16):
            for _attempt in range(200):
                x = random.randint(-box_size, box_size)
                y = random.randint(-box_size, box_size)
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
                    break
            else:
                while (x, y) in used:
                    x = random.randint(-box_size, box_size)
                    y = random.randint(-box_size, box_size)
                pts.append([x, y])
                used.add((x, y))
        return pts

    def structured_config():
        """Try a few structured patterns."""
        patterns = []
        # 4x4 grid centered
        patterns.append([[i-1, j-1] for i in range(4) for j in range(4)])
        # 4x4 grid with offset
        patterns.append([[i, j+2] for i in range(4) for j in range(4)])
        # Two 2x4 rectangles
        patterns.append([[i, j] for i in range(2) for j in range(4)] + 
                       [[i+3, j] for i in range(2) for j in range(4)])
        # 4 columns of 4, spaced out
        patterns.append([[i*3, j] for i in range(4) for j in range(4)])
        return random.choice(patterns)

    def sa_search(initial_pts, time_budget):
        """Simulated annealing with line-completion moves."""
        points = [p[:] for p in initial_pts]
        n4, n5 = score_points(points)
        current_score = evaluate(n4, n5)
        best_score = current_score
        best_points = [p[:] for p in points]
        
        T = 10.0
        min_T = 0.5
        cooling = 0.995
        
        end_time = time.time() + time_budget
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Random jitter: move one point nearby
                idx = random.randint(0, 15)
                ox, oy = points[idx]
                step = random.randint(1, 8)
                nx = ox + random.randint(-step, step)
                ny = oy + random.randint(-step, step)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                    continue
                old_pos = points[idx][:]
                points[idx] = [nx, ny]
                new_n4, new_n5 = score_points(points)
                new_score = evaluate(new_n4, new_n5)
                
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / T):
                        current_score = new_score
                    else:
                        points[idx] = old_pos
                        
            elif move_type < 0.7:
                # Line completion: find a 3-point line and try to add a 4th
                # Build line structure
                line_points = defaultdict(set)
                for i in range(16):
                    for j in range(i + 1, 16):
                        key = canonical_line(points[i], points[j])
                        line_points[key].add(i)
                        line_points[key].add(j)
                
                # Find lines with exactly 3 points
                three_pt_lines = []
                for key, pts in line_points.items():
                    if len(pts) == 3:
                        three_pt_lines.append((key, pts))
                
                if three_pt_lines:
                    key, existing_pts = random.choice(three_pt_lines)
                    a, b, c = key
                    # Find a point NOT on this line
                    candidates = [k for k in range(16) if k not in existing_pts]
                    if candidates:
                        idx = random.choice(candidates)
                        # Find a valid integer position on the line
                        found = False
                        for _ in range(20):
                            if b == 0:
                                # Vertical: x = c/a
                                if a != 0 and c % a == 0:
                                    nx = c // a
                                    ny = random.randint(-1000, 1000)
                                else:
                                    continue
                            else:
                                nx = random.randint(-1000, 1000)
                                val = c - a * nx
                                if val % b != 0:
                                    continue
                                ny = val // b
                            
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                                continue
                            
                            old_pos = points[idx][:]
                            points[idx] = [nx, ny]
                            new_n4, new_n5 = score_points(points)
                            new_score = evaluate(new_n4, new_n5)
                            
                            if new_score > current_score:
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best_points = [p[:] for p in points]
                            else:
                                delta = new_score - current_score
                                if random.random() < math.exp(delta / T):
                                    current_score = new_score
                                else:
                                    points[idx] = old_pos
                            found = True
                            break
                        
            else:
                # Global move: relocate a point to a random position
                idx = random.randint(0, 15)
                nx = random.randint(-50, 50)
                ny = random.randint(-50, 50)
                if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                    continue
                old_pos = points[idx][:]
                points[idx] = [nx, ny]
                new_n4, new_n5 = score_points(points)
                new_score = evaluate(new_n4, new_n5)
                
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / T):
                        current_score = new_score
                    else:
                        points[idx] = old_pos
            
            # Cool down
            T = max(min_T, T * cooling)
        
        return best_points, best_score

    # Multi-start SA
    best_overall_score = -float('inf')
    best_overall_points = None
    
    # Start 1: 4x4 grid
    grid_pts = initial_grid()
    n4, n5 = score_points(grid_pts)
    grid_score = evaluate(n4, n5)
    if grid_score > best_overall_score:
        best_overall_score = grid_score
        best_overall_points = grid_pts
    
    # Run SA from multiple starts
    starts = [
        initial_grid(),
        structured_config(),
        random_config(30),
        random_config(20),
    ]
    
    remaining_time = deadline - time.time()
    if remaining_time < 5:
        # Not enough time, return best so far
        return best_overall_points
    
    time_per_start = remaining_time / len(starts)
    
    for start_pts in starts:
        if time.time() > deadline - 5:
            break
        bp, bs = sa_search(start_pts, time_per_start)
        if bs > best_overall_score:
            best_overall_score = bs
            best_overall_points = bp
    
    # Final validation: ensure no 5 collinear
    fn4, fn5 = score_points(best_overall_points)
    if fn5 > 0:
        # Fallback to grid
        best_overall_points = initial_grid()
    
    return best_overall_points

if __name__ == "__main__":
    import time
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")
    print(pts)
    print(len(pts))
    
    # Verify
    line_points = defaultdict(set)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            key = canonical_line(pts[i], pts[j])
            line_points[key].add(i)
            line_points[key].add(j)
    
    count_4 = 0
    count_5 = 0
    max_collinear = 0
    for k, v in line_points.items():
        if len(v) == 4:
            count_4 += 1
        elif len(v) >= 5:
            count_5 += 1
        max_collinear = max(max_collinear, len(v))
    
    print("Max collinear:", max_collinear)
    print("4-lines:", count_4)
    print("5+-lines:", count_5)
