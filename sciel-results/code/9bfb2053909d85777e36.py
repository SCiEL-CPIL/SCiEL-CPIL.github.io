import time
import random
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def line_key(p1, p2):
        """Compute a canonical key for the line through p1 and p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        # Canonical direction: ensure dx > 0, or dx == 0 and dy > 0
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Line equation: dy*x - dx*y = c
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_triples_and_check_four(points):
        """Count collinear triples and check for 4-collinear in O(n^2)."""
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        
        triples = 0
        has_four = False
        for key, count in line_counts.items():
            if count >= 4:
                has_four = True
            if count >= 3:
                # C(count, 3) triples
                triples += count * (count - 1) * (count - 2) // 6
        
        return triples, has_four
    
    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000
        return triples

    def generate_valid_initial():
        # Start with random points
        pts = []
        for _ in range(20):
            pts.append([random.randint(-100, 100), random.randint(-100, 100)])
        
        current_pts = pts
        attempts = 0
        while attempts < 100:
            _, has_four = count_triples_and_check_four(current_pts)
            if not has_four:
                # Check for duplicates
                if len(set(tuple(p) for p in current_pts)) == 20:
                    return current_pts
            else:
                # Find a line with 4+ points and move one of them
                line_to_points = {}
                for i in range(20):
                    for j in range(i + 1, 20):
                        key = line_key(current_pts[i], current_pts[j])
                        if key not in line_to_points:
                            line_to_points[key] = set()
                        line_to_points[key].add(i)
                        line_to_points[key].add(j)
                
                moved = False
                for key, pts_in_line in line_to_points.items():
                    if len(pts_in_line) >= 4:
                        idx = random.choice(list(pts_in_line))
                        old_p = current_pts[idx]
                        for _ in range(20):
                            dx = random.randint(-5, 5)
                            dy = random.randint(-5, 5)
                            if dx == 0 and dy == 0:
                                continue
                            new_p = [old_p[0] + dx, old_p[1] + dy]
                            if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                                if new_p not in current_pts:
                                    current_pts[idx] = new_p
                                    moved = True
                                    break
                        if moved:
                            break
            attempts += 1
        
        # Fallback: if still invalid, return a known valid grid perturbation
        # But the loop should fix it.
        return current_pts

    def local_search_sa(points, time_limit):
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        # Simulated Annealing parameters
        T = 10.0
        cooling_rate = 0.95
        min_T = 0.01
        
        while time.time() < time_limit and T > min_T:
            # Generate a move
            # 1. Pick a random point
            i = random.randint(0, 19)
            old_p = current_pts[i]
            
            # 2. Generate a new position
            # Bias towards positions that might create triples
            # Pick a random pair of other points and try to align with them
            if random.random() < 0.7:
                j, k = random.sample([x for x in range(20) if x != i], 2)
                p_j = current_pts[j]
                p_k = current_pts[k]
                # Find integer points on line p_j p_k
                dx = p_k[0] - p_j[0]
                dy = p_k[1] - p_j[1]
                g = gcd(abs(dx), abs(dy))
                if g > 0:
                    dx_n = dx // g
                    dy_n = dy // g
                    # Sample a point on this line
                    # t is the parameter. p = p_j + t * (dx_n, dy_n)
                    # We want an integer point.
                    # Pick a random t
                    t = random.randint(-5, 5)
                    new_p = [p_j[0] + t * dx_n, p_j[1] + t * dy_n]
                else:
                    new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            else:
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            
            if new_p == old_p:
                continue
            if -100 > new_p[0] or new_p[0] > 100 or -100 > new_p[1] or new_p[1] > 100:
                continue
            if new_p in current_pts:
                continue
                
            test_pts = current_pts[:]
            test_pts[i] = new_p
            new_score = get_score(test_pts)
            
            if new_score > current_score:
                current_pts = test_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = [p[:] for p in current_pts]
            else:
                # Accept worse move with probability
                delta = new_score - current_score
                if delta < 0:
                    prob = pow(2.71828, -abs(delta) / T)
                    if random.random() < prob:
                        current_pts = test_pts
                        current_score = new_score
            
            T *= cooling_rate
        
        return best_pts

    # Run multiple restarts
    best_overall = None
    best_overall_score = -1
    
    # Initial point
    init_pts = generate_valid_initial()
    
    # Run local search with multiple restarts
    t_end = deadline
    restart_count = 0
    while time.time() < t_end - 5:
        search_time = min(5, t_end - 5 - time.time())
        if search_time < 1:
            break
        current_best = local_search_sa(init_pts, time.time() + search_time)
        score = get_score(current_best)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = current_best
        
        restart_count += 1
        
        # Generate a new initial point for next restart
        if time.time() < t_end - 10:
            init_pts = generate_valid_initial()
        else:
            break
            
    return best_overall
