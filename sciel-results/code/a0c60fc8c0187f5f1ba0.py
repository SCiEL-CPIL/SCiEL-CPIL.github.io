import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        # Sign convention: first non-zero positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def calc_score(points):
        """
        Maximize the number of collinear triples.
        """
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        triples = 0
        for line, count in line_counts.items():
            if count == 3:
                triples += 1
        return triples

    def is_valid(points):
        """Check distinctness, bounds, and no 4 collinear."""
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        for line, count in line_counts.items():
            if count >= 4:
                return False
        return True

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    # Initial configuration: Random
    current_points = []
    while len(current_points) < 20:
        p = random_point()
        if p not in current_points:
            current_points.append(p)
    
    # Ensure initial validity? Random points might have 4 collinear.
    # Let's try to fix it or restart.
    if not is_valid(current_points):
        # Try a few times to find a valid random start
        for _ in range(100):
            current_points = []
            while len(current_points) < 20:
                p = random_point()
                if p not in current_points:
                    current_points.append(p)
            if is_valid(current_points):
                break
    
    if not is_valid(current_points):
        # Fallback to parabola if random fails (unlikely)
        return [[x, x * x] for x in range(-10, 10)]
        
    current_score = calc_score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx][:]
        
        # Generate a new point
        # Try to move it to a position that creates more collinear triples
        # Simple: Random restart of that point
        new_p = random_point()
        
        # Try to improve by placing it on a line with 2 existing points
        # Find pairs of points in current_points (excluding idx)
        # and check if the line segment or extension has a good spot?
        # Or just check if placing new_p creates a triple.
        
        # Heuristic: Try to form a triple with two existing points.
        improved = False
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                p1 = current_points[i]
                p2 = current_points[j]
                # Line through p1, p2
                # Check if we can find a point on this line within bounds
                # that is not already in current_points (except p1, p2)
                # and is distinct from old_p
                # Actually, we just need to pick a point on the line.
                # To avoid 4 collinear, we must ensure no other point is on this line.
                # So this line must currently have exactly 2 points (p1, p2) in the set.
                
                # Check count of points on line p1-p2
                line = get_line(p1, p2)
                # Count
                cnt = 0
                for k in range(20):
                    if k == i or k == j:
                        continue
                    if get_line(p1, current_points[k]) == line:
                        cnt += 1
                if cnt == 0:
                    # Line has only p1 and p2.
                    # We can place a new point on this line.
                    # Generate a point on the line.
                    # p1 + t * (p2 - p1)
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Try small integer steps
                    found = False
                    for t in range(-10, 11):
                        if t == 0 or t == 1:
                            continue
                        nx = p1[0] + t * dx
                        ny = p1[1] + t * dy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            new_candidate = [nx, ny]
                            if new_candidate not in current_points:
                                # Check if this creates 4 collinear?
                                # We know line has 2 points. Adding one makes 3.
                                # So no 4 collinear on this line.
                                # But could it create 4 collinear on another line?
                                # The new point is on line L.
                                # It might form a line with p1 and some other point? No, p1 is on L.
                                # It might form a line with p2 and some other point? No.
                                # It might form a line with two other points not on L?
                                # Yes.
                                # So we must check validity of the whole set.
                                test_points = current_points[:]
                                test_points[idx] = new_candidate
                                if is_valid(test_points):
                                    test_score = calc_score(test_points)
                                    if test_score > current_score:
                                        current_points = test_points
                                        current_score = test_score
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_points = [p[:] for p in current_points]
                                        improved = True
                                        break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            # Try random move
            new_p = random_point()
            if new_p not in current_points:
                test_points = current_points[:]
                test_points[idx] = new_p
                if is_valid(test_points):
                    test_score = calc_score(test_points)
                    if test_score > current_score:
                        current_points = test_points
                        current_score = test_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                    else:
                        # Accept with some probability? No, we want to maximize.
                        # Maybe accept if equal to escape local optima?
                        if test_score == current_score and random.random() < 0.1:
                            current_points = test_points
                            current_score = test_score
    
    # Final check
    if is_valid(best_points):
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
