import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        # Normalize sign
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        triples_with_idx = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
                        
        best_score = 0
        
        # Order by degree (number of triples) to prune earlier
        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 2:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            if current_size + (n - step) <= best_score:
                return
            idx = order[step]
            
            # Branch 1: Do not include idx
            bbb(step + 1, current_mask, current_size)
            
            # Branch 2: Include idx
            valid = True
            for a, b in triples_with_idx[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def random_valid_points():
        while True:
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                return pts

    best_points = None
    best_val = float('inf')
    
    # Phase 1: Find a good initial solution with multiple random starts and hill climbing
    for _ in range(20):
        if elapsed() > deadline - 20:
            break
        pts = random_valid_points()
        current_val = calculate_score(pts)
        if current_val < best_val:
            best_val = current_val
            best_points = pts[:]
        
        # Hill Climbing
        improved = True
        while improved and elapsed() < deadline - 10:
            improved = False
            for i in range(20):
                if elapsed() > deadline - 10:
                    break
                old_p = pts[i]
                # Try a few random moves
                for _ in range(5):
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))
                    if new_p in pts or new_p == old_p:
                        continue
                    if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                        continue
                    
                    # Check validity
                    valid = True
                    # Quick check: does new_p form a 4-collinear with any 3 existing?
                    # Or just full check
                    test_pts = pts[:]
                    test_pts[i] = new_p
                    if has_4_collinear(test_pts):
                        continue
                    
                    new_val = calculate_score(test_pts)
                    if new_val < current_val:
                        current_val = new_val
                        pts = test_pts
                        improved = True
                        break
                    elif new_val == current_val and random.random() < 0.1:
                        # Plateau move
                        pts = test_pts
                        improved = True
                        break
                if improved:
                    break
        
        if current_val < best_val:
            best_val = current_val
            best_points = pts[:]

    # Phase 2: Simulated Annealing
    if best_points is not None and elapsed() < deadline - 5:
        current_pts = best_points[:]
        current_val = best_val
        T = 2.0 # Initial temperature
        cooling_rate = 0.99
        
        while elapsed() < deadline - 2:
            T *= cooling_rate
            if T < 0.01:
                break
            
            # Generate neighbor: move one point
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            # Bias towards local moves to explore neighborhood
            if random.random() < 0.5:
                new_p = (old_p[0] + random.randint(-10, 10), old_p[1] + random.randint(-10, 10))
            else:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
            
            if new_p in current_pts or new_p == old_p:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
            
            test_pts = current_pts[:]
            test_pts[idx] = new_p
            
            if has_4_collinear(test_pts):
                continue
            
            new_val = calculate_score(test_pts)
            
            if new_val < current_val:
                current_val = new_val
                current_pts = test_pts
                if current_val < best_val:
                    best_val = current_val
                    best_points = current_pts[:]
            else:
                # Accept with probability exp(-delta / T)
                delta = new_val - current_val
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_val = new_val
                        current_pts = test_pts

    # Fallback
    if best_points is None:
        best_points = random_valid_points()

    return [[x, y] for x, y in best_points]
