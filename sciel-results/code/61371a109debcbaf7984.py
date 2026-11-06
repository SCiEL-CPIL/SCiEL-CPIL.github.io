import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def are_3_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_3_collinear(points[i], points[j], points[k]) and \
                           are_3_collinear(points[i], points[j], points[l]):
                            return True
        return False

    def count_triples(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_3_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def generate_random_set():
        while True:
            pts = []
            seen = set()
            for _ in range(20):
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in seen:
                        seen.add((x, y))
                        pts.append((x, y))
                        break
            if not has_4_collinear(pts):
                return pts

    best_pts = generate_random_set()
    best_score = count_triples(best_pts)
    current_pts = best_pts[:]
    current_score = best_score
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_p = current_pts[idx]
        
        # Try a few random moves to find a better one
        for _ in range(10):
            new_p = (random.randint(-100, 100), random.randint(-100, 100))
            if new_p == old_p or new_p in current_pts:
                continue
                
            new_pts = current_pts[:]
            new_pts[idx] = new_p
            
            if has_4_collinear(new_pts):
                continue
                
            new_score = count_triples(new_pts)
            
            if new_score < current_score:
                current_pts = new_pts
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts[:]
                break # Accept first improvement
            elif new_score == current_score and random.random() < 0.1:
                # Accept some equal moves to escape local minima
                current_pts = new_pts
                break

    return [[x, y] for x, y in best_pts]
