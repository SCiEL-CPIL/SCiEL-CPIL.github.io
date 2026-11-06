import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def evaluate(points):
        pts = np.array(points, dtype=np.int32)
        n = len(pts)
        num_triples = 0
        max_collinear = 0
        
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                if j + 1 >= n:
                    continue
                rest = pts[j+1:]
                cross = dx * (rest[:, 1] - y1) - dy * (rest[:, 0] - x1)
                count = np.sum(cross == 0) + 2
                
                if count > max_collinear:
                    max_collinear = count
                if count > 2:
                    num_triples += (count - 2)
                    
        return num_triples, max_collinear

    best_points = None
    best_score = -1

    def search():
        nonlocal best_points
        nonlocal best_score
        
        for restart in range(10000):
            if time.time() > deadline:
                break
                
            points = []
            seen = set()
            while len(points) < 20:
                p = random_point()
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    points.append(p)
            
            triples, max_col = evaluate(points)
            if max_col > 3:
                continue

            if triples > best_score:
                best_score = triples
                best_points = [p[:] for p in points]
                
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                idx = random.randint(0, 19)
                old_p = points[idx][:]
                new_p = [old_p[0] + random.randint(-10, 10), old_p[1] + random.randint(-10, 10)]
                new_p[0] = max(-100, min(100, new_p[0]))
                new_p[1] = max(-100, min(100, new_p[1]))
                
                if any((points[i][0] == new_p[0] and points[i][1] == new_p[1]) for i in range(20) if i != idx):
                    continue
                
                points[idx] = new_p
                new_triples, new_max_col = evaluate(points)
                
                if new_max_col > 3:
                    points[idx] = old_p
                    continue
                    
                if new_triples >= triples:
                    triples = new_triples
                    if triples > best_score:
                        best_score = triples
                        best_points = [p[:] for p in points]
                else:
                    points[idx] = old_p

    search()
    
    if best_points is None:
        return [[x, x*x] for x in range(-10, 10)]
        
    return best_points
