import random
import time

def construct():
    start_time = time.time()
    deadline = 120  # Safety margin
    
    def get_collinear_triples(points):
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    if (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(x3[0]-x1[0]) == 0:
                        triples.append((i, j, k))
        return triples
        
    # Corrected:
    def get_collinear_triples(points):
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    if (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0:
                        triples.append((i, j, k))
        return triples
        
    def count_collinear_triples(points):
        return len(get_collinear_triples(points))
        
    # Initialize with random points
    points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
    while len(set(tuple(p) for p in points)) < 20:
        points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
        
    best_score = count_collinear_triples(points)
    best_points = [p[:] for p in points]
    current_score = best_score
    current_points = [p[:] for p in points]
    
    temperature = 100
    cooling_rate = 0.99
    
    while time.time() - start_time < deadline:
        # Move one point to a random location
        i = random.randint(0, 19)
        old_p = current_points[i]
        current_points[i] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        if len(set(tuple(p) for p in current_points)) < 20:
            current_points[i] = old_p
            continue
            
        new_score = count_collinear_triples(current_points)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[i] = old_p
            
        temperature *= cooling_rate
        
    return best_points
