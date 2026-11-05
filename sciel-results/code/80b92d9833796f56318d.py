import random
import math

def construct():
    random.seed(42)
    
    # Pool of candidate points
    # Use a small grid to increase density of collinearities
    pool = [[x, y] for x in range(-10, 10) for y in range(-10, 10)]
    
    # Helper to check if a set of points has any 4 collinear
    def is_valid_no_4(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = pts[i], pts[j]
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = p1[0]*p2[1] - p2[0]*p1[1]
                count = 0
                for k in range(n):
                    p3 = pts[k]
                    if A*p3[0] + B*p3[1] + C == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    # Helper to count 3-point lines
    def count_triples(pts):
        n = len(pts)
        triples = 0
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = pts[i], pts[j]
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = p1[0]*p2[1] - p2[0]*p1[1]
                count = 0
                for k in range(n):
                    p3 = pts[k]
                    if A*p3[0] + B*p3[1] + C == 0:
                        count += 1
                if count == 3:
                    triples += 1
        return triples

    # Initialize points
    points = []
    attempts = 0
    while len(points) < 20 and attempts < 10000:
        p = random.choice(pool)
        if p not in points:
            points.append(p)
            if not is_valid_no_4(points):
                points.pop()
        attempts += 1
        
    if len(points) < 20:
        # Fallback: simple greedy valid fill
        points = []
        for _ in range(20):
            found = False
            for p in random.sample(pool, 100):
                if p not in points:
                    points.append(p)
                    if is_valid_no_4(points):
                        found = True
                        break
                    else:
                        points.pop()
            if not found:
                for p in pool:
                    if p not in points:
                        points.append(p)
                        break

    current_triples = count_triples(points)
    best_points = points[:]
    best_triples = current_triples
    
    # Simulated Annealing
    T = 5.0
    steps = 10000
    
    for step in range(steps):
        T = T * 0.999
        i = random.randint(0, 19)
        old_p = points[i]
        
        # Generate new point
        new_p = None
        for _ in range(10):
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            cand = [old_p[0] + dx, old_p[1] + dy]
            if -10 <= cand[0] <= 9 and -10 <= cand[1] <= 9:
                if cand not in points:
                    new_p = cand
                    break
        if new_p is None:
            new_p = random.choice(pool)
            if new_p in points:
                continue
        
        # Check validity of new set
        new_points = points[:]
        new_points[i] = new_p
        
        # Check if new_p creates a 4-collinearity
        valid = True
        for j in range(20):
            if j == i: continue
            p_j = new_points[j]
            A = new_p[1] - p_j[1]
            B = p_j[0] - new_p[0]
            C = new_p[0]*p_j[1] - p_j[0]*new_p[1]
            count = 0
            for k in range(20):
                if k == i: continue
                p_k = new_points[k]
                if A*p_k[0] + B*p_k[1] + C == 0:
                    count += 1
            if count >= 4:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate change in triples
        old_count = 0
        for j in range(20):
            if j == i: continue
            p_j = points[j]
            A = old_p[1] - p_j[1]
            B = p_j[0] - old_p[0]
            C = old_p[0]*p_j[1] - p_j[0]*old_p[1]
            count = 0
            for k in range(20):
                p_k = points[k]
                if A*p_k[0] + B*p_k[1] + C == 0:
                    count += 1
            if count == 3:
                old_count += 1
        
        new_count = 0
        for j in range(20):
            if j == i: continue
            p_j = new_points[j]
            A = new_p[1] - p_j[1]
            B = p_j[0] - new_p[0]
            C = new_p[0]*p_j[1] - p_j[0]*new_p[1]
            count = 0
            for k in range(20):
                p_k = new_points[k]
                if A*p_k[0] + B*p_k[1] + C == 0:
                    count += 1
            if count == 3:
                new_count += 1
                
        delta_triples = new_count - old_count
        
        if delta_triples > 0 or random.random() < math.exp(delta_triples / T):
            points = new_points
            current_triples += delta_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = points[:]
        
    return best_points
