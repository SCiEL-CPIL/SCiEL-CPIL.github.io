import random
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    n = 20
    coord_range = 100
    
    def is_collinear(p1, p2, p3):
        return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]) == 0)
    
    def count_triples(points):
        triples = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        triples += 1
        return triples

    def has_four_collinear(points):
        # Check all pairs
        for i in range(n):
            for j in range(i+1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if is_collinear(points[i], points[j], points[k]):
                            count += 1
                if count >= 4:
                    return True
        return False

    def generate_initial():
        while True:
            points = []
            for _ in range(n):
                x = random.randint(-coord_range, coord_range)
                y = random.randint(-coord_range, coord_range)
                points.append([x, y])
            if len(set(map(tuple, points))) == n and not has_four_collinear(points):
                return points

    def local_search(points, time_budget):
        current_points = [p[:] for p in points]
        current_triples = count_triples(current_points)
        
        while time.time() - start_time < time_budget:
            # Pick a random point to move
            i = random.randint(0, n-1)
            old_point = current_points[i]
            
            # Try to move to a random new position
            new_point = [random.randint(-coord_range, coord_range), random.randint(-coord_range, coord_range)]
            if new_point == old_point:
                continue
                
            # Check if new_point creates 4 collinear
            # We remove old_point, then add new_point.
            # Check if new_point is collinear with any 3 points in the remaining set.
            
            others = [p for idx, p in enumerate(current_points) if idx != i]
            
            valid = True
            # For each pair (a, b) in others, if new_point is collinear with a, b,
            # check if any c in others (c != a, b) is collinear with a, b.
            for a_idx in range(len(others)):
                for b_idx in range(a_idx+1, len(others)):
                    a = others[a_idx]
                    b = others[b_idx]
                    if is_collinear(new_point, a, b):
                        # Check for third point c
                        for c_idx in range(len(others)):
                            if c_idx == a_idx or c_idx == b_idx:
                                continue
                            if is_collinear(a, b, others[c_idx]):
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Calculate new triples
            new_points = [p[:] for p in current_points]
            new_points[i] = new_point
            new_triples = count_triples(new_points)
            
            if new_triples > current_triples:
                current_points = new_points
                current_triples = new_triples
                
        return current_points

    # Initial points
    points = generate_initial()
    
    # Run local search
    best_points = local_search(points, time_limit)
    
    return best_points
