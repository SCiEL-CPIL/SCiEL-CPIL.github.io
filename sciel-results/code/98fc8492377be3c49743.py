import random
import time

def construct():
    # Start with the 18-point grid configuration
    # Block 1: x in {-10, -5, 0}, y in {0, 1, 2}
    # Block 2: x in {5, 10, 15}, y in {3, 4, 5}
    points = []
    for x in [-10, -5, 0]:
        for y in [0, 1, 2]:
            points.append([x, y])
    for x in [5, 10, 15]:
        for y in [3, 4, 5]:
            points.append([x, y])
    
    # Add 2 more points
    # Try to place them such that they don't create 4-collinear
    # Randomly pick from a wider range
    random.seed(42)
    for _ in range(2):
        while True:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            new_point = [x, y]
            # Check if it's already in points
            if new_point in points:
                continue
            # Check if adding it creates a 4-collinear
            if not creates_four_collinear(points, new_point):
                points.append(new_point)
                break
    
    # Local search
    start_time = time.time()
    timeout = 140  # Leave 10s margin
    best_points = points[:]
    best_score = calculate_heuristic_score(best_points)
    
    while time.time() - start_time < timeout:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_point = best_points[idx]
        
        # Generate a new position
        # Bias towards existing lines or nearby points to create collinearities
        # Simple random
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        new_point = [new_x, new_y]
        
        if new_point == old_point:
            continue
        if new_point in best_points:
            continue
            
        # Temporarily move
        best_points[idx] = new_point
        
        if creates_four_collinear(best_points, None):
            best_points[idx] = old_point
            continue
            
        score = calculate_heuristic_score(best_points)
        if score < best_score:
            best_score = score
            # Optional: update best_points if we want to track global best
            # But we are modifying best_points in place, so it's already updated
        else:
            # Revert if not better (greedy hill climbing)
            best_points[idx] = old_point
            
    return best_points

def creates_four_collinear(points, new_point=None):
    """
    Check if the set of points contains 4 collinear points.
    If new_point is provided, check if adding it to points creates a 4-collinear.
    """
    all_points = points[:]
    if new_point:
        all_points.append(new_point)
        
    n = len(all_points)
    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            p1 = all_points[i]
            p2 = all_points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                p3 = all_points[k]
                # Check if p3 is on the line defined by p1, p2
                # Cross product
                cross = dx * (p3[1] - p1[1]) - dy * (p3[0] - p1[0])
                if cross == 0:
                    count += 1
                    if count >= 4:
                        return True
    return False

def calculate_heuristic_score(points):
    """
    Heuristic: Count the number of collinear triples.
    More triples -> smaller max independent set.
    """
    triples = 0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            count = 2
            for k in range(j + 1, n):
                p3 = points[k]
                cross = dx * (p3[1] - p1[1]) - dy * (p3[0] - p1[0])
                if cross == 0:
                    count += 1
            if count == 3:
                triples += 1
            elif count > 3:
                # Should not happen if valid, but if it does, it's very good for reducing subset size?
                # Actually, 4 collinear is invalid.
                pass
    return -triples # Minimize score means maximize triples

