import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def count_triples(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def has_four_collinear(points):
        n = len(points)
        # Efficient check: for each pair, count points on that line
        # But n=20 is small, O(n^4) is 48450 checks, very fast.
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if is_collinear(points[i], points[j], points[k]) and is_collinear(points[i], points[j], points[l]):
                            return True
        return False

    def generate_initial_points():
        # Start with a grid-like structure to have many potential triples
        # 4x5 grid
        pts = []
        for x in range(4):
            for y in range(5):
                pts.append([x * 10, y * 10])
        # Scale to fit bounds and add some randomness to break 4-collinearities immediately
        # Actually, let's just use random points but biased towards a grid
        random.shuffle(pts)
        return pts

    # We want to MAXIMIZE triples, so we accept moves that INCREASE triples
    # This is a maximization problem with a constraint (no 4 collinear)
    
    points = generate_initial_points()
    # Ensure distinctness and bounds
    # The grid points are distinct.
    # Check if initial has 4-collinear. Grid does.
    # We need to fix it.
    
    # Strategy: Simulated Annealing / Hill Climbing to MAXIMIZE triples
    # Penalty for 4-collinear is infinite.
    
    def get_score(points):
        if has_four_collinear(points):
            return -1000000
        return count_triples(points)

    current_points = points
    current_score = get_score(current_points)
    best_points = current_points
    best_score = current_score
    
    temperature = 50.0
    cooling = 0.99
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        # Move point slightly to stay near grid structure? Or random?
        # Random is safer to explore.
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        
        if new_p in current_points:
            continue
            
        test_points = current_points.copy()
        test_points[idx] = new_p
        
        new_score = get_score(test_points)
        
        # We want to maximize score
        delta = new_score - current_score
        
        if delta > 0:
            current_points = test_points
            current_score = new_score
            if new_score > best_score:
                best_points = test_points
                best_score = new_score
        else:
            # Accept worse move with probability
            # If new_score is -1000000 (invalid), delta is huge negative, prob ~ 0
            if delta == 0:
                if random.random() < 0.5:
                    current_points = test_points
            else:
                prob = pow(2.718281828, delta / temperature) # Note: delta is negative, so exp(neg/T) < 1
                if random.random() < prob:
                    current_points = test_points
                    current_score = new_score
        
        temperature *= cooling
        if temperature < 0.5:
            temperature = 50.0 # Restart

    return best_points
