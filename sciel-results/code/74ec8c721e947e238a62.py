"""Improved program: Structural change to a grid-based initialization with targeted perturbation
and simulated annealing with a faster evaluation proxy (number of collinear triples)
switching to exact score for final verification and refinement."""
import random
import math
import time

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0  # seconds (evaluator kills at 150s, keep safety margin)

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    # p1, p2, p3 are tuples (x, y)
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _count_three_collinear(points):
    """Count the number of triples of collinear points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2  # i and j
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _max_3_free_subset(points):
    """Exact branch and bound for the largest subset with no three collinear.
    points: list of tuples (x, y). Returns the size of the largest 3-free subset."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute collinearity pairs
    collinearity = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinearity[i][j] = mask
            collinearity[j][i] = mask
    
    best = [0]
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        
        # Upper bound
        if size + (n - i) <= best[0]:
            return
        
        # Option 1: exclude point i
        backtrack(i + 1, subset_mask, size)
        
        # Option 2: include point i
        can_add = True
        temp_mask = subset_mask
        while temp_mask:
            a_bit = temp_mask & (-temp_mask)
            a = a_bit.bit_length() - 1
            if collinearity[a][i] & subset_mask:
                can_add = False
                break
            temp_mask ^= a_bit
        
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    # Generate initial configuration: Grid-based
    # Use a 5x4 grid, scaled and perturbed
    # Grid coordinates
    grid_points = []
    for x in range(5):
        for y in range(4):
            grid_points.append((x * 10, y * 10))
    
    # Shuffle and take 20
    random.shuffle(grid_points)
    points = list(grid_points[:20])
    
    # Perturb to break 4-collinearities
    # We need to ensure no 4 are collinear.
    # Grid has many 4-collinear (rows, cols).
    # We will randomly perturb points until valid.
    
    def random_point_near(p):
        return (p[0] + random.randint(-5, 5), p[1] + random.randint(-5, 5))
    
    # Initial fix: Move points to random locations if grid is too structured?
    # Actually, let's start with random points but constrained to a smaller box to increase density of collinearities.
    # Box [-20, 20]
    
    points = []
    for _ in range(20):
        p = (random.randint(-20, 20), random.randint(-20, 20))
        while p in points:
            p = (random.randint(-20, 20), random.randint(-20, 20))
        points.append(p)
    
    # Ensure no four collinear
    attempts = 0
    while _has_four_collinear(points) and attempts < 1000:
        idx = random.randint(0, 19)
        old = points[idx]
        new = (random.randint(-20, 20), random.randint(-20, 20))
        if new not in points:
            points[idx] = new
        attempts += 1
    
    if _has_four_collinear(points):
        # Fallback to random
        points = []
        for _ in range(20):
            p = (random.randint(-100, 100), random.randint(-100, 100))
            while p in points:
                p = (random.randint(-100, 100), random.randint(-100, 100))
            points.append(p)
        while _has_four_collinear(points):
            idx = random.randint(0, 19)
            points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
            while points[idx] in points:
                points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
    
    # Simulated Annealing
    # Objective: Maximize number of collinear triples (proxy for minimizing 3-free subset)
    # Constraint: No 4 collinear.
    
    current_triples = _count_three_collinear(points)
    best_triples = current_triples
    best_points = list(points)
    best_exact_score = _max_3_free_subset(points)
    
    temperature = 10.0
    cooling_rate = 0.995
    min_temperature = 0.1
    
    while temperature > min_temperature and _time_left() > 10.0:
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Try a random new point in a slightly larger box
        new_point = (random.randint(-25, 25), random.randint(-25, 25))
        if new_point in points:
            continue
        
        points[idx] = new_point
        
        if _has_four_collinear(points):
            points[idx] = old_point
            continue
        
        new_triples = _count_three_collinear(points)
        
        # Accept if better (more triples) or with probability
        delta = new_triples - current_triples
        if delta > 0:
            current_triples = new_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = list(points)
                # Update exact score for best
                # Only compute exact score if we have time
                if _time_left() > 20.0:
                    exact_score = _max_3_free_subset(points)
                    if exact_score < best_exact_score:
                        best_exact_score = exact_score
        else:
            if random.random() < math.exp(delta / temperature):
                current_triples = new_triples
        
        temperature *= cooling_rate
    
    # Final refinement: Try to improve exact score directly
    # Use the best points found so far
    points = list(best_points)
    current_exact = _max_3_free_subset(points)
    
    # Local search on exact score
    for _ in range(100):
        if _time_left() < 5.0:
            break
        idx = random.randint(0, 19)
        old_point = points[idx]
        new_point = (old_point[0] + random.randint(-2, 2), old_point[1] + random.randint(-2, 2))
        if new_point in points or new_point[0] < -100 or new_point[0] > 100 or new_point[1] < -100 or new_point[1] > 100:
            continue
        points[idx] = new_point
        if _has_four_collinear(points):
            points[idx] = old_point
            continue
        new_exact = _max_3_free_subset(points)
        if new_exact <= current_exact:
            current_exact = new_exact
            if current_exact < best_exact_score:
                best_exact_score = current_exact
                best_points = list(points)
        else:
            points[idx] = old_point
    
    return [list(p) for p in best_points]
