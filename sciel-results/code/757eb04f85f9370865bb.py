"""Rewritten program: structural change to a local-search / simulated-annealing style
optimization over point configurations, aiming to minimize the largest 3-free subset
while respecting the 'no four collinear' constraint."""
import random
import math
import time
from itertools import combinations

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

def _max_3_free_subset(points):
    """Exact branch and bound for the largest subset with no three collinear.
    points: list of tuples (x, y). Returns the size of the largest 3-free subset."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute collinearity: coll[i] is a bitmask of points collinear with i
    coll = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            coll[i] |= (1 << j)
    
    # Actually, we need to know for each pair (i,j) which k are collinear with both.
    # Let's precompute a 3D structure or use a different B&B approach.
    
    # For n=20, a simpler approach: try to build the largest subset greedily with backtracking.
    # Or use the known result: we want to find the largest subset S such that no 3 points in S are collinear.
    
    # Let's use a branch and bound where we process points in a fixed order.
    # State: index i, current subset S (as a bitmask), current size.
    # We want to maximize the size of S with no 3 collinear.
    
    # Precompute for each point i, the set of points collinear with it.
    # coll_mask[i] = bitmask of points j such that i, j, and some k are collinear? No.
    # We need: for a given subset S, check if any 3 are collinear.
    
    # Better: precompute for each pair (i, j), the set of points k that are collinear with i and j.
    # collinearity[i][j] = bitmask of k (k != i, k != j) such that i, j, k are collinear.
    
    collinearity = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if _is_collinear(points[i], points[j], points[k]):
                    collinearity[i][j] |= (1 << k)
    
    # Now, B&B:
    # We process points in order 0, 1, ..., n-1.
    # At each step, we decide to include or exclude point i.
    # If we include point i, we must ensure that for all pairs (a, b) in the current subset S,
    # point i is not collinear with a and b. That is, collinearity[a][b] & (1 << i) == 0.
    
    best = [0]
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        
        # Upper bound: size + (n - i)
        if size + (n - i) <= best[0]:
            return
        
        # Option 1: exclude point i
        backtrack(i + 1, subset_mask, size)
        
        # Option 2: include point i
        # Check if point i can be added to the current subset
        can_add = True
        # We need to check all pairs in the current subset
        # For efficiency, we can check: for each a in subset, is there a b in subset such that collinearity[a][b] has bit i set?
        # Equivalent: for each a in subset, (collinearity[a][i] & subset_mask) != 0? No.
        # collinearity[a][b] & (1 << i) != 0 means i is collinear with a and b.
        # So we need: for all a, b in subset, collinearity[a][b] & (1 << i) == 0.
        # This is expensive. Let's use a different check.
        # When we add i, we need to ensure no two existing points a, b in subset are collinear with i.
        # That is, for all a in subset, the set of points b in subset that are collinear with a and i must be empty.
        # collinearity[a][i] gives the set of points collinear with a and i.
        # So we need: (collinearity[a][i] & subset_mask) == 0 for all a in subset.
        
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

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = 2  # i and j
            for k in range(n):
                if k == i or k == j:
                    continue
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _count_three_collinear(points):
    """Count the number of triples of collinear points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k > j:
                    if _is_collinear(points[i], points[j], points[k]):
                        count += 1
    return count

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with a configuration that has many collinear triples but no four collinear.
    # Idea: use a grid-like structure with some perturbation.
    # Or start with the parabola and try to move points to create more collinearities.
    
    # Let's start with a random configuration and use local search.
    # Objective: minimize the largest 3-free subset (i.e., maximize the "collinearity density").
    # Constraint: no four collinear.
    
    # Initial configuration: points on a grid with some randomness.
    # Let's try to place points on lines to create many collinear triples.
    
    # Strategy: start with 20 points, try to move them to create more collinear triples.
    # Use simulated annealing or local search.
    
    # Initial points: random within [-100, 100]
    random.seed(42)
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))
    
    # Generate initial configuration
    points = []
    for _ in range(20):
        p = random_point()
        while p in points:
            p = random_point()
        points.append(p)
    
    # Ensure no four collinear in initial config
    while _has_four_collinear(points):
        # Move a random point
        idx = random.randint(0, 19)
        points[idx] = random_point()
        while points[idx] in points:
            points[idx] = random_point()
    
    # Evaluate initial score
    current_score = _max_3_free_subset(points)
    best_score = current_score
    best_points = list(points)
    
    # Local search: try to move a point to a new position to reduce the score
    # (i.e., make the configuration more "collinear" in a way that reduces the largest 3-free subset)
    
    # Actually, we want to MINIMIZE the largest 3-free subset.
    # So we want to make it harder to find a large 3-free subset.
    # This means we want to create more collinear triples, so that any large subset will contain a collinear triple.
    
    # Let's use a different approach: start with a configuration known to have a small 3-free subset.
    # For example, points on a few lines, but with no four collinear.
    
    # Let's try: 4 lines with 5 points each, but perturb to avoid four collinear.
    # Line 1: y = 0, x = -10, -5, 0, 5, 10
    # Line 2: y = 1, x = -9, -4, 1, 6, 11
    # Line 3: y = 2, x = -8, -3, 2, 7, 12
    # Line 4: y = 3, x = -7, -2, 3, 8, 13
    # But these have 5 collinear points, which violates the constraint.
    
    # So we need to perturb. Let's use a different structure.
    
    # Let's try a "near-grid" with perturbations to create collinear triples.
    
    # For now, let's just do a simple local search:
    # 1. Start with a random valid configuration.
    # 2. Try to move each point to a nearby position.
    # 3. If the new configuration is valid (no four collinear) and has a lower score, accept it.
    # 4. Use simulated annealing to escape local minima.
    
    temperature = 10.0
    cooling_rate = 0.99
    min_temperature = 0.1
    
    while temperature > min_temperature and _time_left() > 5.0:
        # Try to move a random point
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Try a few random new positions
        found_improvement = False
        for _ in range(10):
            new_point = (old_point[0] + random.randint(-5, 5), old_point[1] + random.randint(-5, 5))
            if new_point[0] < -100 or new_point[0] > 100 or new_point[1] < -100 or new_point[1] > 100:
                continue
            if new_point in points:
                continue
            
            points[idx] = new_point
            
            if _has_four_collinear(points):
                points[idx] = old_point
                continue
            
            new_score = _max_3_free_subset(points)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = list(points)
                found_improvement = True
                break
            elif new_score == current_score:
                # Accept with some probability to escape local minima
                if random.random() < 0.1:
                    current_score = new_score
                    break
            else:
                # Worse, accept with probability exp(-delta / temperature)
                delta = new_score - current_score
                if random.random() < math.exp(-delta / temperature):
                    current_score = new_score
                    break
        
        if not found_improvement:
            points[idx] = old_point
        
        temperature *= cooling_rate
    
    # Return the best configuration found
    return [list(p) for p in best_points]
